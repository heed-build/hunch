// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import CryptoKit
import XCTest

@testable import Hunch

final class ModelStoreTests: XCTestCase {
  private var root: URL!
  private var endpoint: URL!

  override func setUpWithError() throws {
    root = try temporaryDirectory()
    endpoint = root.appending(path: "endpoint")
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  private func hex(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

  /// Publishes `files` under the fake endpoint and returns a checkpoint describing them, with `corrupt` files given a
  /// wrong sha256.
  private func publish(_ files: [String: String], corrupt: Set<String> = []) throws -> Checkpoint {
    let checkpoint = Checkpoint(
      name: "fake", repository: "owner/fake", revision: "0123456789abcdef",
      files: files.keys.sorted().map { path in
        let data = Data(files[path]!.utf8)
        return Checkpoint.File(
          path, size: Int64(data.count), sha256: corrupt.contains(path) ? String(repeating: "0", count: 64) : hex(data))
      })
    for file in checkpoint.files {
      let url = checkpoint.url(for: file, endpoint: endpoint)
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data(files[file.path]!.utf8).write(to: url)
    }
    return checkpoint
  }

  func testSha256OfAFile() throws {
    let url = root.appending(path: "hello.txt")
    try Data("hello\n".utf8).write(to: url)
    XCTAssertEqual(
      try ModelStore.sha256(of: url), "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03")
  }

  func testPrepareDownloadsVerifiesAndMovesIntoPlace() async throws {
    let checkpoint = try publish(["model.safetensors": "weights", "tokenizer/tokenizer.json": "{}"])
    let destination = root.appending(path: "models/fake")
    let reports = Reports()
    let prepared = try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint) { reports.add($0) }
    XCTAssertEqual(prepared, destination)
    XCTAssertEqual(try String(contentsOf: destination.appending(path: "tokenizer/tokenizer.json"), encoding: .utf8), "{}")
    XCTAssertTrue(ModelStore.isPrepared(checkpoint, at: destination))
    XCTAssertNoThrow(try ModelStore.verify(checkpoint, at: destination))
    XCTAssertEqual(reports.last?.fractionCompleted, 1)
    // No staging directory is left behind.
    let leftovers = try FileManager.default.contentsOfDirectory(atPath: destination.deletingLastPathComponent().path)
    XCTAssertEqual(leftovers, ["fake"])
  }

  func testAPreparedDirectoryIsLeftAlone() async throws {
    let checkpoint = try publish(["model.safetensors": "weights"])
    let destination = root.appending(path: "models/fake")
    try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint)
    // Without the endpoint, only a no-op can succeed.
    try FileManager.default.removeItem(at: endpoint)
    try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint)
    XCTAssertTrue(ModelStore.isPrepared(checkpoint, at: destination))
  }

  func testAChecksumMismatchKeepsTheDestinationUntouched() async throws {
    let checkpoint = try publish(["a.json": "a", "model.safetensors": "weights"], corrupt: ["model.safetensors"])
    let destination = root.appending(path: "models/fake")
    do {
      try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint)
      XCTFail("expected a checksum mismatch")
    } catch let HunchError.checksumMismatch(file, _, actual) {
      XCTAssertEqual(file, "model.safetensors")
      XCTAssertEqual(actual, hex(Data("weights".utf8)))
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.appending(path: "models").path), [])
  }

  func testAnIncompleteDirectoryIsMovedAsideNotDeleted() async throws {
    let checkpoint = try publish(["model.safetensors": "weights"])
    let destination = root.appending(path: "models/fake")
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    try Data("mine".utf8).write(to: destination.appending(path: "notes.txt"))
    try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint)
    XCTAssertTrue(ModelStore.isPrepared(checkpoint, at: destination))
    let siblings = try FileManager.default.contentsOfDirectory(atPath: root.appending(path: "models").path)
    let aside = try XCTUnwrap(siblings.first { $0.hasPrefix("fake.incomplete-") })
    XCTAssertEqual(
      try String(contentsOf: root.appending(path: "models/\(aside)/notes.txt"), encoding: .utf8), "mine")
  }

  func testAMissingFileIsADownloadError() async throws {
    var checkpoint = try publish(["model.safetensors": "weights"])
    checkpoint = Checkpoint(
      name: checkpoint.name, repository: checkpoint.repository, revision: checkpoint.revision,
      files: checkpoint.files + [Checkpoint.File("missing.json", size: 1, sha256: String(repeating: "a", count: 64))])
    do {
      try await ModelStore.prepare(checkpoint, at: root.appending(path: "models/fake"), endpoint: endpoint)
      XCTFail("expected a download error")
    } catch HunchError.downloadFailed(let file, _) {
      XCTAssertEqual(file, "missing.json")
    }
  }

  func testVerifyFindsACorruptFile() async throws {
    let checkpoint = try publish(["model.safetensors": "weights"])
    let destination = root.appending(path: "models/fake")
    try await ModelStore.prepare(checkpoint, at: destination, endpoint: endpoint)
    try Data("WEIGHTS".utf8).write(to: destination.appending(path: "model.safetensors"))
    XCTAssertTrue(ModelStore.isPrepared(checkpoint, at: destination))  // same size
    XCTAssertThrowsError(try ModelStore.verify(checkpoint, at: destination))
  }

  func testBuiltInCheckpointsPointAtTheOrigin() {
    for checkpoint in [Checkpoint.english, .multilingual] {
      XCTAssertTrue(checkpoint.repository.hasPrefix("convaiinnovations/"))
      XCTAssertEqual(checkpoint.revision.count, 40)
      XCTAssertEqual(
        Set(checkpoint.files.map(\.path)),
        ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
         "tokenizer/tokenizer_config.json"])
      for file in checkpoint.files {
        XCTAssertEqual(file.sha256.count, 64)
        XCTAssertTrue(file.sha256.allSatisfy(\.isHexDigit))
        XCTAssertGreaterThan(file.size, 0)
      }
    }
    XCTAssertEqual(
      Checkpoint.english.url(for: Checkpoint.english.files[0], endpoint: ModelStore.huggingFace).absoluteString,
      "https://huggingface.co/convaiinnovations/laya/resolve/7b928d828b7b0e022f929d9bd2e44165aa270148/model.safetensors")
  }

  func testDefaultLocation() {
    let directory = ModelStore.directory(for: .english, root: URL(fileURLWithPath: "/models"))
    XCTAssertEqual(directory.path, "/models/convaiinnovations--laya/7b928d828b7b0e022f929d9bd2e44165aa270148")
    XCTAssertTrue(ModelStore.defaultRoot.path.hasSuffix("Application Support/Hunch/models"))
  }
}

private final class Reports: @unchecked Sendable {
  private let lock = NSLock()
  private var values = [DownloadProgress]()
  func add(_ value: DownloadProgress) { lock.withLock { values.append(value) } }
  var last: DownloadProgress? { lock.withLock { values.last } }
}
