// Changed by Hunch, 2026: rewritten. Downloads pinned origin checkpoints with URLSession instead of the Hub client,
// verifies every file's sha256, downloads into a temporary directory and moves it into place, and never deletes an
// existing directory.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import CryptoKit
import Foundation

/// Download progress over all files of a checkpoint.
public struct DownloadProgress: Sendable, Equatable {
  public let completedBytes: Int64
  public let totalBytes: Int64
  /// The file being downloaded.
  public let file: String

  public var fractionCompleted: Double {
    totalBytes > 0 ? min(1, Double(completedBytes) / Double(totalBytes)) : 1
  }
}

/// Where checkpoints live on disk, and how they get there.
///
/// Inference never downloads: call `prepare` from a setup step, then hand the directory to `DecisionModel`.
public enum ModelStore {
  /// `~/Library/Application Support/Hunch/models`, shared by every app of the user that uses Hunch (inside the
  /// container for sandboxed apps).
  public static var defaultRoot: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appending(path: "Hunch/models", directoryHint: .isDirectory)
  }

  public static let huggingFace = URL(string: "https://huggingface.co")!

  /// `<root>/<owner>--<name>/<revision>`.
  public static func directory(for checkpoint: Checkpoint, root: URL = defaultRoot) -> URL {
    root.appending(path: checkpoint.repository.replacingOccurrences(of: "/", with: "--"), directoryHint: .isDirectory)
      .appending(path: checkpoint.revision, directoryHint: .isDirectory)
  }

  /// Every file is present with its expected size. Cheap; `verify` hashes them.
  public static func isPrepared(_ checkpoint: Checkpoint, at directory: URL) -> Bool {
    checkpoint.files.allSatisfy { file in
      let attributes = try? FileManager.default.attributesOfItem(atPath: directory.appending(path: file.path).path)
      return (attributes?[.size] as? NSNumber)?.int64Value == file.size
    }
  }

  /// Hashes every file and throws on the first mismatch.
  public static func verify(_ checkpoint: Checkpoint, at directory: URL) throws {
    for file in checkpoint.files {
      let url = directory.appending(path: file.path)
      guard FileManager.default.fileExists(atPath: url.path) else { throw HunchError.modelNotPrepared(directory) }
      let actual = try sha256(of: url)
      guard actual == file.sha256 else {
        throw HunchError.checksumMismatch(file: file.path, expected: file.sha256, actual: actual)
      }
    }
  }

  /// Makes `checkpoint` available in `directory` (by default `directory(for:)` under `defaultRoot`) and returns it.
  ///
  /// A directory that already holds every file at its expected size is returned untouched. Otherwise every file is
  /// downloaded from `endpoint` into a temporary directory next to the destination, checked against its sha256, and
  /// the whole directory is moved into place. An incomplete directory already at the destination is moved aside
  /// (to `<name>.incomplete-<timestamp>`), never deleted.
  @discardableResult
  public static func prepare(
    _ checkpoint: Checkpoint,
    at directory: URL? = nil,
    endpoint: URL = huggingFace,
    progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
  ) async throws -> URL {
    let destination = directory ?? Self.directory(for: checkpoint)
    if isPrepared(checkpoint, at: destination) { return destination }

    let fileManager = FileManager.default
    let parent = destination.deletingLastPathComponent()
    try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appending(path: ".hunch-download-\(UUID().uuidString)", directoryHint: .isDirectory)
    try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
    // Only the staging directory this call created is ever removed.
    defer { try? fileManager.removeItem(at: staging) }

    let total = checkpoint.totalSize
    var done: Int64 = 0
    for file in checkpoint.files {
      try Task.checkCancellation()
      let target = staging.appending(path: file.path)
      try fileManager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
      let base = done
      try await Downloader.download(checkpoint.url(for: file, endpoint: endpoint), to: target) { written in
        progress(DownloadProgress(completedBytes: base + min(written, file.size), totalBytes: total, file: file.path))
      }
      let actual = try sha256(of: target)
      guard actual == file.sha256 else {
        throw HunchError.checksumMismatch(file: file.path, expected: file.sha256, actual: actual)
      }
      done += file.size
      progress(DownloadProgress(completedBytes: done, totalBytes: total, file: file.path))
    }

    if isPrepared(checkpoint, at: destination) { return destination }  // another process finished first
    if fileManager.fileExists(atPath: destination.path) {
      let stamp = Int(Date().timeIntervalSince1970)
      let aside = parent.appending(path: "\(destination.lastPathComponent).incomplete-\(stamp)")
      try fileManager.moveItem(at: destination, to: aside)
    }
    try fileManager.moveItem(at: staging, to: destination)
    return destination
  }

  /// Lowercase hex sha256 of a file, read in chunks.
  public static func sha256(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

/// One file download with byte progress, written to a given location.
private final class Downloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
  private let destination: URL
  private let onProgress: @Sendable (Int64) -> Void
  private var continuation: CheckedContinuation<Void, any Error>?
  private var finished = false
  private let lock = NSLock()

  private init(destination: URL, onProgress: @escaping @Sendable (Int64) -> Void) {
    self.destination = destination
    self.onProgress = onProgress
  }

  static func download(_ url: URL, to destination: URL, onProgress: @escaping @Sendable (Int64) -> Void)
    async throws
  {
    let delegate = Downloader(destination: destination, onProgress: onProgress)
    let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
    defer { session.finishTasksAndInvalidate() }
    let task = session.downloadTask(with: url)
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        delegate.lock.withLock { delegate.continuation = continuation }
        task.resume()
      }
    } onCancel: {
      task.cancel()
    }
  }

  private func finish(_ result: Result<Void, any Error>) {
    let continuation: CheckedContinuation<Void, any Error>? = lock.withLock {
      guard !finished else { return nil }
      finished = true
      return self.continuation
    }
    continuation?.resume(with: result)
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    onProgress(totalBytesWritten)
  }

  func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
    let name = downloadTask.originalRequest?.url?.lastPathComponent ?? "file"
    if let response = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(response.statusCode) {
      finish(.failure(HunchError.downloadFailed(file: name, reason: "HTTP \(response.statusCode)")))
      return
    }
    do {
      // The temporary file is gone once this returns, so it is moved here.
      try FileManager.default.moveItem(at: location, to: destination)
      finish(.success(()))
    } catch {
      finish(.failure(HunchError.downloadFailed(file: name, reason: "\(error.localizedDescription)")))
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
    let name = task.originalRequest?.url?.lastPathComponent ?? "file"
    if let error {
      if (error as? URLError)?.code == .cancelled {
        finish(.failure(CancellationError()))
      } else {
        finish(.failure(HunchError.downloadFailed(file: name, reason: error.localizedDescription)))
      }
    } else {
      finish(.failure(HunchError.downloadFailed(file: name, reason: "no file was written")))
    }
  }
}
