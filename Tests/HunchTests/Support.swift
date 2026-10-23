// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation
import XCTest

@testable import Hunch

/// One token per Unicode scalar (value + 1000), and four special tokens. Like mmBERT's tokenizer, it encodes the
/// empty string as a lone token (999).
struct CharacterEncoder: TokenEncoder {
  static let specials = SpecialTokens(cls: "[CLS]", sep: "[SEP]", mask: "[MASK]", pad: "[PAD]")
  static let ids = ["[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3]

  func encode(_ text: String) -> [Int] { text.isEmpty ? [999] : text.unicodeScalars.map { Int($0.value) + 1000 } }
  func tokenID(_ token: String) -> Int? { Self.ids[token] }
}

func ids(_ text: String) -> [Int] { CharacterEncoder().encode(text) }

func fixture(_ name: String) throws -> URL {
  guard let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil) else {
    throw XCTSkip("Fixture \(name) is missing")
  }
  return url
}

func temporaryDirectory() throws -> URL {
  let url = FileManager.default.temporaryDirectory.appending(path: "hunch-tests-\(UUID().uuidString)")
  try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  return url
}

/// A prepared checkpoint directory named by an environment variable, or a skip.
func checkpointDirectory(_ variable: String) throws -> URL {
  guard let path = ProcessInfo.processInfo.environment[variable], !path.isEmpty else {
    throw XCTSkip("Set \(variable) to a prepared checkpoint directory to run this test.")
  }
  return URL(fileURLWithPath: path)
}
