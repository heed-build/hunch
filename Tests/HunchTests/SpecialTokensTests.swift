// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class SpecialTokensTests: XCTestCase {
  func testModernBERTStyleConfig() throws {
    let specials = try SpecialTokens.load(from: try fixture("tokenizer-modernbert"))
    XCTAssertEqual(specials, SpecialTokens(cls: "[CLS]", sep: "[SEP]", mask: "[MASK]", pad: "[PAD]"))
  }

  func testMmBERTStyleConfig() throws {
    let specials = try SpecialTokens.load(from: try fixture("tokenizer-mmbert"))
    XCTAssertEqual(specials, SpecialTokens(cls: "<bos>", sep: "<eos>", mask: "<mask>", pad: "<pad>"))
  }

  func testAddedTokenObjects() throws {
    let json = #"""
      {"cls_token": {"content": "<s>", "lstrip": false}, "sep_token": "</s>",
       "mask_token": {"content": "<mask>"}, "pad_token": "<pad>"}
      """#
    let specials = try SpecialTokens.parse(Data(json.utf8))
    XCTAssertEqual(specials, SpecialTokens(cls: "<s>", sep: "</s>", mask: "<mask>", pad: "<pad>"))
  }

  func testMissingTokenIsAnError() {
    XCTAssertThrowsError(try SpecialTokens.parse(Data(#"{"cls_token": "[CLS]"}"#.utf8)))
    XCTAssertThrowsError(try SpecialTokens.parse(Data("[]".utf8)))
    XCTAssertThrowsError(try SpecialTokens.load(from: URL(fileURLWithPath: "/nonexistent")))
  }
}
