// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class PromptBuilderTests: XCTestCase {
  private func builder(maxLength: Int = 512, headMaxLength: Int = 192, truncation: Truncation = .keepStart) throws
    -> PromptBuilder
  {
    try PromptBuilder(
      encoder: CharacterEncoder(), maxLength: maxLength, headMaxLength: headMaxLength,
      specials: CharacterEncoder.specials, truncation: truncation)
  }

  private func sequence(_ builder: PromptBuilder, state: String, _ question: Question) throws -> PreparedQuestion {
    try builder.prepare(
      id: "q", stateTokens: builder.tokenizeState(state), prefix: builder.makePrefix(for: question))
  }

  func testNoulLayout() throws {
    let prepared = try sequence(try builder(), state: "hi", .noul(instructions: "Is it?"))
    let expected =
      [1] + ids("noul question: Is it?") + [2]
      + [3] + ids(" false: no, the statement does not hold")
      + [3] + ids(" true: yes, the statement holds") + [2]
      + ids("hi") + [2]
    XCTAssertEqual(prepared.tokenIDs, expected)
    XCTAssertEqual(prepared.labels, ["false", "true"])
    XCTAssertEqual(prepared.kind, .noul)
    XCTAssertEqual(prepared.markers.map { prepared.tokenIDs[$0] }, [3, 3])
  }

  func testChoiceRendersDescriptionsAndReportsLabels() throws {
    let prepared = try sequence(
      try builder(), state: "",
      .choice(instructions: "Route", options: [Option("billing", description: "money"), "other"]))
    let expected =
      [1] + ids("choice question: Route") + [2] + [3] + ids(" billing: money") + [3] + ids(" other") + [2, 2]
    XCTAssertEqual(prepared.tokenIDs, expected)
    XCTAssertEqual(prepared.labels, ["billing", "other"])
  }

  func testScoreRendersLevelsAndReportsTheirNames() throws {
    let prepared = try sequence(try builder(), state: "", .score(instructions: "How bad", levels: ["low", "high"]))
    let expected =
      [1] + ids("score question: How bad") + [2] + [3] + ids(" level 0: low") + [3] + ids(" level 1: high") + [2, 2]
    XCTAssertEqual(prepared.tokenIDs, expected)
    XCTAssertEqual(prepared.labels, ["low", "high"])
  }

  func testCustomNoulWording() throws {
    let prepared = try sequence(
      try builder(), state: "", .noul(instructions: "x", falseDescription: "nope", trueDescription: "yep"))
    XCTAssertEqual(
      prepared.tokenIDs, [1] + ids("noul question: x") + [2] + [3] + ids(" false: nope") + [3] + ids(" true: yep") + [2, 2])
  }

  func testEmptyStateAddsNoTokens() throws {
    let builder = try builder()
    XCTAssertEqual(CharacterEncoder().encode(""), [999])
    XCTAssertEqual(builder.tokenizeState(""), [])
    let prepared = try sequence(builder, state: "", .noul(instructions: "x"))
    XCTAssertEqual(Array(prepared.tokenIDs.suffix(2)), [2, 2])
  }

  func testMaskLiteralsInTheStateBecomeSpaces() throws {
    XCTAssertEqual(try builder().tokenizeState("a[MASK]b"), ids("a b"))
  }

  func testKeepStartKeepsTheBeginningOfALongState() throws {
    let state = String(repeating: "a", count: 100) + String(repeating: "z", count: 100)
    let builder = try builder(maxLength: 120, headMaxLength: 60)
    let prepared = try sequence(builder, state: state, .noul(instructions: "x", falseDescription: "f", trueDescription: "t"))
    XCTAssertEqual(prepared.tokenIDs.count, 120)
    XCTAssertEqual(prepared.tokenIDs.last, 2)
    let kept = prepared.tokenIDs[prepared.tokenIDs.count - 2]
    XCTAssertEqual(kept, ids("a")[0])
  }

  func testKeepEndKeepsTheEndOfALongState() throws {
    let state = String(repeating: "a", count: 100) + "bcdefghijklmnopqrstuvwxyz"
    let builder = try builder(maxLength: 120, headMaxLength: 60, truncation: .keepEnd)
    let question = Question.noul(instructions: "x", falseDescription: "f", trueDescription: "t")
    let prefix = try builder.makePrefix(for: question)
    let prepared = try sequence(builder, state: state, question)
    XCTAssertEqual(prepared.tokenIDs.count, 120)
    let room = 120 - prefix.tokens.count - 1
    XCTAssertEqual(Array(prepared.tokenIDs[prefix.tokens.count..<119]), Array(ids(state).suffix(room)))
    XCTAssertEqual(prepared.tokenIDs.last, 2)
  }

  func testAShortStateIsTheSameEitherWay() throws {
    let question = Question.noul(instructions: "x")
    XCTAssertEqual(
      try sequence(try builder(truncation: .keepStart), state: "short", question).tokenIDs,
      try sequence(try builder(truncation: .keepEnd), state: "short", question).tokenIDs)
  }

  func testLongOptionsAreCutToFitTheHead() throws {
    let options = (0..<6).map { Option("option \($0) " + String(repeating: "x", count: 80)) }
    let prepared = try sequence(try builder(headMaxLength: 64), state: "", .choice(instructions: "pick", options: options))
    XCTAssertEqual(prepared.markers.count, 6)
    // Heading [CLS] … [SEP] at most max(8, budget) + 2, options at most (64 - 16) / 6 tokens each.
    XCTAssertLessThanOrEqual(prepared.markers[1] - prepared.markers[0], 8)
  }

  func testInvalidQuestionsAreRefused() throws {
    let builder = try builder()
    XCTAssertThrowsError(try builder.makePrefix(for: .choice(instructions: "x", options: [])))
    XCTAssertThrowsError(try builder.makePrefix(for: .choice(instructions: "x", options: ["a", "a"])))
    XCTAssertThrowsError(try builder.makePrefix(for: .score(instructions: "x", levels: [])))
    XCTAssertThrowsError(try builder.makePrefix(for: .score(instructions: "x", levels: ["a", "a"])))
  }

  func testMissingSpecialTokensAreRefused() {
    XCTAssertThrowsError(
      try PromptBuilder(
        encoder: CharacterEncoder(), maxLength: 512, headMaxLength: 192,
        specials: SpecialTokens(cls: "<bos>", sep: "[SEP]", mask: "[MASK]", pad: "[PAD]")))
  }

  func testOptionsFromLabels() {
    XCTAssertEqual([Option](labels: ["a", "b"]), [Option("a"), Option("b")])
  }
}
