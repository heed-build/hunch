// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class ScoringTests: XCTestCase {
  private func calibration(_ t: Float) -> Calibration {
    Calibration(temperatures: .init(byKind: [.choice: t, .score: t, .noul: t]))
  }

  func testNoulAppliesTheTemperature() {
    let answer = Scoring.answer(kind: .noul, labels: ["false", "true"], logits: [0, 2], act: 0.8, calibration: calibration(2))
    let expected = 1 / (1 + exp(-1.0))  // softmax([0, 2] / 2)[1]
    XCTAssertEqual(answer.probability!, expected, accuracy: 1e-6)
    XCTAssertEqual(answer.probabilities["true"]!, expected, accuracy: 1e-6)
    XCTAssertEqual(answer.probabilities["false"]!, 1 - expected, accuracy: 1e-6)
    XCTAssertEqual(answer.confidence, expected, accuracy: 1e-6)
    XCTAssertEqual(answer.calibratedConfidence, expected, accuracy: 1e-6)
    XCTAssertNil(answer.choice)
    XCTAssertNil(answer.score)
    XCTAssertEqual(answer.act, 0.8, accuracy: 1e-6)
  }

  func testTheClampBoundsSharpening() {
    // A checkpoint temperature of 0.1 would turn logits [0, 0.5] into 0.993; clamped to 0.5 it is 0.731.
    let calibration = Calibration(temperatures: .init(byKind: [.noul: 0.1]))
    let answer = Scoring.answer(kind: .noul, labels: ["false", "true"], logits: [0, 0.5], act: 1, calibration: calibration)
    XCTAssertEqual(answer.probability!, 1 / (1 + exp(-1.0)), accuracy: 1e-6)
  }

  func testChoice() {
    let answer = Scoring.answer(
      kind: .choice, labels: ["a", "b", "c"], logits: [1, 3, 1], act: 1, calibration: calibration(1))
    let e = exp(2.0)
    XCTAssertEqual(answer.choice, "b")
    XCTAssertEqual(answer.probabilities["b"]!, e / (e + 2), accuracy: 1e-6)
    XCTAssertEqual(answer.calibratedConfidence, e / (e + 2), accuracy: 1e-6)
    let p = [1 / (e + 2), e / (e + 2), 1 / (e + 2)]
    let entropy = -p.reduce(0) { $0 + $1 * log($1) }
    XCTAssertEqual(answer.confidence, 1 - entropy / log(3), accuracy: 1e-5)
    XCTAssertNil(answer.probability)
  }

  func testScoreReportsTheLikeliestLevelAndTheExpectedLevel() {
    let answer = Scoring.answer(
      kind: .score, labels: ["low", "mid", "high"], logits: [0, 0, log(2)], act: 1, calibration: calibration(1))
    XCTAssertEqual(answer.choice, "high")
    XCTAssertEqual(answer.probabilities["high"]!, 0.5, accuracy: 1e-6)
    XCTAssertEqual(answer.score!, 0 * 0.25 + 1 * 0.25 + 2 * 0.5, accuracy: 1e-6)
  }

  func testUniformChoiceHasZeroConfidence() {
    let answer = Scoring.answer(kind: .choice, labels: ["a", "b"], logits: [1, 1], act: 1, calibration: calibration(1))
    XCTAssertEqual(answer.confidence, 0, accuracy: 1e-6)
    XCTAssertEqual(answer.calibratedConfidence, 0.5, accuracy: 1e-6)
  }

  func testBinningRecalibratesTheAnswerProbability() throws {
    var calibration = calibration(1)
    calibration.binning["choice:2"] = try Calibration.Binning(values: [0.1, 0.3])
    let answer = Scoring.answer(kind: .choice, labels: ["a", "b"], logits: [0, 5], act: 1, calibration: calibration)
    XCTAssertEqual(answer.calibratedConfidence, 0.3)
    XCTAssertGreaterThan(answer.probabilities["b"]!, 0.99)
  }

  func testAnswersRoundTripThroughJSON() throws {
    let answer = Answer(
      choice: "a", score: 1.5, probabilities: ["a": 1], confidence: 0.5, calibratedConfidence: 0.6, act: 0.9)
    XCTAssertEqual(try JSONDecoder().decode(Answer.self, from: JSONEncoder().encode(answer)), answer)
  }
}
