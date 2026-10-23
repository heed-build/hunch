// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation
import XCTest

@testable import Hunch

/// Runs laya-mlx's 63-question parity set against a prepared checkpoint and compares token ids, selected answers,
/// probabilities and the action head with the Python reference's outputs (Fixtures/parity-expected-*.json).
///
/// Needs the checkpoints and the Metal shaders: set HUNCH_ENGLISH_CHECKPOINT and/or HUNCH_MULTILINGUAL_CHECKPOINT to
/// directories prepared with `ModelStore.prepare(.english)` / `(.multilingual)`.
final class ParityTests: XCTestCase {
  func testEnglish() async throws {
    try await parity(checkpointDirectory("HUNCH_ENGLISH_CHECKPOINT"), expected: "parity-expected-english.json")
  }

  func testMultilingual() async throws {
    try await parity(
      checkpointDirectory("HUNCH_MULTILINGUAL_CHECKPOINT"), expected: "parity-expected-multilingual.json")
  }

  private func parity(_ directory: URL, expected name: String) async throws {
    let requests = try JSONDecoder().decode(
      [RequestSpec].self, from: Data(contentsOf: try fixture("parity-requests.json")))
    let expected = try JSONDecoder().decode(
      ExpectedFile.self, from: Data(contentsOf: try fixture(name))
    ).cases
    XCTAssertEqual(requests.map(\.name), expected.map(\.name))

    let model = try await DecisionModel(directory: directory)
    var questions = 0
    var identicalTokens = 0
    var sameAnswer = 0
    var maxDelta = 0.0
    var firstAnswers = [String: Answer]()
    for (request, reference) in zip(requests, expected) {
      let asked = request.questions.mapValues(\.question)
      let tokens = try await model.tokenIDs(state: request.state, questions: asked)
      let answers = try await model.answer(state: request.state, questions: asked)
      if firstAnswers.isEmpty { firstAnswers = answers }
      for (id, spec) in request.questions {
        questions += 1
        let want = try XCTUnwrap(reference.answers[id])
        let answer = try XCTUnwrap(answers[id])
        if tokens[id] == reference.tokens[id] {
          identicalTokens += 1
        } else {
          XCTFail("\(request.name)/\(id): token ids differ")
        }
        let got = spec.orderedLabels.map { answer.probabilities[$0] ?? .nan }
        XCTAssertEqual(got.count, want.probabilities.count, "\(request.name)/\(id)")
        let delta = zip(got, want.probabilities).map { abs($0 - $1) }.max() ?? .nan
        maxDelta = max(maxDelta, delta)
        XCTAssertLessThanOrEqual(delta, 0.01, "\(request.name)/\(id): probabilities \(got) vs \(want.probabilities)")
        if argmax(got) == argmax(want.probabilities) {
          sameAnswer += 1
        } else {
          XCTFail("\(request.name)/\(id): selected answer differs")
        }
        XCTAssertEqual(answer.act, want.act, accuracy: 0.02, "\(request.name)/\(id): act")
      }
    }
    print(
      "parity \(name): \(questions) questions, tokens identical \(identicalTokens), same answer \(sameAnswer), "
        + "max |Δp| \(String(format: "%.4f", maxDelta))")
    XCTAssertEqual(questions, 63)

    // Unloading frees the model; the next answer loads it again and answers the same.
    await model.unload()
    let unloaded = await model.isLoaded
    XCTAssertFalse(unloaded)
    let first = requests[0]
    let again = try await model.answer(state: first.state, questions: first.questions.mapValues(\.question))
    XCTAssertEqual(again, firstAnswers)
  }

  private func argmax(_ values: [Double]) -> Int {
    values.indices.max { values[$0] < values[$1] } ?? -1
  }
}

private struct RequestSpec: Decodable {
  struct QuestionSpec: Decodable {
    let t: String
    let ins: String
    let options: [String]?
    let levels: [String]?
    let falseText: String?
    let trueText: String?

    /// Options here are already rendered ("billing: invoices, …"), so they are passed as labels.
    var question: Question {
      switch t {
      case "choice": .choice(instructions: ins, options: [Option](labels: options ?? []))
      case "score": .score(instructions: ins, levels: levels ?? [])
      default: .noul(instructions: ins, falseDescription: falseText, trueDescription: trueText)
      }
    }

    var orderedLabels: [String] {
      switch t {
      case "choice": options ?? []
      case "score": levels ?? []
      default: ["false", "true"]
      }
    }
  }

  let name: String
  let state: String
  let questions: [String: QuestionSpec]
}

private struct ExpectedFile: Decodable {
  struct Case: Decodable {
    let name: String
    let tokens: [String: [Int]]
    let answers: [String: ExpectedAnswer]
  }

  struct ExpectedAnswer: Decodable {
    let type: String
    let probabilities: [Double]
    let act: Double
  }

  let cases: [Case]
}
