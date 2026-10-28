// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import Hunch

/// Head training on a tiny randomly initialised network, on the CPU: no checkpoint and no GPU.
final class TrainingTests: XCTestCase {
  static let question = Question.noul(instructions: "The state agrees.")

  /// "yes" means true and "no" false: a head can learn that from the encoder's output alone.
  static let examples: [TrainingExample] = (0..<16).map { index in
    let yes = index.isMultiple(of: 2)
    return TrainingExample(state: yes ? "yes" : "no", question: question, answer: yes ? "true" : "false")
  }

  static let training = TrainingOptions(epochs: 40, learningRate: 3e-3, batchSize: 4, seed: 7)

  func testTrainingTheHeadLearnsTheExamples() throws {
    try onCPU {
      let model = try tinyModel()
      let trainer = HeadTrainer(model: model)
      let prepared = try HeadTrainer.prepare(Self.examples, builder: model.builder)
      let before = try trainer.evaluate(prepared, batchSize: 8)
      let report = try trainer.train(prepared, options: Self.training) { _ in }
      let after = try trainer.evaluate(prepared, batchSize: 8)
      XCTAssertEqual(after.correct, 16, "before: \(before.correct)/16")
      XCTAssertLessThan(after.meanLoss, before.meanLoss)
      XCTAssertLessThan(report.lastEpochLoss, report.firstEpochLoss)
      XCTAssertEqual(report.steps, 40 * 4)
      XCTAssertEqual(report.examples, 16)
    }
  }

  func testTrainingLeavesTheEncoderAndActionHeadAlone() throws {
    try onCPU {
      let model = try tinyModel()
      let frozen = { (name: String) in !DecisionNetwork.isHeadParameter(name) }
      let before = HeadTrainer.snapshot(parameters(model.network, where: frozen))
      let headBefore = HeadTrainer.snapshot(parameters(model.network, where: DecisionNetwork.isHeadParameter))
      try HeadTrainer(model: model).train(
        HeadTrainer.prepare(Self.examples, builder: model.builder), options: Self.training
      ) { _ in }
      let after = parameters(model.network, where: frozen)
      XCTAssertEqual(Set(before.keys), Set(after.keys))
      XCTAssertTrue(before.keys.contains { $0.hasPrefix("encoder.") })
      XCTAssertTrue(before.keys.contains { $0.hasPrefix("act_head.") })
      for (name, value) in before {
        XCTAssertTrue(allClose(value, after[name]!).item(Bool.self), "\(name) changed")
      }
      let headAfter = parameters(model.network, where: DecisionNetwork.isHeadParameter)
      XCTAssertTrue(headBefore.contains { !allClose($0.value, headAfter[$0.key]!).item(Bool.self) })
      XCTAssertTrue(headAfter.values.allSatisfy { $0.dtype == .float32 }, "the head keeps its type")
    }
  }

  func testSavedHeadLoadsAndResetGoesBackToTheCheckpoint() throws {
    try onCPU {
      let model = try tinyModel()
      let trainer = HeadTrainer(model: model)
      let prepared = try HeadTrainer.prepare(Self.examples, builder: model.builder)
      let base = try trainer.evaluate(prepared, batchSize: 8)
      try trainer.train(prepared, options: Self.training) { _ in }
      let trained = try trainer.evaluate(prepared, batchSize: 8)

      let url = try temporaryDirectory().appending(path: "head.safetensors")
      try MLX.save(
        arrays: trainer.headParameters().mapValues { $0.asType(.float16) },
        metadata: [HeadFile.formatKey: HeadFile.format, HeadFile.hiddenSizeKey: "64", "examples": "16"], url: url)

      trainer.restore(model.baseHead)
      XCTAssertEqual(try trainer.evaluate(prepared, batchSize: 8), base)

      let metadata = try trainer.load(url)
      XCTAssertEqual(metadata["examples"], "16")
      let loaded = try trainer.evaluate(prepared, batchSize: 8)
      XCTAssertEqual(loaded.correct, trained.correct)
      XCTAssertEqual(loaded.meanLoss, trained.meanLoss, accuracy: 0.01)
    }
  }

  func testAHeadForAnotherCheckpointIsRefused() throws {
    try onCPU {
      let model = try tinyModel()
      let trainer = HeadTrainer(model: model)
      let directory = try temporaryDirectory()
      let wrongSize = directory.appending(path: "wrong-size.safetensors")
      try MLX.save(
        arrays: trainer.headParameters(),
        metadata: [HeadFile.formatKey: HeadFile.format, HeadFile.hiddenSizeKey: "1024"], url: wrongSize)
      XCTAssertThrowsError(try trainer.load(wrongSize)) {
        XCTAssertEqual($0 as? HunchError, .invalidModel("wrong-size.safetensors was trained for another checkpoint."))
      }
      let notAHead = directory.appending(path: "not-a-head.safetensors")
      try MLX.save(arrays: ["x": MLXArray([1, 2])], metadata: [:], url: notAHead)
      XCTAssertThrowsError(try trainer.load(notAHead)) {
        XCTAssertEqual($0 as? HunchError, .invalidModel("not-a-head.safetensors is not a Hunch head file."))
      }
      let missing = directory.appending(path: "missing.safetensors")
      var partial = trainer.headParameters()
      partial.removeValue(forKey: partial.keys.sorted().first!)
      try MLX.save(
        arrays: partial, metadata: [HeadFile.formatKey: HeadFile.format, HeadFile.hiddenSizeKey: "64"], url: missing)
      XCTAssertThrowsError(try trainer.load(missing)) {
        XCTAssertEqual($0 as? HunchError, .invalidModel("missing.safetensors does not hold this checkpoint's head."))
      }
    }
  }

  func testAnAnswerThatIsNotAnOptionIsRefused() throws {
    try onCPU {
      let model = try tinyModel()
      let bad = [TrainingExample(state: "yes", question: Self.question, answer: "maybe")]
      XCTAssertThrowsError(try HeadTrainer.prepare(bad, builder: model.builder)) {
        XCTAssertEqual($0 as? HunchError, .invalidTraining("Example 0: \"maybe\" is not one of false, true."))
      }
      let score = [
        TrainingExample(state: "x", question: .score(instructions: "How much?", levels: ["low", "high"]), answer: "high")
      ]
      XCTAssertEqual(try HeadTrainer.prepare(score, builder: model.builder).first?.target, 1)
      let negative = [TrainingExample(state: "yes", question: Self.question, answer: "true", weight: -1)]
      XCTAssertThrowsError(try HeadTrainer.prepare(negative, builder: model.builder))
    }
  }

  func testWeightsDecideWhatCounts() throws {
    try onCPU {
      // Contradicting examples: the heavier side wins.
      let model = try tinyModel()
      let examples = (0..<8).map { index in
        TrainingExample(
          state: "yes", question: Self.question, answer: index < 6 ? "false" : "true", weight: index < 6 ? 0.05 : 1)
      }
      let trainer = HeadTrainer(model: model)
      let prepared = try HeadTrainer.prepare(examples, builder: model.builder)
      try trainer.train(prepared, options: Self.training) { _ in }
      let truth = try HeadTrainer.prepare(
        [TrainingExample(state: "yes", question: Self.question, answer: "true")], builder: model.builder)
      XCTAssertEqual(try trainer.evaluate(truth, batchSize: 1).correct, 1)
    }
  }

  func testCancellingStopsBetweenSteps() async throws {
    let outcome = await Task.detached { Self.trainInACancelledTask() }.value
    XCTAssertEqual(outcome, "cancelled after 0 steps")
  }

  /// Cancels the task it runs in, then trains: the first step must not be taken.
  private static func trainInACancelledTask() -> String {
    withUnsafeCurrentTask { $0?.cancel() }
    var steps = 0
    do {
      try onCPU {
        let model = try tinyModel()
        let prepared = try HeadTrainer.prepare(examples, builder: model.builder)
        try HeadTrainer(model: model).train(prepared, options: training) { _ in steps += 1 }
      }
      return "finished after \(steps) steps"
    } catch is CancellationError {
      return "cancelled after \(steps) steps"
    } catch {
      return "\(error)"
    }
  }

  func testTheSameSeedTrainsTheSameHead() throws {
    try onCPU {
      func run() throws -> Evaluation {
        let model = try tinyModel(seed: 3)
        let trainer = HeadTrainer(model: model)
        let prepared = try HeadTrainer.prepare(Self.examples, builder: model.builder)
        try trainer.train(prepared, options: TrainingOptions(epochs: 2, learningRate: 1e-3, batchSize: 4, seed: 1)) {
          _ in
        }
        return try trainer.evaluate(prepared, batchSize: 8)
      }
      XCTAssertEqual(try run(), try run())
    }
  }

  // MARK: Support

  private func parameters(_ network: DecisionNetwork, where keep: (String) -> Bool) -> [String: MLXArray] {
    var result = [String: MLXArray]()
    for (name, value) in network.parameters().flattened() where keep(name) {
      result[name] = value
    }
    return result
  }
}

private func onCPU<R>(_ body: () throws -> R) throws -> R {
  try Device.withDefaultDevice(.cpu, body)
}

/// A one-layer, 64-wide network with random weights over `CharacterEncoder`'s token ids.
private func tinyModel(seed: UInt64 = 0) throws -> LoadedModel {
  MLXRandom.seed(seed)
  let encoder = """
    {"vocab_size": 1200, "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 1,
     "num_attention_heads": 2, "norm_eps": 1e-5, "attention_bias": false, "mlp_bias": false, "norm_bias": false,
     "local_attention": 128, "max_position_embeddings": 512, "layer_types": ["full_attention"],
     "rope_parameters": {"full_attention": {"rope_theta": 10000.0}}}
    """
  let agent = """
    {"head_layers": 1, "max_len": 128, "head_max_len": 64, "temperature": [1, 1, 1],
     "temperature_by_options": {}, "act_costs": {"escalate": 0.5}}
    """
  let configuration = ModelConfiguration(
    encoder: try JSONDecoder().decode(EncoderConfiguration.self, from: Data(encoder.utf8)),
    agent: try JSONDecoder().decode(AgentConfiguration.self, from: Data(agent.utf8)))
  let network = DecisionNetwork(configuration)
  network.train(false)
  eval(network.parameters())
  let builder = try PromptBuilder(
    encoder: CharacterEncoder(), maxLength: 128, headMaxLength: 64, specials: CharacterEncoder.specials)
  return LoadedModel(
    network: network, builder: builder, padTokenID: 0, calibration: Calibration(configuration.agent), hiddenSize: 64)
}
