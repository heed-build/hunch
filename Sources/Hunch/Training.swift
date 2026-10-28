// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// A question about a state with the answer it should get: what head training learns from and evaluation scores.
public struct TrainingExample: Hashable, Sendable {
  public var state: String
  public var question: Question
  /// The right answer's label: a choice option's label, a score level, or "true"/"false" for a noul question.
  public var answer: String
  /// How much this example counts in the loss, relative to the others.
  public var weight: Float

  public init(state: String, question: Question, answer: String, weight: Float = 1) {
    self.state = state
    self.question = question
    self.answer = answer
    self.weight = weight
  }
}

/// How head training runs.
public struct TrainingOptions: Sendable, Equatable {
  /// Passes over the examples.
  public var epochs: Int
  public var learningRate: Float
  /// Examples per step.
  public var batchSize: Int
  /// Orders the examples within each epoch, so a run can be repeated.
  public var seed: UInt64
  /// Encoder outputs kept between epochs, in bytes; batches past it are encoded again every epoch. The encoder is
  /// frozen, so its output for an example never changes.
  public var encoderCacheBytes: Int

  public init(
    epochs: Int = 3, learningRate: Float = 5e-5, batchSize: Int = 8, seed: UInt64 = 0,
    encoderCacheBytes: Int = 256 * 1024 * 1024
  ) {
    self.epochs = epochs
    self.learningRate = learningRate
    self.batchSize = batchSize
    self.seed = seed
    self.encoderCacheBytes = encoderCacheBytes
  }
}

/// Where a training run is, reported after every step.
public struct TrainingProgress: Sendable, Equatable {
  public let epoch: Int
  public let step: Int
  public let steps: Int
  /// The weighted mean loss of the step just taken.
  public let loss: Double
}

/// What a training run did.
public struct TrainingReport: Sendable, Equatable {
  public let examples: Int
  public let steps: Int
  /// The weighted mean loss over the first and the last epoch.
  public let firstEpochLoss: Double
  public let lastEpochLoss: Double
  public let seconds: Double
}

/// How a model answers a set of examples.
public struct Evaluation: Sendable, Equatable {
  public let count: Int
  /// Examples whose most likely answer is the expected one.
  public let correct: Int
  /// Mean cross-entropy of the expected answer, in nats, before calibration.
  public let meanLoss: Double

  public var accuracy: Double { count == 0 ? 0 : Double(correct) / Double(count) }

  public init(count: Int, correct: Int, meanLoss: Double) {
    self.count = count
    self.correct = correct
    self.meanLoss = meanLoss
  }
}

/// Head training: the decision head (its transformer layers, the question type embedding and the option scorer) is
/// fine-tuned on labelled examples while the encoder and the action head stay frozen. The trained head is a few
/// dozen megabytes (the English checkpoint's is about 52 MB at float16) and can be saved, loaded on top of the same
/// checkpoint, and reset to the checkpoint's own.
extension DecisionModel {
  /// Scores `examples` with the head in use: how many get their expected answer, and the mean loss.
  public func evaluate(_ examples: [TrainingExample]) async throws -> Evaluation {
    let model = try await model()
    let prepared = try HeadTrainer.prepare(examples, builder: model.builder)
    return try HeadTrainer(model: model).evaluate(prepared, batchSize: options.batchSize)
  }

  /// Fine-tunes the head in use on `examples`. The new head is used from the next `answer`; it lives in memory
  /// until `saveHead` writes it, so `unload()` loses an unsaved one. Cancelling the task stops between steps and
  /// restores the head the run started from.
  @discardableResult
  public func trainHead(
    on examples: [TrainingExample], options training: TrainingOptions = TrainingOptions(),
    progress: @escaping @Sendable (TrainingProgress) -> Void = { _ in }
  ) async throws -> TrainingReport {
    guard !examples.isEmpty else { throw HunchError.invalidTraining("At least one example is required.") }
    guard training.epochs > 0, training.batchSize > 0, training.learningRate > 0 else {
      throw HunchError.invalidTraining("Epochs, batch size and learning rate must be positive.")
    }
    let model = try await model()
    let prepared = try HeadTrainer.prepare(examples, builder: model.builder)
    let trainer = HeadTrainer(model: model)
    let start = trainer.headSnapshot()
    do {
      let report = try trainer.train(prepared, options: training, progress: progress)
      promptCache = PromptPrefixCache()
      return report
    } catch {
      trainer.restore(start)
      throw error
    }
  }

  /// Writes the head in use to a safetensors file, with `metadata` and the checkpoint it belongs to.
  public func saveHead(to url: URL, metadata: [String: String] = [:]) async throws {
    let model = try await model()
    var metadata = metadata
    metadata[HeadFile.formatKey] = HeadFile.format
    metadata[HeadFile.hiddenSizeKey] = String(model.hiddenSize)
    let parameters = HeadTrainer(model: model).headParameters()
    try mlxCall("Saving the head failed") {
      try MLX.save(arrays: parameters.mapValues { $0.asType(.float16) }, metadata: metadata, url: url)
    }
  }

  /// Uses the head saved in `url` instead of the one in use, and again after the model is unloaded and loaded.
  /// Returns the file's metadata.
  @discardableResult
  public func loadHead(from url: URL) async throws -> [String: String] {
    let model = try await model()
    let metadata = try HeadTrainer(model: model).load(url)
    headURL = url
    promptCache = PromptPrefixCache()
    return metadata
  }

  /// Goes back to the checkpoint's own head.
  public func resetHead() async throws {
    let model = try await model()
    HeadTrainer(model: model).restore(model.baseHead)
    headURL = nil
    promptCache = PromptPrefixCache()
  }

  /// The head file in use, nil for the checkpoint's own (or one trained and not saved).
  public var head: URL? { headURL }
}

enum HeadFile {
  static let formatKey = "hunch.format"
  static let format = "head-1"
  static let hiddenSizeKey = "hunch.hidden_size"
}

struct PreparedExample: Sendable {
  let question: PreparedQuestion
  let target: Int
  let weight: Float
}

/// Trains, evaluates, saves and restores a loaded model's head. Only used on the model's queue.
struct HeadTrainer {
  let model: LoadedModel
  var network: DecisionNetwork { model.network }

  /// Each example tokenized like a question, with the index of its expected option.
  static func prepare(_ examples: [TrainingExample], builder: PromptBuilder) throws -> [PreparedExample] {
    var prefixes = PromptPrefixCache()
    return try examples.enumerated().map { index, example in
      let prefix = try prefixes.value(for: example.question) { try builder.makePrefix(for: example.question) }
      guard let target = prefix.labels.firstIndex(of: example.answer) else {
        throw HunchError.invalidTraining(
          "Example \(index): \"\(example.answer)\" is not one of \(prefix.labels.joined(separator: ", ")).")
      }
      guard example.weight.isFinite, example.weight >= 0 else {
        throw HunchError.invalidTraining("Example \(index): the weight must be finite and not negative.")
      }
      let question = try builder.prepare(
        id: String(index), stateTokens: builder.tokenizeState(example.state), prefix: prefix)
      return PreparedExample(question: question, target: target, weight: example.weight)
    }
  }

  /// The type the head's parameters are kept in (float16 for the published checkpoints).
  var headType: DType { headParameters().values.first?.dtype ?? .float16 }

  /// The head's parameters as the network holds them. An update replaces their values in place, so keep
  /// `headSnapshot()` instead to go back to them later.
  func headParameters() -> [String: MLXArray] {
    var result = [String: MLXArray]()
    for (name, value) in network.parameters().flattened() where DecisionNetwork.isHeadParameter(name) {
      result[name] = value
    }
    return result
  }

  /// Copies of the head's parameters that later updates leave alone.
  func headSnapshot() -> [String: MLXArray] {
    Self.snapshot(headParameters())
  }

  static func snapshot(_ parameters: [String: MLXArray]) -> [String: MLXArray] {
    let copies = parameters.mapValues { stopGradient($0) }
    eval(Array(copies.values))
    return copies
  }

  func restore(_ parameters: [String: MLXArray]) {
    network.update(parameters: ModuleParameters.unflattened(parameters))
    eval(network.parameters())
  }

  func load(_ url: URL) throws -> [String: String] {
    let (arrays, metadata) = try mlxCall("Loading the head failed") { try MLX.loadArraysAndMetadata(url: url) }
    guard metadata[HeadFile.formatKey] == HeadFile.format else {
      throw HunchError.invalidModel("\(url.lastPathComponent) is not a Hunch head file.")
    }
    guard metadata[HeadFile.hiddenSizeKey] == String(model.hiddenSize) else {
      throw HunchError.invalidModel("\(url.lastPathComponent) was trained for another checkpoint.")
    }
    let expected = headParameters()
    guard Set(arrays.keys) == Set(expected.keys) else {
      throw HunchError.invalidModel("\(url.lastPathComponent) does not hold this checkpoint's head.")
    }
    for (name, value) in arrays where value.shape != expected[name]!.shape {
      throw HunchError.invalidModel("\(url.lastPathComponent): \(name) has the wrong shape.")
    }
    let type = headType
    restore(arrays.mapValues { $0.asType(type) })
    return metadata
  }

  func evaluate(_ examples: [PreparedExample], batchSize: Int) throws -> Evaluation {
    var correct = 0
    var loss = 0.0
    for start in stride(from: 0, to: examples.count, by: batchSize) {
      let slice = Array(examples[start..<min(start + batchSize, examples.count)])
      let batch = try TrainingBatch(slice, padTokenID: model.padTokenID)
      let (logits, losses) = try mlxCall("Evaluation failed") {
        let encoded = network.encoder(batch.inputs.tokenIDs, attentionMask: batch.inputs.attentionMask)
        let logits = network.optionLogits(
          encoded: encoded, attentionMask: batch.inputs.attentionMask,
          markerPositions: batch.inputs.markerPositions, markerMask: batch.inputs.markerMask,
          questionTypes: batch.inputs.questionTypes)
        let losses = crossEntropy(logits: logits, targets: batch.targets, reduction: .none)
        try checkedEval(logits, losses)
        return (logits.asArray(Float.self), losses.asArray(Float.self))
      }
      let width = logits.count / slice.count
      for (row, example) in slice.enumerated() {
        let options = Array(logits[(row * width)..<(row * width + example.question.markers.count)])
        let best = options.indices.max { options[$0] < options[$1] }!
        if best == example.target { correct += 1 }
        loss += Double(losses[row])
      }
    }
    return Evaluation(
      count: examples.count, correct: correct, meanLoss: examples.isEmpty ? 0 : loss / Double(examples.count))
  }

  func train(
    _ examples: [PreparedExample], options: TrainingOptions, progress: (TrainingProgress) -> Void
  ) throws -> TrainingReport {
    let started = Date()
    // Trains at float32, then keeps the head in the type the checkpoint has it in.
    let type = headType
    restore(headParameters().mapValues { $0.asType(.float32) })
    network.freeze()
    network.head.unfreeze()
    network.typeEmbedding.unfreeze()
    network.scorer.unfreeze()
    defer {
      network.unfreeze()
      restore(headParameters().mapValues { $0.asType(type) })
    }

    // Batches are fixed (so their encoder output can be kept); their order changes every epoch.
    var batches = [TrainingBatch]()
    for start in stride(from: 0, to: examples.count, by: options.batchSize) {
      batches.append(try TrainingBatch(
        Array(examples[start..<min(start + options.batchSize, examples.count)]), padTokenID: model.padTokenID))
    }
    var cache = [Int: MLXArray]()
    var cachedBytes = 0
    func encoded(_ index: Int) throws -> MLXArray {
      if let hit = cache[index] { return hit }
      let inputs = batches[index].inputs
      let value = try mlxCall("Encoding failed") {
        let value = network.encoder(inputs.tokenIDs, attentionMask: inputs.attentionMask)
        try checkedEval(value)
        return value
      }
      if cachedBytes + value.nbytes <= options.encoderCacheBytes {
        cache[index] = value
        cachedBytes += value.nbytes
      }
      return value
    }

    let optimizer = AdamW(learningRate: options.learningRate, weightDecay: 0.01)
    // Weights count across the whole set, not within a batch: each is divided by the mean weight, and a step's loss
    // is the mean over its rows, so a batch of light examples moves the head less than one of heavy examples.
    let meanWeight = max(examples.reduce(0) { $0 + $1.weight } / Float(examples.count), 1e-6)
    let step = valueAndGrad(model: network) { (network: DecisionNetwork, arguments: [MLXArray]) -> [MLXArray] in
      let logits = network.optionLogits(
        encoded: arguments[0], attentionMask: arguments[1], markerPositions: arguments[2],
        markerMask: arguments[3], questionTypes: arguments[4])
      let losses = crossEntropy(logits: logits, targets: arguments[5], reduction: .none)
      return [(losses * arguments[6]).mean()]
    }

    var generator = SplitMix64(seed: options.seed)
    let steps = options.epochs * batches.count
    var taken = 0
    var firstEpochLoss = 0.0
    var lastEpochLoss = 0.0
    for epoch in 0..<options.epochs {
      var epochLoss = 0.0
      var epochWeight = 0.0
      for index in batches.indices.shuffled(using: &generator) {
        try Task.checkCancellation()
        let batch = batches[index]
        let hidden = try encoded(index).asType(.float32)
        let loss = try mlxCall("Training failed") {
          let (values, gradients) = step(network, [
            hidden, batch.inputs.attentionMask, batch.inputs.markerPositions, batch.inputs.markerMask,
            batch.inputs.questionTypes, batch.targets, batch.weights / meanWeight,
          ])
          optimizer.update(model: network, gradients: gradients)
          try checkedEval(values[0], network.parameters(), optimizer)
          return Double(values[0].item(Float.self))
        }
        taken += 1
        epochLoss += loss * Double(batch.rows) * Double(meanWeight)
        epochWeight += Double(batch.totalWeight)
        progress(TrainingProgress(epoch: epoch, step: taken, steps: steps, loss: loss))
      }
      let mean = epochWeight > 0 ? epochLoss / epochWeight : 0
      if epoch == 0 { firstEpochLoss = mean }
      lastEpochLoss = mean
    }
    return TrainingReport(
      examples: examples.count, steps: taken, firstEpochLoss: firstEpochLoss, lastEpochLoss: lastEpochLoss,
      seconds: Date().timeIntervalSince(started))
  }
}

/// A batch of examples: the model inputs, the expected option per row, and the weights.
struct TrainingBatch {
  let inputs: InferenceBatch
  let targets: MLXArray
  let weights: MLXArray
  let totalWeight: Float
  let rows: Int

  init(_ examples: [PreparedExample], padTokenID: Int) throws {
    rows = examples.count
    inputs = try InferenceBatch(examples.map(\.question), padTokenID: padTokenID)
    targets = MLXArray(examples.map { Int32($0.target) })
    weights = MLXArray(examples.map(\.weight))
    totalWeight = examples.reduce(0) { $0 + $1.weight }
  }
}

/// A small seeded generator, so an epoch's order can be repeated.
struct SplitMix64: RandomNumberGenerator {
  private var state: UInt64

  init(seed: UInt64) { state = seed }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}
