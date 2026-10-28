// Changed by Hunch, 2026: LayaAgent became DecisionModel: answers Hunch Questions, runs on its own serial queue
// instead of the cooperative pool, loads lazily and can unload, sets MLX's cache limit, applies a Calibration
// (clamped temperatures, binning, languages), reads special tokens from the tokenizer config, maps origin tensor
// names, supports keep-end truncation, turns MLX errors into thrown errors, and can use a trained head (Training.swift).
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Dispatch
import Foundation
import MLX
import MLXNN
import Tokenizers

/// A Laya checkpoint loaded for answering typed questions.
///
/// Loading and inference run on a serial queue of the model's own, never on Swift's cooperative pool. `unload()`
/// frees the weights; the next `answer` loads them again.
public actor DecisionModel {
  public struct Options: Sendable {
    /// Which end of a state that is too long for the checkpoint survives.
    public var truncation: Truncation
    /// Questions per forward pass.
    public var batchSize: Int
    /// Replaces the checkpoint's own calibration when set.
    public var calibration: Calibration?
    /// A head file written by `saveHead`, used instead of the checkpoint's own head.
    public var head: URL?

    public init(
      truncation: Truncation = .keepStart, batchSize: Int = 16, calibration: Calibration? = nil, head: URL? = nil
    ) {
      self.truncation = truncation
      self.batchSize = batchSize
      self.calibration = calibration
      self.head = head
    }
  }

  /// 256 MiB: keeps peak memory near 1.8 GB on the English checkpoint at no measured speed cost.
  public static let defaultCacheLimit = 256 * 1024 * 1024

  public let directory: URL
  public let options: Options

  private let queue: DispatchSerialQueue
  private var loaded: LoadedModel?
  private var loading: Task<LoadedModel, any Error>?
  var promptCache = PromptPrefixCache()
  /// The head file in use; loaded again after `unload()`.
  var headURL: URL?

  public nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

  /// Loads the checkpoint in `directory`. `cacheLimit` sets MLX's buffer cache limit, which is process-wide; pass
  /// nil to leave it alone.
  public init(directory: URL, cacheLimit: Int? = DecisionModel.defaultCacheLimit, options: Options = Options())
    async throws
  {
    guard options.batchSize > 0 else { throw HunchError.invalidModel("Batch size must be positive.") }
    self.directory = directory
    self.options = options
    queue = InferenceExecutor.makeQueue()
    headURL = options.head
    if let cacheLimit { Memory.cacheLimit = cacheLimit }
    _ = try await model()
  }

  public var isLoaded: Bool { loaded != nil }

  /// Answers `questions` about `state`. `language` (e.g. "de") selects a calibration's per-language temperatures.
  public func answer(state: String, questions: [String: Question], language: String? = nil) async throws
    -> [String: Answer]
  {
    guard !questions.isEmpty else { throw HunchError.invalidQuestion("At least one question is required.") }
    let model = try await model()
    let prepared = try prepare(state: state, questions: questions, builder: model.builder)

    var answers = [String: Answer]()
    for start in stride(from: 0, to: prepared.count, by: options.batchSize) {
      let questions = Array(prepared[start..<min(start + options.batchSize, prepared.count)])
      let (rows, width, actions) = try model.forward(questions)
      for (row, question) in questions.enumerated() {
        let logits = Array(rows[(row * width)..<(row * width + question.markers.count)])
        answers[question.id] = Scoring.answer(
          kind: question.kind, labels: question.labels, logits: logits, act: actions[row][0],
          calibration: model.calibration, language: language)
      }
    }
    return answers
  }

  /// The token ids each question is answered from, for parity checks and debugging.
  public func tokenIDs(state: String, questions: [String: Question]) async throws -> [String: [Int]] {
    let model = try await model()
    let prepared = try prepare(state: state, questions: questions, builder: model.builder)
    return Dictionary(uniqueKeysWithValues: prepared.map { ($0.id, $0.tokenIDs) })
  }

  /// Frees the weights and MLX's buffer cache. The next `answer` loads the checkpoint again.
  public func unload() {
    loading?.cancel()
    loading = nil
    loaded = nil
    promptCache = PromptPrefixCache()
    Memory.clearCache()
  }

  // MARK: Internals

  private func prepare(state: String, questions: [String: Question], builder: PromptBuilder) throws
    -> [PreparedQuestion]
  {
    let stateTokens = builder.tokenizeState(state)
    var prepared = [PreparedQuestion]()
    for (id, question) in questions {
      let prefix = try promptCache.value(for: question) { try builder.makePrefix(for: question) }
      prepared.append(try builder.prepare(id: id, stateTokens: stateTokens, prefix: prefix))
    }
    return prepared.sorted { $0.id < $1.id }
  }

  func model() async throws -> LoadedModel {
    if let loaded { return loaded }
    if let loading { return try await loading.value }
    let task = Task { try await self.load() }
    loading = task
    defer { loading = nil }
    let model = try await task.value
    loaded = model
    return model
  }

  private func load() async throws -> LoadedModel {
    let configuration = try ModelConfiguration.load(from: directory)
    let tokenizerDirectory = directory.appending(path: "tokenizer")
    let specials = try SpecialTokens.load(from: tokenizerDirectory)
    let tokenizer: any Tokenizer
    do {
      tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    } catch {
      throw HunchError.invalidModel("The tokenizer could not be loaded: \(error)")
    }
    try Task.checkCancellation()
    let encoder = TransformersEncoder(tokenizer: tokenizer)
    guard let padTokenID = encoder.tokenID(specials.pad) else {
      throw HunchError.invalidModel("The tokenizer is missing its padding token \(specials.pad).")
    }
    let builder = try PromptBuilder(
      encoder: encoder, maxLength: configuration.agent.maxLength,
      headMaxLength: configuration.agent.headMaxLength, specials: specials, truncation: options.truncation)
    let network = try mlxCall("Loading the checkpoint failed") {
      let network = DecisionNetwork(configuration)
      let loaded = try MLX.loadArrays(url: directory.appending(path: "model.safetensors"))
      var weights = [String: MLXArray]()
      for (name, value) in loaded {
        guard let name = WeightNames.mlxName(name) else { continue }
        weights[name] = value.asType(.float16)
      }
      try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
      network.train(false)
      eval(network.parameters())
      return network
    }
    let model = LoadedModel(
      network: network, builder: builder, padTokenID: padTokenID,
      calibration: options.calibration ?? Calibration(configuration.agent),
      hiddenSize: configuration.encoder.hiddenSize)
    if let headURL { try HeadTrainer(model: model).load(headURL) }
    return model
  }
}

/// A checkpoint in memory. Only used on the model's queue.
final class LoadedModel: @unchecked Sendable {
  let network: DecisionNetwork
  let builder: PromptBuilder
  let padTokenID: Int
  let calibration: Calibration
  let hiddenSize: Int
  /// The checkpoint's own head, for `resetHead`.
  let baseHead: [String: MLXArray]

  init(network: DecisionNetwork, builder: PromptBuilder, padTokenID: Int, calibration: Calibration, hiddenSize: Int) {
    self.network = network
    self.builder = builder
    self.padTokenID = padTokenID
    self.calibration = calibration
    self.hiddenSize = hiddenSize
    var head = [String: MLXArray]()
    for (name, value) in network.parameters().flattened() where DecisionNetwork.isHeadParameter(name) {
      head[name] = value
    }
    baseHead = HeadTrainer.snapshot(head)
  }

  /// One forward pass: the option logits (row-major, `width` per question) and the action probabilities.
  func forward(_ questions: [PreparedQuestion]) throws -> (logits: [Float], width: Int, actions: [[Float]]) {
    let batch = try InferenceBatch(questions, padTokenID: padTokenID)
    return try mlxCall("Inference failed") {
      let output = network(
        tokenIDs: batch.tokenIDs,
        attentionMask: batch.attentionMask,
        markerPositions: batch.markerPositions,
        markerMask: batch.markerMask,
        questionTypes: batch.questionTypes
      )
      try checkedEval(output.logits, output.actions)
      let width = output.actions.dim(1)
      let flat = output.actions.asArray(Float.self)
      let actions = stride(from: 0, to: flat.count, by: width).map {
        Scoring.softmax(Array(flat[$0..<$0 + width]))
      }
      return (output.logits.asArray(Float.self), output.logits.dim(1), actions)
    }
  }
}

/// Runs `body` with MLX errors turned into `HunchError.inferenceFailed` instead of ending the process.
func mlxCall<R>(_ what: String, _ body: () throws -> R) throws -> R {
  do {
    return try withError { try body() }
  } catch let error as HunchError {
    throw error
  } catch {
    throw HunchError.inferenceFailed("\(what): \(error)")
  }
}

struct TransformersEncoder: TokenEncoder {
  let tokenizer: any Tokenizer

  func encode(_ text: String) -> [Int] {
    tokenizer.encode(text: text, addSpecialTokens: false)
  }

  func tokenID(_ token: String) -> Int? {
    tokenizer.convertTokenToId(token)
  }
}
