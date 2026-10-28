// Changed by Hunch, 2026: renamed the module class DecisionModel to DecisionNetwork; split the forward pass into
// the frozen encoder and the trainable decision part, and named the parameters head training updates.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import MLX
import MLXNN

final class HeadAttention: Module {
  let heads: Int
  let headDimension: Int
  @ModuleInfo(key: "in_proj") var input: Linear
  @ModuleInfo(key: "out_proj") var output: Linear

  init(dimensions: Int) {
    heads = max(1, dimensions / 64)
    headDimension = dimensions / heads
    _input.wrappedValue = Linear(dimensions, 3 * dimensions)
    _output.wrappedValue = Linear(dimensions, dimensions)
  }

  func callAsFunction(_ values: MLXArray, mask: MLXArray) -> MLXArray {
    let batch = values.dim(0)
    let length = values.dim(1)
    let qkv = input(values).reshaped(batch, length, 3, heads, headDimension)
    let parts = qkv.split(parts: 3, axis: 2)
    let query = parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3)
    let key = parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3)
    let value = parts[2].squeezed(axis: 2).transposed(0, 2, 1, 3)
    let attended = MLXFast.scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: 1 / Float(headDimension).squareRoot(),
      mask: mask
    )
    return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
  }
}

final class HeadLayer: Module {
  @ModuleInfo(key: "self_attn") var attention: HeadAttention
  @ModuleInfo var norm1: LayerNorm
  @ModuleInfo var norm2: LayerNorm
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  init(dimensions: Int) {
    _attention.wrappedValue = HeadAttention(dimensions: dimensions)
    _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
    _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
    _linear1.wrappedValue = Linear(dimensions, 4 * dimensions)
    _linear2.wrappedValue = Linear(4 * dimensions, dimensions)
  }

  func callAsFunction(_ values: MLXArray, mask: MLXArray) -> MLXArray {
    let attended = values + attention(norm1(values), mask: mask)
    return attended + linear2(relu(linear1(norm2(attended))))
  }
}

final class DecisionHead: Module {
  @ModuleInfo var layers: [HeadLayer]

  init(dimensions: Int, count: Int) {
    _layers.wrappedValue = (0..<count).map { _ in HeadLayer(dimensions: dimensions) }
  }

  func callAsFunction(_ values: MLXArray, mask: MLXArray) -> MLXArray {
    layers.reduce(values) { $1($0, mask: mask) }
  }
}

final class DecisionNetwork: Module {
  @ModuleInfo var encoder: ModernBERT
  @ModuleInfo var head: DecisionHead
  @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
  @ModuleInfo var scorer: Sequential
  @ModuleInfo(key: "act_head") var actionHead: Sequential

  init(_ configuration: ModelConfiguration) {
    let dimensions = configuration.encoder.hiddenSize
    _encoder.wrappedValue = ModernBERT(configuration.encoder)
    _head.wrappedValue = DecisionHead(dimensions: dimensions, count: configuration.agent.headLayers)
    _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: dimensions)
    _scorer.wrappedValue = Sequential {
      LayerNorm(dimensions: dimensions)
      Linear(dimensions, dimensions)
      GELU()
      Linear(dimensions, 1)
    }
    _actionHead.wrappedValue = Sequential {
      Linear(dimensions + 4, 256)
      GELU()
      Linear(256, configuration.agent.actionCosts.count + 1)
    }
  }

  /// The parameter name prefixes head training updates: the decision head's layers, the question type embedding and
  /// the option scorer. The encoder and the action head stay as the checkpoint has them.
  static let headPrefixes = ["head.", "type_emb.", "scorer."]

  static func isHeadParameter(_ name: String) -> Bool {
    headPrefixes.contains { name.hasPrefix($0) }
  }

  func callAsFunction(
    tokenIDs: MLXArray,
    attentionMask: MLXArray,
    markerPositions: MLXArray,
    markerMask: MLXArray,
    questionTypes: MLXArray
  ) -> (logits: MLXArray, actions: MLXArray) {
    decide(
      encoded: encoder(tokenIDs, attentionMask: attentionMask), attentionMask: attentionMask,
      markerPositions: markerPositions, markerMask: markerMask, questionTypes: questionTypes)
  }

  /// The option logits only, from encoder output: what head training differentiates.
  func optionLogits(
    encoded: MLXArray, attentionMask: MLXArray, markerPositions: MLXArray, markerMask: MLXArray,
    questionTypes: MLXArray
  ) -> MLXArray {
    let hidden = headStates(encoded: encoded, attentionMask: attentionMask, questionTypes: questionTypes)
    return markerLogits(hidden, markerPositions: markerPositions, markerMask: markerMask)
  }

  private func headStates(encoded: MLXArray, attentionMask: MLXArray, questionTypes: MLXArray) -> MLXArray {
    let hidden = encoded + typeEmbedding(questionTypes)[0..., .newAxis, 0...].asType(encoded.dtype)
    let mask = attentionMask.asType(.bool)[0..., .newAxis, .newAxis, 0...]
    return head(hidden, mask: mask)
  }

  private func markerLogits(_ hidden: MLXArray, markerPositions: MLXArray, markerMask: MLXArray) -> MLXArray {
    let length = hidden.dim(1)
    let dimensions = hidden.dim(2)
    let batchOffsets = MLXArray.arange(hidden.dim(0))[0..., .newAxis] * length
    let indices = (batchOffsets + maximum(markerPositions, 0)).reshaped(-1)
    let markers = hidden.reshaped(-1, dimensions).take(indices, axis: 0)
      .reshaped(hidden.dim(0), markerPositions.dim(1), dimensions)
    let logits = scorer(markers).squeezed(axis: -1).asType(.float32)
    return which(markerMask, logits, MLXArray(-10_000 as Float))
  }

  /// Everything after the encoder: the decision head, the option logits and the action head.
  func decide(
    encoded: MLXArray, attentionMask: MLXArray, markerPositions: MLXArray, markerMask: MLXArray,
    questionTypes: MLXArray
  ) -> (logits: MLXArray, actions: MLXArray) {
    let hidden = headStates(encoded: encoded, attentionMask: attentionMask, questionTypes: questionTypes)
    let logits = markerLogits(hidden, markerPositions: markerPositions, markerMask: markerMask)
    let probabilities = softmax(logits, axis: -1)
    let count = maximum(markerMask.sum(axis: -1), 2).asType(.float32)
    let entropy = -(probabilities * log(maximum(probabilities, 1e-9))).sum(axis: -1) / log(count)
    let sortedProbabilities = sorted(probabilities, axis: -1)
    let width = sortedProbabilities.dim(1)
    let top = sortedProbabilities[0..., (width - 2)..<width]
    let features = stacked(
      [top[0..., 1], top[0..., 1] - top[0..., 0], entropy, count / 255],
      axis: -1
    )
    let pooled = concatenated([hidden[0..., 0, 0...].asType(.float32), features], axis: -1)
    let actions = actionHead(pooled.asType(DType.float16)).asType(DType.float32)
    return (logits, actions)
  }
}
