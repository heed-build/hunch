import MLX
import MLXFast
import MLXNN

final class ModernBERTEmbeddings: Module {
  @ModuleInfo(key: "tok_embeddings") var tokenEmbeddings: Embedding
  @ModuleInfo var norm: LayerNorm

  init(_ configuration: EncoderConfiguration) {
    _tokenEmbeddings.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize,
      dimensions: configuration.hiddenSize
    )
    _norm.wrappedValue = LayerNorm(
      dimensions: configuration.hiddenSize,
      eps: configuration.normEpsilon,
      bias: configuration.normBias
    )
  }

  func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
    norm(tokenEmbeddings(tokenIDs))
  }
}

final class EncoderAttention: Module {
  let heads: Int
  let headDimension: Int
  let ropeBase: Float

  @ModuleInfo(key: "Wqkv") var queryKeyValue: Linear
  @ModuleInfo(key: "Wo") var output: Linear

  init(_ configuration: EncoderConfiguration, type: String) {
    heads = configuration.numAttentionHeads
    headDimension = configuration.headDimension
    ropeBase = configuration.ropeParameters[type]?.ropeTheta ?? 10_000
    _queryKeyValue.wrappedValue = Linear(
      configuration.hiddenSize,
      3 * configuration.hiddenSize,
      bias: configuration.attentionBias
    )
    _output.wrappedValue = Linear(
      configuration.hiddenSize,
      configuration.hiddenSize,
      bias: configuration.attentionBias
    )
  }

  func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
    let batch = input.dim(0)
    let length = input.dim(1)
    let qkv = queryKeyValue(input).reshaped(batch, length, 3, heads, headDimension)
    let parts = qkv.split(parts: 3, axis: 2)
    let query = rope(parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3))
    let key = rope(parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3))
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

  private func rope(_ input: MLXArray) -> MLXArray {
    MLXFast.RoPE(
      input,
      dimensions: headDimension,
      traditional: false,
      base: ropeBase,
      scale: 1,
      offset: 0
    )
  }
}

final class EncoderMLP: Module {
  @ModuleInfo(key: "Wi") var input: Linear
  @ModuleInfo(key: "Wo") var output: Linear

  init(_ configuration: EncoderConfiguration) {
    _input.wrappedValue = Linear(
      configuration.hiddenSize,
      2 * configuration.intermediateSize,
      bias: configuration.mlpBias
    )
    _output.wrappedValue = Linear(
      configuration.intermediateSize,
      configuration.hiddenSize,
      bias: configuration.mlpBias
    )
  }

  func callAsFunction(_ values: MLXArray) -> MLXArray {
    let parts = input(values).split(parts: 2, axis: -1)
    return output(gelu(parts[0]) * parts[1])
  }
}

final class EncoderLayer: Module {
  let attentionType: String
  @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm?
  @ModuleInfo(key: "attn") var attention: EncoderAttention
  @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
  @ModuleInfo var mlp: EncoderMLP

  init(_ configuration: EncoderConfiguration, index: Int) {
    attentionType = configuration.layerTypes[index]
    if index > 0 {
      _attentionNorm.wrappedValue = LayerNorm(
        dimensions: configuration.hiddenSize,
        eps: configuration.normEpsilon,
        bias: configuration.normBias
      )
    }
    _attention.wrappedValue = EncoderAttention(configuration, type: attentionType)
    _mlpNorm.wrappedValue = LayerNorm(
      dimensions: configuration.hiddenSize,
      eps: configuration.normEpsilon,
      bias: configuration.normBias
    )
    _mlp.wrappedValue = EncoderMLP(configuration)
  }

  func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
    let normalized = attentionNorm?(input) ?? input
    let attended = input + attention(normalized, mask: mask)
    return attended + mlp(mlpNorm(attended))
  }
}

final class ModernBERT: Module {
  let localAttention: Int
  @ModuleInfo var embeddings: ModernBERTEmbeddings
  @ModuleInfo var layers: [EncoderLayer]
  @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

  init(_ configuration: EncoderConfiguration) {
    localAttention = configuration.localAttention
    _embeddings.wrappedValue = ModernBERTEmbeddings(configuration)
    _layers.wrappedValue = (0..<configuration.numHiddenLayers).map {
      EncoderLayer(configuration, index: $0)
    }
    _finalNorm.wrappedValue = LayerNorm(
      dimensions: configuration.hiddenSize,
      eps: configuration.normEpsilon,
      bias: configuration.normBias
    )
  }

  func callAsFunction(_ tokenIDs: MLXArray, attentionMask: MLXArray) -> MLXArray {
    let masks = makeAttentionMasks(attentionMask, window: localAttention)
    var hidden = embeddings(tokenIDs)
    for layer in layers {
      hidden = layer(hidden, mask: masks[layer.attentionType]!)
    }
    return finalNorm(hidden)
  }
}

private func makeAttentionMasks(_ input: MLXArray, window: Int) -> [String: MLXArray] {
  let valid = input.asType(.bool)
  let full = valid[0..., .newAxis, .newAxis, 0...]
  let positions = MLXArray.arange(valid.dim(1))
  let distance = abs(positions[0..., .newAxis] - positions[.newAxis, 0...])
  let nearby = distance .<= (window / 2)
  let local =
    (nearby[.newAxis, .newAxis, 0..., 0...] .|| (.!valid[0..., .newAxis, 0..., .newAxis]))
    .&& full
  return ["full_attention": full, "sliding_attention": local]
}
