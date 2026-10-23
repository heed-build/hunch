// Changed by Hunch, 2026: renamed the error type, added SpecialTokens (read from tokenizer_config.json).
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation

struct EncoderConfiguration: Decodable {
  let vocabSize: Int
  let hiddenSize: Int
  let intermediateSize: Int
  let numHiddenLayers: Int
  let numAttentionHeads: Int
  let normEpsilon: Float
  let attentionBias: Bool
  let mlpBias: Bool
  let normBias: Bool
  let localAttention: Int
  let maxPositionEmbeddings: Int
  let layerTypes: [String]
  let ropeParameters: [String: RopeConfiguration]

  var headDimension: Int { hiddenSize / numAttentionHeads }

  enum CodingKeys: String, CodingKey {
    case vocabSize = "vocab_size"
    case hiddenSize = "hidden_size"
    case intermediateSize = "intermediate_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case normEpsilon = "norm_eps"
    case attentionBias = "attention_bias"
    case mlpBias = "mlp_bias"
    case normBias = "norm_bias"
    case localAttention = "local_attention"
    case maxPositionEmbeddings = "max_position_embeddings"
    case layerTypes = "layer_types"
    case ropeParameters = "rope_parameters"
  }
}

struct RopeConfiguration: Decodable {
  let ropeTheta: Float

  enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
}

struct AgentConfiguration: Decodable {
  let headLayers: Int
  let maxLength: Int
  let headMaxLength: Int
  let temperatures: [Float]
  let temperaturesByOptions: [String: Float]
  let actionCosts: [String: Float]

  enum CodingKeys: String, CodingKey {
    case headLayers = "head_layers"
    case maxLength = "max_len"
    case headMaxLength = "head_max_len"
    case temperatures = "temperature"
    case temperaturesByOptions = "temperature_by_options"
    case actionCosts = "act_costs"
  }
}

struct ModelConfiguration {
  let encoder: EncoderConfiguration
  let agent: AgentConfiguration

  static func load(from directory: URL) throws -> Self {
    let decoder = JSONDecoder()
    let encoderURL = directory.appending(path: "encoder/config.json")
    let agentURL = directory.appending(path: "rl_agent_config.json")
    guard FileManager.default.fileExists(atPath: encoderURL.path),
      FileManager.default.fileExists(atPath: agentURL.path),
      FileManager.default.fileExists(atPath: directory.appending(path: "model.safetensors").path)
    else { throw HunchError.modelNotPrepared(directory) }

    let encoder = try decoder.decode(EncoderConfiguration.self, from: Data(contentsOf: encoderURL))
    let agent = try decoder.decode(AgentConfiguration.self, from: Data(contentsOf: agentURL))
    guard encoder.hiddenSize % encoder.numAttentionHeads == 0,
      encoder.headDimension.isMultiple(of: 2),
      encoder.layerTypes.count == encoder.numHiddenLayers,
      agent.temperatures.count == 3,
      agent.headMaxLength > 4,
      agent.headMaxLength < agent.maxLength,
      agent.maxLength <= encoder.maxPositionEmbeddings
    else { throw HunchError.invalidModel("The checkpoint configuration is not supported.") }
    return Self(encoder: encoder, agent: agent)
  }
}

/// The tokenizer's special tokens, from `tokenizer_config.json`. ModernBERT uses `[CLS]`/`[SEP]`/`[MASK]`/`[PAD]`;
/// mmBERT (the multilingual checkpoint) uses `<bos>`/`<eos>`/`<mask>`/`<pad>`, so they cannot be hardcoded.
struct SpecialTokens: Equatable, Sendable {
  let cls: String
  let sep: String
  let mask: String
  let pad: String

  static func load(from tokenizerDirectory: URL) throws -> SpecialTokens {
    let url = tokenizerDirectory.appending(path: "tokenizer_config.json")
    guard let data = try? Data(contentsOf: url) else {
      throw HunchError.invalidModel("\(url.path) is missing.")
    }
    return try parse(data)
  }

  static func parse(_ data: Data) throws -> SpecialTokens {
    guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
      throw HunchError.invalidModel("tokenizer_config.json is not a JSON object.")
    }
    // A token is either a string or an AddedToken object with its text in "content".
    func token(_ key: String) throws -> String {
      if let text = object[key] as? String, !text.isEmpty { return text }
      if let added = object[key] as? [String: Any], let text = added["content"] as? String, !text.isEmpty {
        return text
      }
      throw HunchError.invalidModel("tokenizer_config.json has no \(key).")
    }
    return SpecialTokens(
      cls: try token("cls_token"), sep: try token("sep_token"), mask: try token("mask_token"),
      pad: try token("pad_token"))
  }
}
