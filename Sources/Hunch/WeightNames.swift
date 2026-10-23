// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

/// Maps a checkpoint's tensor names onto this module tree.
///
/// The origin checkpoints (`convaiinnovations/laya`, `convaiinnovations/laya-multilingual`) are saved from PyTorch:
/// `nn.MultiheadAttention` stores `in_proj_weight`/`in_proj_bias`, and `nn.Sequential` numbers its children
/// directly (`scorer.1.weight`). MLX names them `in_proj.weight` and `scorer.layers.1.weight`. Names that are
/// already in MLX form pass through, so MLX conversions load too. The `temperature` tensor is dropped: calibration
/// comes from `rl_agent_config.json` or a calibration file.
enum WeightNames {
  static func mlxName(_ name: String) -> String? {
    if name == "temperature" { return nil }
    var parts = name.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
    if let last = parts.last, last == "in_proj_weight" || last == "in_proj_bias" {
      parts.removeLast()
      parts += ["in_proj", last == "in_proj_weight" ? "weight" : "bias"]
    }
    if parts.count >= 2, ["scorer", "act_head"].contains(parts[0]), Int(parts[1]) != nil {
      parts.insert("layers", at: 1)
    }
    return parts.joined(separator: ".")
  }
}
