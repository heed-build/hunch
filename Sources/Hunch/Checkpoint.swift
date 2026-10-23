// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation

/// A Laya checkpoint on Hugging Face, pinned to a revision, with the size and sha256 of every file Hunch needs.
///
/// The built-in checkpoints point at the origin repositories published by Convai Innovations, not at mirrors or
/// conversions. Their weights are Apache-2.0; the multilingual tokenizer derives from Gemma's, so Hunch never
/// redistributes it: it is always downloaded from the origin.
public struct Checkpoint: Hashable, Sendable {
  public struct File: Hashable, Sendable {
    /// Relative to the repository root, and to the prepared directory.
    public let path: String
    public let size: Int64
    /// Lowercase hex.
    public let sha256: String

    public init(_ path: String, size: Int64, sha256: String) {
      self.path = path
      self.size = size
      self.sha256 = sha256.lowercased()
    }
  }

  /// A short name for logs and directory names, e.g. "english".
  public let name: String
  /// "owner/name" on Hugging Face.
  public let repository: String
  /// A full commit hash.
  public let revision: String
  public let files: [File]

  public init(name: String, repository: String, revision: String, files: [File]) {
    self.name = name
    self.repository = repository
    self.revision = revision
    self.files = files
  }

  public var totalSize: Int64 { files.reduce(0) { $0 + $1.size } }

  /// Where `file` is downloaded from, below a Hugging Face compatible `endpoint`.
  public func url(for file: File, endpoint: URL) -> URL {
    endpoint.appending(path: repository).appending(path: "resolve").appending(path: revision)
      .appending(path: file.path)
  }

  /// `convaiinnovations/laya`: ModernBERT-large, English, 512 tokens. About 807 MiB.
  public static let english = Checkpoint(
    name: "english",
    repository: "convaiinnovations/laya",
    revision: "7b928d828b7b0e022f929d9bd2e44165aa270148",
    files: [
      File(
        "model.safetensors", size: 842_609_210,
        sha256: "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"),
      File(
        "rl_agent_config.json", size: 745,
        sha256: "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd"),
      File(
        "encoder/config.json", size: 2083,
        sha256: "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979"),
      File(
        "tokenizer/tokenizer.json", size: 3_583_228,
        sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
      File(
        "tokenizer/tokenizer_config.json", size: 308,
        sha256: "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12"),
    ]
  )

  /// `convaiinnovations/laya-multilingual`: mmBERT-base, multilingual, 1024 tokens. About 647 MiB.
  public static let multilingual = Checkpoint(
    name: "multilingual",
    repository: "convaiinnovations/laya-multilingual",
    revision: "1720e3e3357cfe1e281542e223f8273b0890ca34",
    files: [
      File(
        "model.safetensors", size: 643_835_514,
        sha256: "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"),
      File(
        "rl_agent_config.json", size: 472,
        sha256: "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6"),
      File(
        "encoder/config.json", size: 1938,
        sha256: "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4"),
      File(
        "tokenizer/tokenizer.json", size: 34_363_188,
        sha256: "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"),
      File(
        "tokenizer/tokenizer_config.json", size: 502,
        sha256: "424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7"),
    ]
  )
}
