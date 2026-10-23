// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class WeightNamesTests: XCTestCase {
  func testOriginNamesMapToMLXNames() {
    XCTAssertEqual(WeightNames.mlxName("head.layers.0.self_attn.in_proj_weight"), "head.layers.0.self_attn.in_proj.weight")
    XCTAssertEqual(WeightNames.mlxName("head.layers.1.self_attn.in_proj_bias"), "head.layers.1.self_attn.in_proj.bias")
    XCTAssertEqual(WeightNames.mlxName("scorer.3.weight"), "scorer.layers.3.weight")
    XCTAssertEqual(WeightNames.mlxName("act_head.0.bias"), "act_head.layers.0.bias")
  }

  func testMLXNamesPassThrough() {
    for name in [
      "head.layers.0.self_attn.in_proj.weight", "scorer.layers.3.weight", "act_head.layers.2.bias",
      "encoder.layers.0.attn.Wqkv.weight", "encoder.embeddings.tok_embeddings.weight", "type_emb.weight",
      "head.layers.0.self_attn.out_proj.bias",
    ] {
      XCTAssertEqual(WeightNames.mlxName(name), name)
    }
  }

  func testTemperatureIsDropped() {
    XCTAssertNil(WeightNames.mlxName("temperature"))
  }
}
