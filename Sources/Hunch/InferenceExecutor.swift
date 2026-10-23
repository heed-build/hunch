// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Dispatch

/// A serial queue of its own for model loading and inference. MLX evaluation blocks its thread for tens to hundreds
/// of milliseconds; on Swift's cooperative pool that would starve other tasks.
enum InferenceExecutor {
  static func makeQueue() -> DispatchSerialQueue {
    DispatchSerialQueue(label: "build.heed.hunch.inference", qos: .userInitiated)
  }
}
