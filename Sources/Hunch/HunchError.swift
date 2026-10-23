// Changed by Hunch, 2026: renamed LayaError to HunchError; added calibration, download and checksum cases.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation

public enum HunchError: Error, LocalizedError, Sendable, Equatable {
  case invalidModel(String)
  case invalidQuestion(String)
  case invalidCalibration(String)
  case modelNotPrepared(URL)
  /// MLX reported an error during loading or inference.
  case inferenceFailed(String)
  case downloadFailed(file: String, reason: String)
  case checksumMismatch(file: String, expected: String, actual: String)

  public var errorDescription: String? {
    switch self {
    case .invalidModel(let message), .invalidQuestion(let message), .invalidCalibration(let message),
      .inferenceFailed(let message):
      message
    case .modelNotPrepared(let directory):
      "No checkpoint is prepared at \(directory.path). Call ModelStore.prepare first."
    case .downloadFailed(let file, let reason):
      "Downloading \(file) failed: \(reason)"
    case .checksumMismatch(let file, let expected, let actual):
      "\(file) has sha256 \(actual), expected \(expected)."
    }
  }
}
