// Changed by Hunch, 2026: replaced LayaRequest/LayaQuestion/LayaAnswer with Question and Answer, shaped like
// heed's Decider (choice, score, noul), and added calibrated confidence.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation

/// A typed question about a state: pick one of some options, place it on an ordered scale, or say how likely a
/// statement is true.
public enum Question: Hashable, Sendable {
  /// One of `options`. Option labels must be unique.
  case choice(instructions: String, options: [Option])
  /// One of `levels`, ordered from lowest to highest. Level names must be unique.
  case score(instructions: String, levels: [String])
  /// Whether the statement in `instructions` is true ("noul" is the checkpoint's name for this type). The two
  /// descriptions replace the checkpoint's default wording of the false and true options.
  case noul(instructions: String, falseDescription: String? = nil, trueDescription: String? = nil)

  public var instructions: String {
    switch self {
    case .choice(let instructions, _), .score(let instructions, _), .noul(let instructions, _, _): instructions
    }
  }

  public var kind: QuestionKind {
    switch self {
    case .choice: .choice
    case .score: .score
    case .noul: .noul
    }
  }
}

/// An option of a choice question. `label` is what an answer reports; `description`, when given, is shown to the
/// model as "label: description".
public struct Option: Hashable, Sendable, ExpressibleByStringLiteral {
  public let label: String
  public let description: String?

  public init(_ label: String, description: String? = nil) {
    self.label = label
    self.description = description
  }

  public init(stringLiteral label: String) {
    self.init(label)
  }
}

extension Array where Element == Option {
  /// Options without descriptions, from their labels.
  public init(labels: [String]) {
    self = labels.map { Option($0) }
  }
}

public enum QuestionKind: String, Codable, Sendable, CaseIterable {
  case choice
  case score
  case noul

  /// The row of the checkpoint's type embedding and of its temperature list.
  var index: Int {
    switch self {
    case .choice: 0
    case .score: 1
    case .noul: 2
    }
  }
}

/// An answer to a `Question`.
///
/// `confidence` is the model's own and uncalibrated (normalized entropy for choice and score, max(p) for noul).
/// `calibratedConfidence` is the probability of the reported answer under the calibration's temperature, passed
/// through its binning map when it has one; it is only as good as the data the calibration was fitted on. Neither
/// should gate an irreversible action on its own.
public struct Answer: Hashable, Sendable, Codable {
  /// choice: the most likely option label; score: the most likely level; noul: nil.
  public var choice: String?
  /// score: the expected level, 0 for the first … count − 1 for the last.
  public var score: Double?
  /// noul: P(true).
  public var probability: Double?
  /// choice and score: per option label or level, summing to 1; noul: "true" and "false".
  public var probabilities: [String: Double]
  /// 0…1, uncalibrated.
  public var confidence: Double
  /// 0…1, the probability mass on the reported answer under the calibration.
  public var calibratedConfidence: Double
  /// The checkpoint's action head: the probability that acting on this answer beats escalating it.
  public var act: Double

  public init(
    choice: String? = nil, score: Double? = nil, probability: Double? = nil, probabilities: [String: Double],
    confidence: Double, calibratedConfidence: Double, act: Double
  ) {
    self.choice = choice
    self.score = score
    self.probability = probability
    self.probabilities = probabilities
    self.confidence = confidence
    self.calibratedConfidence = calibratedConfidence
    self.act = act
  }
}

struct PreparedQuestion: Sendable {
  let id: String
  let kind: QuestionKind
  let tokenIDs: [Int]
  let markers: [Int]
  let labels: [String]
}
