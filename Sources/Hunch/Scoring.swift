// Changed by Hunch, 2026: moved out of LayaAgent.swift (makeAnswer, softmax, entropy confidence) so it can be
// tested without a checkpoint; temperatures come from Calibration (clamped); answers are Hunch's Answer.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation

enum Scoring {
  /// Turns one question's option logits into an answer.
  static func answer(
    kind: QuestionKind, labels: [String], logits: [Float], act: Float, calibration: Calibration,
    language: String? = nil
  ) -> Answer {
    let count = logits.count
    let temperature = calibration.temperature(for: kind, optionCount: count, language: language)
    let probabilities = softmax(logits.map { $0 / temperature })
    let winner = probabilities.indices.max { probabilities[$0] < probabilities[$1] }!
    let top = Double(probabilities[winner])
    let calibrated = calibration.calibratedConfidence(top, kind: kind, optionCount: count, language: language)
    let mapped = Dictionary(uniqueKeysWithValues: zip(labels, probabilities.map(Double.init)))
    switch kind {
    case .choice:
      return Answer(
        choice: labels[winner], probabilities: mapped, confidence: entropyConfidence(probabilities),
        calibratedConfidence: calibrated, act: Double(act))
    case .score:
      let expected = probabilities.enumerated().reduce(0.0) { $0 + Double($1.offset) * Double($1.element) }
      return Answer(
        choice: labels[winner], score: expected, probabilities: mapped,
        confidence: entropyConfidence(probabilities), calibratedConfidence: calibrated, act: Double(act))
    case .noul:
      let yes = Double(probabilities[1])
      return Answer(
        probability: yes, probabilities: mapped, confidence: max(yes, 1 - yes), calibratedConfidence: calibrated,
        act: Double(act))
    }
  }

  static func softmax(_ values: [Float]) -> [Float] {
    let maximum = values.max()!
    let exponentials = values.map { exp($0 - maximum) }
    let total = exponentials.reduce(0, +)
    return exponentials.map { $0 / total }
  }

  /// 1 − H(p) / log(k): how concentrated the whole distribution is. Not calibrated.
  static func entropyConfidence(_ probabilities: [Float]) -> Double {
    guard probabilities.count > 1 else { return 1 }
    let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
    return Double(max(0, min(1, 1 - entropy / log(Float(probabilities.count)))))
  }
}
