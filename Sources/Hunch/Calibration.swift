// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.
// The temperature buckets, the [0.5, 5] clamp and the binning map follow Laya's calibration
// (https://github.com/NandhaKishorM/laya, Apache-2.0); see NOTICE.

import Foundation

/// How raw scores become probabilities and confidences: a temperature per question kind, optionally per option-count
/// bucket ("choice:3-5") and per language, and an optional binning map that recalibrates the answer's probability.
///
/// A checkpoint carries its own calibration in `rl_agent_config.json`. A calibration file replaces it whole. The file
/// format is laya-mlx's calibration JSON (`temperature`, `temperature_by_options`, `binning_map`), plus an optional
/// `lang_temperatures` object of language → `{temperature, temperature_by_options}`.
///
/// Every temperature is clamped to `temperatureRange`: the shipped checkpoints include fits such as 0.1 for
/// `choice:11+`, which would publish a 0.24 top probability as 0.9999.
public struct Calibration: Hashable, Sendable {
  public static let temperatureRange: ClosedRange<Float> = 0.5...5

  /// Temperatures for one language, or for all of them.
  public struct Temperatures: Hashable, Sendable {
    /// Per question kind.
    public var byKind: [QuestionKind: Float]
    /// Per option-count bucket, e.g. "choice:3-5" or "noul:2"; wins over `byKind`.
    public var byOptions: [String: Float]

    public init(byKind: [QuestionKind: Float] = [:], byOptions: [String: Float] = [:]) {
      self.byKind = byKind.mapValues(Calibration.clamp)
      self.byOptions = byOptions.mapValues(Calibration.clamp)
    }

    func temperature(for kind: QuestionKind, optionCount: Int) -> Float {
      byOptions[Calibration.bucket(kind, optionCount: optionCount)] ?? byKind[kind] ?? 1
    }
  }

  /// A histogram map from a raw answer probability to a calibrated one, for one bucket.
  public struct Binning: Hashable, Sendable {
    public let values: [Double]

    public init(values: [Double]) throws {
      guard !values.isEmpty, values.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
        throw HunchError.invalidCalibration("A binning map needs one or more values in [0, 1].")
      }
      self.values = values
    }

    func apply(_ confidence: Double) -> Double {
      let bin = min(values.count - 1, max(0, Int(confidence * Double(values.count))))
      return values[bin]
    }
  }

  public var temperatures: Temperatures
  /// Per language (an ISO 639-1 code such as "de"); a language override also skips the binning map.
  public var languages: [String: Temperatures]
  /// Per option-count bucket.
  public var binning: [String: Binning]

  public init(
    temperatures: Temperatures = Temperatures(), languages: [String: Temperatures] = [:],
    binning: [String: Binning] = [:]
  ) {
    self.temperatures = temperatures
    self.languages = Dictionary(languages.map { (Self.normalize($0.key), $0.value) }, uniquingKeysWith: { $1 })
    self.binning = binning
  }

  /// No temperature scaling, no binning.
  public static let identity = Calibration(
    temperatures: Temperatures(byKind: [.choice: 1, .score: 1, .noul: 1]))

  /// `t` confined to `temperatureRange`; 1 when it is not a finite number.
  public static func clamp(_ t: Float) -> Float {
    guard t.isFinite else { return 1 }
    return min(temperatureRange.upperBound, max(temperatureRange.lowerBound, t))
  }

  /// The option-count bucket temperatures and binning maps are keyed by: "choice:2", "score:3-5", "noul:2" and so on.
  public static func bucket(_ kind: QuestionKind, optionCount: Int) -> String {
    let size = optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
    return "\(kind.rawValue):\(size)"
  }

  public func temperature(for kind: QuestionKind, optionCount: Int, language: String? = nil) -> Float {
    if let language, let override = languages[Self.normalize(language)] {
      return override.temperature(for: kind, optionCount: optionCount)
    }
    return temperatures.temperature(for: kind, optionCount: optionCount)
  }

  /// The calibrated probability of an answer whose raw probability is `confidence`.
  public func calibratedConfidence(
    _ confidence: Double, kind: QuestionKind, optionCount: Int, language: String? = nil
  ) -> Double {
    let clamped = min(1, max(0, confidence))
    if let language, languages[Self.normalize(language)] != nil { return clamped }
    guard let map = binning[Self.bucket(kind, optionCount: optionCount)] else { return clamped }
    return map.apply(clamped)
  }

  static func normalize(_ language: String) -> String {
    String(language.split(separator: "-").first ?? "").lowercased()
  }
}

// MARK: Reading

extension Calibration {
  /// Reads a calibration file.
  public init(contentsOf url: URL) throws {
    guard let data = try? Data(contentsOf: url) else {
      throw HunchError.invalidCalibration("\(url.path) cannot be read.")
    }
    try self.init(json: data)
  }

  /// Reads laya-mlx's calibration JSON. The shape is checked; out-of-range temperatures are clamped, out-of-range
  /// binning values are refused.
  public init(json data: Data) throws {
    guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
      throw HunchError.invalidCalibration("A calibration must be a JSON object.")
    }
    let temperatures = try Self.temperatures(object, context: "calibration")
    var languages = [String: Temperatures]()
    if let raw = object["lang_temperatures"], !(raw is NSNull) {
      guard let entries = raw as? [String: Any] else {
        throw HunchError.invalidCalibration("lang_temperatures must be an object of language → temperatures.")
      }
      for (language, value) in entries {
        // A null entry or a missing temperature list inherits the base temperatures, as laya-mlx does.
        let entry = value as? [String: Any] ?? [:]
        guard value is NSNull || value is [String: Any] else {
          throw HunchError.invalidCalibration("lang_temperatures.\(language) must be an object.")
        }
        var resolved = try Self.temperatures(
          entry, context: "lang_temperatures.\(language)", required: false)
        if entry["temperature"] == nil || entry["temperature"] is NSNull {
          resolved.byKind = temperatures.byKind
        }
        languages[language] = resolved
      }
    }
    var binning = [String: Binning]()
    if let raw = object["binning_map"], !(raw is NSNull) {
      guard let entries = raw as? [String: Any] else {
        throw HunchError.invalidCalibration("binning_map must be an object of bucket → {bins, values}.")
      }
      for (bucket, value) in entries {
        guard let entry = value as? [String: Any], let bins = entry["bins"] as? Int, bins >= 1,
          let values = (entry["values"] as? [NSNumber])?.map(\.doubleValue), values.count == bins
        else {
          throw HunchError.invalidCalibration("binning_map.\(bucket) must have bins ≥ 1 and that many values.")
        }
        binning[bucket] = try Binning(values: values)
      }
    }
    self.init(temperatures: temperatures, languages: languages, binning: binning)
  }

  /// The checkpoint's own calibration.
  init(_ agent: AgentConfiguration) {
    var byKind = [QuestionKind: Float]()
    for kind in QuestionKind.allCases where kind.index < agent.temperatures.count {
      byKind[kind] = agent.temperatures[kind.index]
    }
    self.init(temperatures: Temperatures(byKind: byKind, byOptions: agent.temperaturesByOptions))
  }

  private static func temperatures(_ object: [String: Any], context: String, required: Bool = true) throws
    -> Temperatures
  {
    var byKind = [QuestionKind: Float]()
    if let raw = object["temperature"], !(raw is NSNull) {
      guard let list = raw as? [Any], list.count == 3 else {
        throw HunchError.invalidCalibration("\(context).temperature must be a list of 3 numbers.")
      }
      for kind in QuestionKind.allCases {
        // A value that is not a number (or is a boolean) falls back to the neutral 1, as the clamp does.
        let number = list[kind.index] as? NSNumber
        let isBool = number.map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? true
        byKind[kind] = isBool ? 1 : number!.floatValue
      }
    } else if required {
      throw HunchError.invalidCalibration("\(context).temperature must be a list of 3 numbers.")
    }
    var byOptions = [String: Float]()
    if let raw = object["temperature_by_options"], !(raw is NSNull) {
      guard let entries = raw as? [String: Any] else {
        throw HunchError.invalidCalibration("\(context).temperature_by_options must be an object of bucket → number.")
      }
      for (bucket, value) in entries {
        let number = value as? NSNumber
        let isBool = number.map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? true
        byOptions[bucket] = isBool ? 1 : number!.floatValue
      }
    }
    return Temperatures(byKind: byKind, byOptions: byOptions)
  }
}
