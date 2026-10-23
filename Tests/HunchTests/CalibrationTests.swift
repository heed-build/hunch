// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class CalibrationTests: XCTestCase {
  func testClamp() {
    XCTAssertEqual(Calibration.clamp(0.1006), 0.5)
    XCTAssertEqual(Calibration.clamp(9), 5)
    XCTAssertEqual(Calibration.clamp(1.7), 1.7)
    XCTAssertEqual(Calibration.clamp(.nan), 1)
    XCTAssertEqual(Calibration.clamp(.infinity), 1)
    XCTAssertEqual(Calibration.clamp(-.infinity), 1)
  }

  func testBuckets() {
    XCTAssertEqual(Calibration.bucket(.noul, optionCount: 2), "noul:2")
    XCTAssertEqual(Calibration.bucket(.choice, optionCount: 3), "choice:3-5")
    XCTAssertEqual(Calibration.bucket(.choice, optionCount: 5), "choice:3-5")
    XCTAssertEqual(Calibration.bucket(.score, optionCount: 6), "score:6-10")
    XCTAssertEqual(Calibration.bucket(.choice, optionCount: 10), "choice:6-10")
    XCTAssertEqual(Calibration.bucket(.choice, optionCount: 11), "choice:11+")
  }

  /// The English checkpoint's rl_agent_config.json ships choice:11+ at 0.1006; it must be clamped to 0.5.
  func testCheckpointCalibrationIsClamped() throws {
    let json = #"""
      {"head_layers": 2, "max_len": 512, "head_max_len": 192, "act_costs": {"escalate": 0.5},
       "temperature": [1.6369, 1.2514, 1.9834],
       "temperature_by_options": {"choice:3-5": 1.7602, "choice:11+": 0.10058, "noul:2": 1.9834}}
      """#
    let agent = try JSONDecoder().decode(AgentConfiguration.self, from: Data(json.utf8))
    let calibration = Calibration(agent)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 20), 0.5)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 4), 1.7602)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 7), 1.6369)  // no bucket: per kind
    XCTAssertEqual(calibration.temperature(for: .score, optionCount: 3), 1.2514)
    XCTAssertEqual(calibration.temperature(for: .noul, optionCount: 2), 1.9834)
  }

  func testCalibrationFile() throws {
    let json = #"""
      {"version": 2, "temperature": [2.0, 0.01, 99],
       "temperature_by_options": {"choice:2": 3.0},
       "binning_map": {"noul:2": {"bins": 4, "values": [0.1, 0.2, 0.7, 0.9]}},
       "lang_temperatures": {"de": {"temperature": [4.0, 4.0, 4.0]}, "fr": null,
                             "es": {"temperature_by_options": {"noul:2": 0.7}}}}
      """#
    let calibration = try Calibration(json: Data(json.utf8))
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 2), 3.0)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 3), 2.0)
    XCTAssertEqual(calibration.temperature(for: .score, optionCount: 3), 0.5)
    XCTAssertEqual(calibration.temperature(for: .noul, optionCount: 2), 5)
    // Languages: own temperatures, region subtags ignored, null inherits the base list.
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 2, language: "de-AT"), 4)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 2, language: "fr"), 2)
    XCTAssertEqual(calibration.temperature(for: .noul, optionCount: 2, language: "es"), 0.7)
    XCTAssertEqual(calibration.temperature(for: .score, optionCount: 3, language: "es"), 0.5)
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 2, language: "nl"), 3)
    // Binning: bin = floor(p * bins), capped at the last.
    XCTAssertEqual(calibration.calibratedConfidence(0.55, kind: .noul, optionCount: 2), 0.7)
    XCTAssertEqual(calibration.calibratedConfidence(1.0, kind: .noul, optionCount: 2), 0.9)
    XCTAssertEqual(calibration.calibratedConfidence(0.0, kind: .noul, optionCount: 2), 0.1)
    XCTAssertEqual(calibration.calibratedConfidence(0.55, kind: .choice, optionCount: 2), 0.55)
    // A language override skips the binning map, as laya-mlx does.
    XCTAssertEqual(calibration.calibratedConfidence(0.55, kind: .noul, optionCount: 2, language: "de"), 0.55)
  }

  func testBooleansAndJunkTemperaturesAreNeutral() throws {
    let calibration = try Calibration(json: Data(#"{"temperature": [true, "x", 2]}"#.utf8))
    XCTAssertEqual(calibration.temperature(for: .choice, optionCount: 3), 1)
    XCTAssertEqual(calibration.temperature(for: .score, optionCount: 3), 1)
    XCTAssertEqual(calibration.temperature(for: .noul, optionCount: 2), 2)
  }

  func testMalformedFilesAreRefused() {
    let bad = [
      "[]",
      #"{}"#,
      #"{"temperature": [1, 2]}"#,
      #"{"temperature": {"a": 1, "b": 2, "c": 3}}"#,
      #"{"temperature": [1, 1, 1], "temperature_by_options": [1]}"#,
      #"{"temperature": [1, 1, 1], "binning_map": {"noul:2": {"bins": 2, "values": [0.5]}}}"#,
      #"{"temperature": [1, 1, 1], "binning_map": {"noul:2": {"bins": 1, "values": [1.5]}}}"#,
      #"{"temperature": [1, 1, 1], "lang_temperatures": {"de": 3}}"#,
      #"{"temperature": [1, 1, 1], "lang_temperatures": {"de": {"temperature": [1]}}}"#,
    ]
    for json in bad {
      XCTAssertThrowsError(try Calibration(json: Data(json.utf8)), json)
    }
  }

  func testReadsFromAFile() throws {
    let directory = try temporaryDirectory()
    let url = directory.appending(path: "calibration.json")
    try Data(#"{"temperature": [1.5, 1.5, 1.5]}"#.utf8).write(to: url)
    XCTAssertEqual(try Calibration(contentsOf: url).temperature(for: .noul, optionCount: 2), 1.5)
    XCTAssertThrowsError(try Calibration(contentsOf: directory.appending(path: "missing.json")))
  }
}
