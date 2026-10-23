// Changed by Hunch, 2026: ported to DecisionModel and Question/Answer; gated on HUNCH_ENGLISH_CHECKPOINT; the
// report goes to /tmp/hunch-support-benchmark.json.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation
import Hunch
import XCTest

final class SupportBenchmarkTests: XCTestCase {
  func testSupportBenchmark() async throws {
    let directory = try checkpointDirectory("HUNCH_ENGLISH_CHECKPOINT")
    let cases = try loadSupportCases()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try await DecisionModel(directory: directory)
    let loadMilliseconds = milliseconds(since: loadStart)

    let warmupStart = DispatchTime.now().uptimeNanoseconds
    _ = try await model.answer(state: cases[0].state, questions: supportQuestions)
    let warmupMilliseconds = milliseconds(since: warmupStart)

    var samples = [SupportSample]()
    for _ in 0..<5 {
      for item in cases {
        let start = DispatchTime.now().uptimeNanoseconds
        let answers = try await model.answer(state: item.state, questions: supportQuestions)
        samples.append(
          SupportSample(
            id: item.id,
            prediction: supportPrediction(from: answers),
            milliseconds: milliseconds(since: start)
          ))
      }
    }

    let report = makeSupportReport(
      cases: cases,
      samples: samples,
      loadMilliseconds: loadMilliseconds,
      warmupMilliseconds: warmupMilliseconds
    )
    let data = try JSONEncoder.pretty.encode(report)
    try data.write(to: URL(fileURLWithPath: "/tmp/hunch-support-benchmark.json"))
    XCTAssertEqual(report.questionAccuracy, 0.74, accuracy: 0.0001)
    XCTAssertEqual(report.repeatStability, 1, accuracy: 0.0001)
  }

  private func loadSupportCases() throws -> [SupportCase] {
    let repository = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let data = try Data(contentsOf: repository.appending(path: "Benchmarks/support_cases.json"))
    return try JSONDecoder().decode([SupportCase].self, from: data)
  }
}

let urgencyLevels = [
  "routine; no time pressure",
  "soon; should be handled within a few days",
  "urgent; blocking work or needs action today",
  "critical; severe security, safety, or widespread outage",
]

let supportQuestions: [String: Question] = [
  "department": .choice(
    instructions: "Which department should handle this customer request?",
    options: [
      Option("account", description: "Sign-in, identity, access, and account settings"),
      Option("billing", description: "Charges, invoices, payments, and refunds"),
      Option("sales", description: "Pricing, plans, purchasing, and product evaluation"),
      Option("shipping", description: "Delivery, tracking, damaged packages, and returns in transit"),
      Option("technical", description: "Bugs, outages, errors, and product malfunctions"),
    ]
  ),
  "refund": .noul(instructions: "Does the customer explicitly request a refund or their money back?"),
  "urgency": .score(instructions: "How urgent is the request?", levels: urgencyLevels),
]

func milliseconds(since start: UInt64) -> Double {
  Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}
