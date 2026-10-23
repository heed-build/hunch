// Changed by Hunch, 2026: keyed by Question.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

struct PromptPrefixCache {
  private let capacity: Int
  private var values = [Question: PreparedPromptPrefix]()
  private var order = [Question]()

  init(capacity: Int = 128) {
    self.capacity = capacity
  }

  mutating func value(
    for question: Question,
    create: () throws -> PreparedPromptPrefix
  ) rethrows -> PreparedPromptPrefix {
    if let value = values[question] {
      order.removeAll { $0 == question }
      order.append(question)
      return value
    }

    let value = try create()
    if values.count == capacity, let oldest = order.first {
      values.removeValue(forKey: oldest)
      order.removeFirst()
    }
    values[question] = value
    order.append(question)
    return value
  }
}
