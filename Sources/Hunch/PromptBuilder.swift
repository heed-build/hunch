// Changed by Hunch, 2026: special tokens come from tokenizer_config.json instead of being hardcoded; the tokenizer
// sits behind TokenEncoder so prompts can be tested without a checkpoint; added keep-end truncation; an empty state
// adds no tokens; adapted to Question.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import Foundation

/// Which end of a state that does not fit the checkpoint's token budget survives.
public enum Truncation: String, Sendable, Codable {
  /// Keep the beginning (the reference implementation's behaviour for text states).
  case keepStart
  /// Keep the end: terminal screens and logs end in the line that decides the question.
  case keepEnd
}

/// What prompt building needs from a tokenizer.
protocol TokenEncoder {
  /// Token ids for `text`, without special tokens.
  func encode(_ text: String) -> [Int]
  func tokenID(_ token: String) -> Int?
}

/// Builds the checkpoint's input: `[CLS] <type> question: instructions [SEP] [MASK] option … [SEP] state [SEP]`.
struct PromptBuilder {
  let encoder: any TokenEncoder
  let maxLength: Int
  let headMaxLength: Int
  let classTokenID: Int
  let separatorTokenID: Int
  let maskTokenID: Int
  let maskToken: String
  let truncation: Truncation

  init(
    encoder: any TokenEncoder, maxLength: Int, headMaxLength: Int, specials: SpecialTokens,
    truncation: Truncation = .keepStart
  ) throws {
    self.encoder = encoder
    self.truncation = truncation
    self.maxLength = maxLength
    self.headMaxLength = headMaxLength
    guard let classTokenID = encoder.tokenID(specials.cls),
      let separatorTokenID = encoder.tokenID(specials.sep),
      let maskTokenID = encoder.tokenID(specials.mask)
    else { throw HunchError.invalidModel("The tokenizer is missing its class, separator or mask token.") }
    self.classTokenID = classTokenID
    self.separatorTokenID = separatorTokenID
    self.maskTokenID = maskTokenID
    maskToken = specials.mask
  }

  func makePrefix(for question: Question) throws -> PreparedPromptPrefix {
    let definition = try QuestionDefinition(question)
    let (tokens, markers) = tokenizePrefix(definition)
    return PreparedPromptPrefix(
      kind: definition.kind,
      tokens: tokens,
      markers: markers,
      labels: definition.labels,
      optionCount: definition.options.count
    )
  }

  /// The state's tokens. An empty state has none: some tokenizers (mmBERT's) encode "" as a lone space token,
  /// which the reference never adds.
  func tokenizeState(_ state: String) -> [Int] {
    guard !state.isEmpty else { return [] }
    return encoder.encode(state.replacingOccurrences(of: maskToken, with: " "))
  }

  func prepare(id: String, stateTokens: [Int], prefix: PreparedPromptPrefix) throws -> PreparedQuestion {
    guard prefix.markers.count == prefix.optionCount else {
      throw HunchError.invalidQuestion("Question '\(id)' has too many options for the token budget.")
    }
    var tokens = prefix.tokens
    let room = max(0, maxLength - tokens.count - 1)
    switch truncation {
    case .keepStart: tokens += stateTokens.prefix(room)
    case .keepEnd: tokens += stateTokens.suffix(room)
    }
    tokens.append(separatorTokenID)
    return PreparedQuestion(
      id: id,
      kind: prefix.kind,
      tokenIDs: Array(tokens.prefix(maxLength)),
      markers: prefix.markers.filter { $0 < maxLength },
      labels: prefix.labels
    )
  }

  private func tokenizePrefix(_ question: QuestionDefinition) -> ([Int], [Int]) {
    let cleanInstructions = question.instructions.replacingOccurrences(of: maskToken, with: " ")
    var heading = encoder.encode("\(question.kind.rawValue) question: \(cleanInstructions)")
    var options = question.options.map { option in
      [maskTokenID] + encoder.encode(" " + option.replacingOccurrences(of: maskToken, with: " ")).prefix(48)
    }
    var headingBudget = headMaxLength - options.reduce(0) { $0 + $1.count }
    if headingBudget < 16 {
      let perOption = max(4, (headMaxLength - 16) / max(1, options.count))
      options = options.map { Array($0.prefix(perOption)) }
      headingBudget = headMaxLength - options.reduce(0) { $0 + $1.count }
    }
    heading = Array(heading.prefix(max(8, headingBudget)))

    var tokens = [classTokenID] + heading + [separatorTokenID]
    var markers = [Int]()
    for option in options {
      markers.append(tokens.count)
      tokens += option
    }
    tokens.append(separatorTokenID)
    return (tokens, markers)
  }
}

struct PreparedPromptPrefix: Sendable {
  let kind: QuestionKind
  let tokens: [Int]
  let markers: [Int]
  let labels: [String]
  let optionCount: Int
}

/// A question as the checkpoint sees it: rendered option texts, and the labels answers report.
struct QuestionDefinition {
  let kind: QuestionKind
  let instructions: String
  let options: [String]
  let labels: [String]

  init(_ question: Question) throws {
    switch question {
    case .choice(let instructions, let options):
      guard !options.isEmpty else {
        throw HunchError.invalidQuestion("Choice options cannot be empty.")
      }
      let labels = options.map(\.label)
      guard Set(labels).count == labels.count else {
        throw HunchError.invalidQuestion("Choice option labels must be unique.")
      }
      kind = .choice
      self.instructions = instructions
      self.labels = labels
      self.options = options.map {
        guard let description = $0.description, !description.isEmpty else { return $0.label }
        return "\($0.label): \(description)"
      }
    case .score(let instructions, let levels):
      guard !levels.isEmpty else {
        throw HunchError.invalidQuestion("Score levels cannot be empty.")
      }
      guard Set(levels).count == levels.count else {
        throw HunchError.invalidQuestion("Score levels must be unique.")
      }
      kind = .score
      self.instructions = instructions
      options = levels.enumerated().map { "level \($0.offset): \($0.element)" }
      labels = levels
    case .noul(let instructions, let falseDescription, let trueDescription):
      kind = .noul
      self.instructions = instructions
      options = [
        "false: \(falseDescription ?? "no, the statement does not hold")",
        "true: \(trueDescription ?? "yes, the statement holds")",
      ]
      labels = ["false", "true"]
    }
  }
}
