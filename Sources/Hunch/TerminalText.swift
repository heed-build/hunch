// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import Foundation

/// Opt-in cleaning for text captured from a terminal before it becomes a state.
///
/// Escape sequences, box drawing, block elements, braille spinners and agent UI glyphs carry no meaning for the
/// model and crowd out the lines that do; screens also end in the line that decides most questions. Cleaning a
/// screen this way and keeping its last ~20 lines moved "is it waiting for the user" on the English checkpoint from
/// chance to near-perfect in our tests.
public enum TerminalText {
  /// - Parameters:
  ///   - text: Raw terminal output.
  ///   - lastLines: How many lines to keep from the end, or nil for all.
  /// - Returns: The text without escape sequences or decoration, runs of spaces and tabs collapsed, lines trimmed,
  ///   repeated blank lines folded into one, and trailing blank lines dropped.
  public static func clean(_ text: String, lastLines: Int? = 20) -> String {
    let plain = removeDecoration(removeEscapes(text))
    var lines = [String]()
    for raw in plain.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
      let line = collapseSpaces(raw)
      if line.isEmpty, lines.last?.isEmpty ?? true { continue }
      lines.append(line)
    }
    while lines.last?.isEmpty == true { lines.removeLast() }
    if let lastLines { lines = Array(lines.suffix(max(0, lastLines))) }
    return lines.joined(separator: "\n")
  }

  /// Removes CSI sequences (`ESC [ … final`), OSC sequences (`ESC ] … BEL` or `ESC ] … ESC \`) and two-character
  /// escapes (`ESC` followed by `@`…`_`).
  public static func removeEscapes(_ text: String) -> String {
    var output = String.UnicodeScalarView()
    var scalars = text.unicodeScalars[...]
    while let scalar = scalars.popFirst() {
      guard scalar == "\u{1B}" else {
        output.append(scalar)
        continue
      }
      guard let next = scalars.first else { break }
      switch next {
      case "[":
        scalars.removeFirst()
        // Parameter and intermediate bytes, then one final byte in @…~.
        while let byte = scalars.first, (0x20...0x3F).contains(byte.value) { scalars.removeFirst() }
        if let final = scalars.first, (0x40...0x7E).contains(final.value) { scalars.removeFirst() }
      case "]":
        scalars.removeFirst()
        while let byte = scalars.popFirst() {
          if byte == "\u{07}" { break }
          if byte == "\u{1B}", scalars.first == "\\" {
            scalars.removeFirst()
            break
          }
        }
      default:
        if (0x40...0x5F).contains(next.value) { scalars.removeFirst() }
      }
    }
    return String(output)
  }

  /// Replaces box drawing (U+2500–U+257F), block elements (U+2580–U+259F), braille patterns (U+2800–U+28FF) and the
  /// glyphs coding agents draw as bullets and spinners with spaces.
  public static func removeDecoration(_ text: String) -> String {
    String(
      String.UnicodeScalarView(
        text.unicodeScalars.map { isDecoration($0) ? " " : $0 }))
  }

  static let glyphs: Set<Unicode.Scalar> = ["⎿", "⏺", "✻", "✶", "✳", "✢", "✽", "·", "•"]

  static func isDecoration(_ scalar: Unicode.Scalar) -> Bool {
    switch scalar.value {
    case 0x2500...0x259F, 0x2800...0x28FF: true
    default: glyphs.contains(scalar)
    }
  }

  private static func collapseSpaces(_ line: Substring) -> String {
    var output = ""
    var pendingSpace = false
    for character in line {
      if character == " " || character == "\t" {
        pendingSpace = !output.isEmpty
      } else {
        if pendingSpace { output.append(" ") }
        pendingSpace = false
        output.append(character)
      }
    }
    return output
  }
}
