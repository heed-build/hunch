// Copyright 2026 Hunch contributors. Licensed under the Apache License, Version 2.0; see LICENSE.

import XCTest

@testable import Hunch

final class TerminalTextTests: XCTestCase {
  func testRemovesEscapeSequences() {
    XCTAssertEqual(TerminalText.removeEscapes("\u{1B}[1;32mok\u{1B}[0m done"), "ok done")
    XCTAssertEqual(TerminalText.removeEscapes("\u{1B}]0;title\u{07}text"), "text")
    XCTAssertEqual(TerminalText.removeEscapes("\u{1B}]8;;https://x\u{1B}\\link\u{1B}]8;;\u{1B}\\"), "link")
    XCTAssertEqual(TerminalText.removeEscapes("a\u{1B}7b\u{1B}8c"), "a7b8c")  // ESC 7 is not in @…_
    XCTAssertEqual(TerminalText.removeEscapes("a\u{1B}Mb"), "ab")
    XCTAssertEqual(TerminalText.removeEscapes("\u{1B}[?25l\u{1B}[2Khidden"), "hidden")
    XCTAssertEqual(TerminalText.removeEscapes("trailing\u{1B}"), "trailing")
  }

  func testRemovesDecoration() {
    XCTAssertEqual(TerminalText.removeDecoration("╭─ box ─╮"), "   box   ")
    XCTAssertEqual(TerminalText.removeDecoration("⠋ Thinking"), "  Thinking")
    XCTAssertEqual(TerminalText.removeDecoration("⏺ Done · ✻ ▌"), "  Done      ")
  }

  func testCleanScreen() {
    let screen = """
      \u{1B}[2J╭──────────────╮
      │ Do you want   │
      │   to proceed? │
      ╰──────────────╯


      ❯ 1. Yes\t\t 2. No   \u{1B}[0m

      """
    XCTAssertEqual(TerminalText.clean(screen), "Do you want\nto proceed?\n\n❯ 1. Yes 2. No")
  }

  func testKeepsTheLastLines() {
    let screen = (1...30).map { "line \($0)" }.joined(separator: "\n")
    XCTAssertEqual(TerminalText.clean(screen, lastLines: 2), "line 29\nline 30")
    XCTAssertEqual(TerminalText.clean(screen).split(separator: "\n").count, 20)
    XCTAssertEqual(TerminalText.clean(screen, lastLines: nil).split(separator: "\n").count, 30)
    XCTAssertEqual(TerminalText.clean(screen, lastLines: 0), "")
  }

  func testWindowsLineEndingsAndEmptyInput() {
    XCTAssertEqual(TerminalText.clean("a\r\nb\r\n"), "a\nb")
    XCTAssertEqual(TerminalText.clean(""), "")
    XCTAssertEqual(TerminalText.clean("\n\n  \n"), "")
  }
}
