<!-- Changed by Hunch, 2026: rewritten for Hunch. Derived from laya-mlx-swift's README (Apache-2.0); see NOTICE. -->
# Hunch

Local, non-autoregressive typed decisions on Apple silicon. Hunch answers typed questions about a piece of text (pick
one of some options, place it on a scale, say how likely a statement is) in one forward pass of a small encoder, in
tens of milliseconds, without a server, Python or a network call at inference time.

Hunch runs [Laya](https://github.com/NandhaKishorM/laya) checkpoints, the typed-decision models published by Convai
Innovations, with Apple's [MLX](https://github.com/ml-explore/mlx-swift).

- Native Swift library, macOS 14+ on Apple silicon
- Downloads pinned checkpoints from their origin, checked by sha256, only when you call `prepare`
- Inference on a queue of its own, off Swift's cooperative pool; explicit `unload()`
- Calibrated confidences: clamped temperatures, calibration files, per-language temperatures

## Where it comes from

Hunch is a fork of [MstyAI/laya-mlx-swift](https://github.com/MstyAI/laya-mlx-swift), the MLX Swift port of Laya,
which follows [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx), the Python MLX port. The model, the
prompt format and the calibration method are Laya's, by Convai Innovations and the Laya contributors. All are
Apache-2.0. Hunch is not affiliated with or endorsed by Convai Innovations or the Laya project; "Laya" is used only to
say which checkpoints Hunch runs. See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

What changed from laya-mlx-swift: the library is `Hunch`; the API is `Question`/`Answer`/`DecisionModel`; special
tokens are read from the tokenizer config (the multilingual checkpoint answered wrongly with hardcoded BERT tokens);
temperatures are clamped to [0.5, 5]; calibration files and per-language temperatures load; a too-long state can keep
its end; an empty state adds no tokens; MLX errors are thrown instead of ending the process; checkpoints come from
the origin repositories with sha256 checks instead of a third-party conversion; and an opt-in cleaner prepares
terminal text.

## Add the package

```swift
.package(url: "https://github.com/heed-build/hunch", branch: "main")
```

and add the `Hunch` product to your target.

## Prepare a checkpoint once

From a setup step, not on the inference path. The English checkpoint is about 807 MiB.

```swift
import Hunch

let directory = try await ModelStore.prepare(.english) { progress in
  print("\(Int(progress.fractionCompleted * 100))%")
}
```

| Checkpoint | Origin | Encoder | Tokens | Download |
|---|---|---|---|---|
| `.english` | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 512 | 807 MiB |
| `.multilingual` | [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 1024 | 647 MiB |

Each is pinned to a revision with the size and sha256 of every file. `prepare` returns at once when the directory
already holds every file; otherwise it downloads into a temporary directory, checks every sha256, and moves the
directory into place. It never deletes an existing directory: an incomplete one is moved aside. The default location
is `~/Library/Application Support/Hunch/models/<owner>--<name>/<revision>`; pass `at:` to choose another, and
`ModelStore.verify(_:at:)` to re-hash a prepared directory.

The weights and tokenizers are Apache-2.0 (the encoders are ModernBERT, Apache-2.0, and mmBERT, MIT). The
multilingual tokenizer derives from Gemma's, so Hunch never ships it: it is always downloaded from the origin. This
repository contains no weights or tokenizer files.

## Ask

```swift
let model = try await DecisionModel(directory: directory)

let answers = try await model.answer(
  state: "The payment was declined twice and the customer is angry.",
  questions: [
    "route": .choice(
      instructions: "Choose the best destination.",
      options: [
        Option("billing", description: "Payment and invoice problems"),
        Option("technical", description: "Product bugs and outages"),
        "sales",
      ]),
    "urgent": .noul(instructions: "This needs urgent human attention."),
    "mood": .score(instructions: "How upset is the customer?", levels: ["calm", "concerned", "upset"]),
  ])

answers["route"]?.choice        // "billing"
answers["urgent"]?.probability  // P(true)
answers["mood"]?.score          // expected level, 0…2
```

| Question | Use it for | Answer |
|---|---|---|
| `.choice` | routing, picking an action | `choice`, `probabilities` per option label |
| `.score` | placing on an ordered scale | `choice` (likeliest level), `score` (expected level), `probabilities` per level |
| `.noul` | whether a statement holds | `probability` of true, `probabilities` for "true" and "false" |

Every answer also has `confidence` (the model's own, uncalibrated), `calibratedConfidence` (the probability of the
reported answer under the calibration) and `act` (the checkpoint's action head). The checkpoints are a fast base to
specialise, not a zero-shot decision engine: measure on your own labelled data before trusting an answer, and do
not let either confidence gate an irreversible action on its own. Statement-style instructions ("The last line asks
the user to confirm something") work better than questions.

## Options

```swift
let model = try await DecisionModel(
  directory: directory,
  cacheLimit: DecisionModel.defaultCacheLimit,       // MLX buffer cache, process-wide; nil leaves it alone
  options: .init(
    truncation: .keepEnd,                            // keep the end of a state that does not fit
    batchSize: 16,                                   // questions per forward pass
    calibration: try Calibration(contentsOf: url)))  // replaces the checkpoint's own

await model.unload()  // frees the weights; the next answer loads them again
```

- **Calibration.** A checkpoint's temperatures come from its `rl_agent_config.json`. A calibration file in
  laya-mlx's format (`temperature`, `temperature_by_options`, `binning_map`, plus Hunch's optional
  `lang_temperatures`) replaces them. Every temperature is clamped to [0.5, 5]: the English checkpoint ships 0.1 for
  `choice:11+`, which would report a 0.24 top probability as 0.9999. Pass `language:` to `answer` to use a language's
  temperatures.
- **Terminal text.** `TerminalText.clean(_:lastLines:)` strips escape sequences, box drawing, spinners and agent UI
  glyphs, collapses spacing and keeps the last lines (20 by default). On terminal screens this made the difference
  between chance and near-perfect for "is it waiting for the user" on the English checkpoint.
- **Memory.** The English checkpoint peaks near 1.8 GB with the default 256 MiB cache limit.

## Build and test

```bash
swift build
swift test
```

MLX compiles its Metal shaders while building, which needs the Metal toolchain (`xcodebuild -downloadComponent
MetalToolchain` on Xcode 26 and later). The unit tests need no weights and no GPU. The parity tests run laya-mlx's
63-question set against prepared checkpoints and compare token ids and probabilities with laya-mlx's answers:

```bash
HUNCH_ENGLISH_CHECKPOINT=… HUNCH_MULTILINGUAL_CHECKPOINT=… swift test
```

`HUNCH_ENGLISH_CHECKPOINT` also enables the support benchmark in [BENCHMARKS.md](BENCHMARKS.md). An app built with
SwiftPM rather than Xcode must ship `mlx-swift_Cmlx.bundle` (it holds `default.metallib`) next to its executable or in
its `Contents/Resources`.

## License

Apache-2.0, see [LICENSE](LICENSE). Laya, laya-mlx and laya-mlx-swift notices are in [NOTICE](NOTICE); dependency
notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
