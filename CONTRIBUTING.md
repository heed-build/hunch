<!-- Changed by Hunch, 2026: Hunch's build and test commands. Derived from laya-mlx-swift (Apache-2.0); see NOTICE. -->
# Contributing

Issues and focused pull requests are welcome.

Before opening a pull request:

```bash
swift build
swift test
```

Run the parity tests too when a change touches prompts, tokenization, weights or scoring (see the README).

Keep public API changes small and include tests for prompt formatting, model loading, or output behavior when those contracts change. Never commit model weights or tokenizer files.
