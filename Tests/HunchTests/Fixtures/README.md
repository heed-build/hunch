# Parity fixtures

- `parity-requests.json`: laya-mlx's 63-question parity set (16 states), as plain requests with the options
  already rendered. The cases and the example state and questions come from laya-mlx
  (https://github.com/mizorewww/laya-mlx, Apache-2.0, `benchmarks/common.py`, `examples/`).
- `parity-expected-english.json`, `parity-expected-multilingual.json`: what laya-mlx (Python, float16) answered
  for those requests on the English and multilingual Laya checkpoints: token ids per question, probabilities in
  option order (rounded to 4 places), and the action head's probability. Recorded on 2026-10-22.
- `tokenizer-*/tokenizer_config.json`: minimal configs in the shape of ModernBERT's and mmBERT's, for special-token
  loading. No tokenizer vocabularies or model weights are included here, or anywhere in this repository.

See NOTICE.
