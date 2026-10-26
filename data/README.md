# Data format

Training expects three UTF-8 JSONL files:

```text
DATA_DIR/
  math.jsonl
  code.jsonl
  reason.jsonl
```

Every non-empty line must be a JSON object with a unique, non-empty `id` and a non-empty string `prompt`:

```json
{"id": "sample-1", "prompt": "..."}
```

Extra fields are preserved but ignored during training. The repository does not redistribute the original experiment datasets; users are responsible for obtaining datasets and complying with their licenses.

Evaluation additionally expects:

- Math: `gold.answer`, with model answers written in `\\boxed{...}`.
- Reasoning: `gold.answer`, containing an option letter from A through E.
- Code: `gold.kind`, `gold.inputs`, and `gold.outputs`; call-based tests also use `gold.fn_name`.

The files under `data/example/` are tiny synthetic format examples, not a benchmark.
