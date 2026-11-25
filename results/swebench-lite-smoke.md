# One SWE-bench Lite case on a 16 GB M1 Pro

This experiment uses `pallets__flask-4992`, selected before running either model. The source checkout is Flask at `4c288bc97ea371817199908d0d9b12de9dae327e`. The issue is to let `Config.from_file` open a file in binary mode so Python's TOML loader can read it. The official test patch adds a single test for the new behavior. On the original source it failed with an unexpected `text` keyword, and all 18 existing tests passed. [SWE-bench Lite dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) · [Flask pull request](https://github.com/pallets/flask/pull/4992)

| Model | First attempt | Second attempt | Wall time to green |
| --- | --- | --- | ---: |
| Gemma 4 12B Instruct Q4_0, resident | 18/19 tests | 19/19 tests | 81.79 s |
| Inkling 2 Mini UD-IQ2_XXS, SSD | 18/19 tests | 19/19 tests | 286.40 s |

The controller supplied the original method and applied only the returned replacement for that method. Both models received the same task description, original source, and test feedback format. Each had two greedy attempts of up to 768 generated tokens. Both used Metal, one slot, four threads, 8,192 context positions, and microbatch 512. The Inkling profile used a 3,072 MiB expert cache and an opening `{` in the assistant text block. The default Gemma chat path needed no assistant prefix. The evaluator ran the official `FAIL_TO_PASS` test and all 18 official `PASS_TO_PASS` tests offline under macOS Seatbelt.

Both first attempts added a working binary option under a different name than the `text` keyword used by the official tests. The initial reformulation asked for a compatible option and left its name open. After receiving the failure, Inkling changed the method to use one `text: bool = True` parameter. Gemma kept its `binary` parameter and added a second `text` option. Both passed the tests. The smaller public interface in Inkling’s patch is closer to the single option requested and the interface Flask ultimately shipped.

The controller provided the edit location and ran pytest on macOS. This is a one-case guided patch experiment with the official test data, outside the full SWE-bench evaluation harness.

## Reproduce on Apple Silicon

Use Python 3.11 or later for the TOML test. From the SchwartzStar checkout, prepare the pinned Flask source and a scoring environment. Install `datasets` in the Python environment used to extract the fixture.

```bash
case_dir=$(mktemp -d)
git clone --filter=blob:none https://github.com/pallets/flask.git "$case_dir/flask"
git -C "$case_dir/flask" checkout --detach 4c288bc97ea371817199908d0d9b12de9dae327e
python3 -m venv "$case_dir/score"
"$case_dir/score/bin/python" -m pip install \
  'pytest==7.4.4' 'werkzeug==2.3.7' 'jinja2==3.1.6' 'itsdangerous==2.2.0' click
python3 - "$case_dir/test.patch" <<'PY'
from pathlib import Path
import sys
from datasets import load_dataset

rows = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
case = next(row for row in rows if row["instance_id"] == "pallets__flask-4992")
Path(sys.argv[1]).write_text(case["test_patch"], encoding="utf-8")
PY
```

Start `INKLING_REASONING_BUDGET=0 ./serve.sh coding` and run the probe with `--prefill-json`. For Gemma, stop Inkling first, start `INKLING_CTX=8192 INKLING_BATCH=512 ./serve.sh resident "$GEMMA_MODEL"`, give the probe a fresh output directory, and omit `--prefill-json`.

```bash
python3 tools/swebench_lite_probe.py "$case_dir/inkling-run" \
  --checkout "$case_dir/flask" --test-patch "$case_dir/test.patch" \
  --python "$case_dir/score/bin/python" --prefill-json
```

The probe checks the frozen source and test patch before copying the checkout. It verifies the baseline, limits edits to the target method, and evaluates both attempts in the same isolated copy. Use `--baseline-only` to verify the scorer independently of a model.

## Input depth with surrounding Flask source

This follow-up used only IQ2 SSD and the 32-token reasoning budget selected in the [pilot sweep](humaneval-smoke.md#reasoning-budget-sweep). It prepended source from the same Flask checkout before the target and asked for a one-attempt edit of numbered lines. Microbatch remained 512 and the expert cache 3,072 MiB. Keys used Q4_0 and values F16 for all three runs.

| Configured context | Evaluated input | Prefill | First text | Wall time | Peak RSS | Peak physical footprint | Outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4,096 | 2,968 | 73.36 tok/s | 43.93 s | 90.01 s | 4,584 MiB | 3,758 MiB | Edit rejected |
| 8,192 | 6,168 | 71.76 tok/s | 89.32 s | 136.42 s | 4,661 MiB | 3,900 MiB | Edit rejected |
| 16,384 | 11,968 | 65.62 tok/s | 185.92 s | 233.64 s | 4,803 MiB | 4,210 MiB | Edit rejected |

The model returned the same 229-token proposal in each trial. It tried to replace the entire method, and the bounded replacement string ended inside a docstring. The controller rejected it before running tests. These runs show successful ingestion and the cost of a fresh longer prompt. The edit controller and prompt protocol prevented an end-to-end repair in all three configurations.
