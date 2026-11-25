# Django repair pilot on a 16 GB M1 Pro

The selected SWE-bench Verified instance is `django__django-11400`, based on Django commit `1f8382d34d54061eddc41df6994e20ee38c60907`. Its official test patch adds six checks for ordering of related fields. The local baseline on Python 3.11 and SQLite failed those six checks and passed 58 regression checks. The model received the original issue and searched the checkout with a small tool-calling controller. The official test patch was held outside its working copy. [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)

The controller allowed 18 steps. It exposed directory listing, literal source search, numbered file reading, source edits and a local test command. After eight steps it asked the model to attempt an initial edit. Both models used Metal, four threads, one slot, 16,384 configured positions, microbatch 256, F16 keys and values, and the same controller. Inkling IQ2 streamed experts from SSD with a 2,304 MiB cache and a 128-token reasoning budget. Gemma 4 12B Q4_0 used resident weights and zero reasoning budget.

| Model | Steps | Agent wall time | Target checks passed | Regression checks passed |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 Q4_0 | 18 | 187.75 s | 4/6 | 57/58 |
| Inkling IQ2 SSD | 17 | 658.20 s | 4/6 | 57/58 |

Both patches modified `django/contrib/admin/filters.py`. Neither resolved the complete issue. The remaining regression failure was a date-sensitive test that also failed when the unchanged baseline was rerun after midnight. The test suite and baseline were evaluated in a separate offline macOS sandbox with the original test patch applied.

In an earlier controller configuration without the explicit edit prompt, Gemma and both IQ2 budgets 32 and 128 made no changes within 18 steps. IQ3 at budget 128 also made no changes. Their wall times were 113.05 s for Gemma, 428.47 s for IQ2 at 32, 599.88 s for IQ2 at 128, and 857.39 s for IQ3 at 128. The change in controller instructions substantially affected the outcome, so these runs describe the complete model-and-controller setup on one issue. The local Python environment and execution protocol differ from the official SWE-bench container harness.
