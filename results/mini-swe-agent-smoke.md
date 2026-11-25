# mini-SWE-agent local smoke test

On a MacBook Pro M1 Pro with 16 GB, mini-SWE-agent 2.4.6 was asked to repair a small shell script that added a discount percentage instead of subtracting it. The fixture had two files and one test, which checks that 1,200 cents with a 15% discount produces 1,020 cents. Each run started from a fresh copy. Agent shell commands ran in a macOS sandbox with network disabled and writes restricted to the fixture and its temporary directory. The model endpoint ran on the host. The controller allowed 12 calls and 180 seconds.

| Order | Model | Result | Tool calls | Agent wall time |
| --- | --- | --- | ---: |
| 1 | Gemma 4 12B Instruct Q4_0, resident, zero reasoning budget | No patch; test failed | 12 | 38.65 s |
| 2 | Inkling 2 Mini UD-IQ2_XXS, SSD, 128-token reasoning budget | Patch applied; test passed | 3 | 41.64 s |
| 3 | Inkling 2 Mini UD-IQ2_XXS, SSD, 128-token reasoning budget | Patch applied; test passed | 3 | 43.06 s |
| 4 | Gemma 4 12B Instruct Q4_0, resident, zero reasoning budget | No patch; test failed | 12 | 38.54 s |
| 5 | Gemma 4 12B Instruct Q4_0, resident, 128-token reasoning budget | No patch; test failed | 3 | 88.96 s |
| 6 | Inkling 2 Mini UD-IQ2_XXS, SSD, coding profile default | Patch applied; test passed | 3 | 43.24 s |

Inkling read the two files, used the macOS `sed -i ''` syntax to change the arithmetic, ran the test and submitted. Gemma attempted edits with a `sed` invocation rejected by the macOS tool and reached the step limit. The repeat used greedy decoding and returned the same outcomes. Both servers used 8,192 configured positions, microbatch 512, four threads and one slot. Inkling used a 3,072 MiB expert cache. The two reasoning settings were chosen to make each model's chat response usable with this setup.

With reasoning enabled at 128 tokens, Gemma read the files and then returned two responses without the tool call required by the agent. The controller stopped on repeated format errors. This was one additional run.

The final Inkling run checked the updated `coding` launcher without setting a reasoning override. Its server log confirmed the new default of 128 reasoning tokens. It exercised the same small fixture and does not change the scope of this smoke test.

The first Gemma run reached a peak server RSS of 7,357 MiB and a physical footprint of 1,930 MiB. The first Inkling run reached 4,861 MiB RSS and 4,141 MiB physical footprint. System memory pressure differed between the runs. macOS RSS and footprint count mapped files differently, and these values do not represent total system usage.

This is a single, simple local task and a test of the agent integration. It does not yield a DeepSWE, SWE-bench Pro or full-suite score. [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent/tree/v2.4.6)
