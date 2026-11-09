# AP304 frozen ranking model

This experiment uses **best only** (18,357,945 parameters, 73,534,168 bytes), stored with Git LFS. Run `git lfs pull` after cloning. The unused `last` checkpoint is intentionally omitted.

- Weights SHA256: `f6dfe88f7e25d5be934ee0b065ae76d94aa89edc88340164244ecd149fa89928`
- Normalization SHA256: `7ee5781e253ca45c2f31176e11ec5f065fd529df8afd9dc80bde5e3fec9d2de6`
- Source: user-provided `ap304_full_raw_rank_weights_20261028.zip`; original provenance/configuration retained here.
- Output: dimensionless per-finger ranking scores, **not force N**. Both experiment arms fit score→N mappings on test5.

**Input boundary:** MCAP raw uV requires the existing training-format causal EMG filter before entering `RankPredictor`; the bundle itself does not filter. Apply this exactly once using the experiment extractor. Passing raw DC-biased MCAP values directly produced invalid near-flat predictions. The checked-in report uses corrected preprocessing.

`UPSTREAM_README.md` describes the original two-checkpoint archive. Its references to `last` and its ambiguous “raw/no filtering” wording must be read with the scope and input boundary above. `MANIFEST.json` preserves the original manifest; only the best/runtime files are needed by its per-file inference checks.

Runtime files are copied unchanged from the supplied archive. No model retraining or weight editing was performed.
