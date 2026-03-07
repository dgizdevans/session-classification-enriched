# Enriched Baselines for Prefix-Level Session Classification: Limits and Diagnostics

This repository contains the code for Experiment 2 of a research project on prefix-based session type prediction. The experiment tests whether enriching early session prefixes with temporal intervals (delta t) and global context (device, geo, traffic source) improves session classification, under a unified evaluation protocol across three model families: higher-order Markov chains, LightGBM with engineered features (plus SHAP analysis), and SASRec with temporal/contextual ablations.

Large artifacts (datasets, trained models, prediction files) are not stored in GitHub. The notebook writes artifacts to a local folder (`/content/exp2_artifacts` in Colab) and can optionally back them up to a Google Cloud Storage (GCS) bucket.

All large artifacts (datasets, trained models, prediction files) are **not** stored in GitHub. They are available via Google Drive – see the [Reproducibility](#-reproducibility) section below.

## 📁 Repository Structure
```
.
├── exp2.ipynb          # Main notebook with the full experiment pipeline
├── README.md           # This file
└── requirements.txt    # Python dependencies
```


## Experiment overview

Data:
- BigQuery public GA4 e-commerce sample: `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
- 4,295,584 events, aggregated into 360,129 sessions
- Session key: (`user_pseudo_id`, `ga_session_id`)
- Temporal split by `session_end_ts` (70/15/15): 252,090 train, 54,019 val, 54,020 test

Task:
- Given the first `t` events of a session, predict the final session type (prefix-level classification)
- Prefix lengths: `t = 1..43` where `T_MAX = 43` (p95 train session length)
- Test evaluation covers 54,020 sessions and 497,852 prefixes

Labels (rule-based hierarchy, assigned from the full session):
- Buyer: at least one `purchase`
- Intent: at least one of `add_to_cart`, `begin_checkout`, `add_payment_info`, `add_shipping_info` (but no purchase)
- Researcher: no Buyer/Intent signals, but at least 3 product interactions from `{view_item, view_item_list, select_item}`
- Browser: all remaining sessions

Label distribution (session-level):
- Browser 88.81%, Researcher 5.54%, Intent 4.30%, Buyer 1.35%

Models:
- Markov-2 / Markov-3 (count-based higher-order Markov baselines)
- LightGBM (engineered prefix features, weighted training, SHAP analysis)
- SASRec transformer with ablations: Base, +Time, +Context, +Time+Context

Metric:
- Macro-F1 (unweighted mean over the 4 classes), computed on all test prefixes

## Results summary (test set)

Aggregate results below correspond to the representative prediction artifacts used for cross-model diagnostics (Section 7). SASRec variants use the best-seed run selected by validation Macro-F1 (out of 3 seeds).

| Model | Test Macro-F1 | Buyer F1 | Intent F1 | Researcher F1 | Browser F1 |
|---|---:|---:|---:|---:|---:|
| SASRec +Time+Context | **0.5759** | 0.3334 | 0.4819 | 0.5931 | 0.8952 |
| SASRec Base | 0.5731 | 0.2857 | 0.5207 | 0.5911 | 0.8948 |
| SASRec +Time | 0.5729 | 0.3078 | 0.5100 | 0.5804 | 0.8936 |
| SASRec +Context | 0.5718 | 0.3454 | 0.4569 | 0.5890 | 0.8959 |
| LightGBM | 0.5632 | 0.2974 | 0.4875 | 0.5772 | 0.8906 |
| Markov-3 | 0.4221 | 0.1106 | 0.2843 | 0.4417 | 0.8518 |
| Markov-2 | 0.3794 | 0.0572 | 0.2372 | 0.3857 | 0.8372 |

SASRec ablation stability across 3 seeds (42, 43, 44), mean +/- std:

| Variant | Val Macro-F1 (mean +/- std) | Test Macro-F1 (mean +/- std) |
|---|---:|---:|
| Base | 0.5660 +/- 0.0069 | 0.5668 +/- 0.0047 |
| +Time | **0.5699 +/- 0.0073** | **0.5731 +/- 0.0021** |
| +Context | 0.5673 +/- 0.0004 | 0.5694 +/- 0.0019 |
| +Time+Context | 0.5675 +/- 0.0026 | 0.5688 +/- 0.0059 |

## Key findings

- SASRec variants lead overall, with small differences between Base, +Time, +Context, and +Time+Context.
- LightGBM is competitive but below SASRec on Macro-F1.
- Markov baselines perform substantially worse and rely heavily on Browser predictions.
- Buyer remains the hardest class across all models, and the main non-Browser boundary is Buyer-Intent ambiguity.

## Reproducibility and artifacts

The notebook writes artifacts to a fixed local directory:
- Colab default: `/content/exp2_artifacts/`

Optionally, the notebook also uploads timestamped backups to a GCS bucket (`gs://<your-bucket>/`) for versioned recovery.

### Artifact inventory (high level)

Data:
- `sessions.parquet`, `split_boundaries.json`
- `prefix_train.parquet`, `prefix_val.parquet`, `prefix_test.parquet` (LightGBM feature matrices)

Models:
- `markov_best_alphas.json`
- `lgbm_final.pkl`, `lgbm_best_params.json`
- `sasrec_{variant}_seeds3.pt`, `sasrec_ablation_summary.json`

Predictions:
- `markov2_test_predictions.parquet`, `markov3_test_predictions.parquet`
- `lgbm_test_predictions.parquet`
- `sasrec_{variant}_seed{seed}_test_predictions.parquet`
- `sasrec_{variant}_bestseed_test_predictions.parquet`
- `confusion_matrices.npz`

## Running `exp2.ipynb`

1. Clone the repository.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Open `exp2.ipynb` in Jupyter or Colab.
4. If you want to pull raw data from BigQuery and/or upload backups to GCS, authenticate with Google Cloud in your environment and set the config variables in Section 1.1 (`PROJECT_ID`, `GCS_BUCKET`).

A full rerun executes all cells sequentially. GPU is required for SASRec training in Section 6.

### Key versioning parameters

Fixed constants used in the notebook:
- `T_MAX = 43`
- SASRec `N_RUNS = 3`, seeds `[42, 43, 44]`
- SASRec: `HIDDEN_DIM=64`, `BATCH_SIZE=512`, `dropout=0.1`, `lr=1e-3`, `MAX_EPOCHS=30`, `PATIENCE=3`, Adam (no weight decay), left-pad to `T_MAX`
- Optuna: `TPESampler(seed=42)`, `n_jobs=1`
- LightGBM final fit: `n_jobs=1`
- Label mapping: Buyer=0, Intent=1, Researcher=2, Browser=3

## License

MIT

[MIT](LICENSE)


* Google Cloud for the public GA4 sample dataset.
* The open‑source community for the incredible tools used in this work.
