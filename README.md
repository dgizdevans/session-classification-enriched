# Enriched Baselines for Prefix-Level Session Classification: Limits and Diagnostics

This repository contains the code and documentation for **Experiment 2** of a research project on prefix‑based session type prediction.  
Building on the baseline established in [session-classification-baseline](https://github.com/dgizdevans/session-classification-baseline), this experiment investigates whether adding **inter‑event time intervals** and **global session context** (device, geo, traffic source) improves early classification of e‑commerce sessions.

All large artifacts (datasets, trained models, prediction files) are **not** stored in GitHub. They are available via Google Drive – see the [Reproducibility](#-reproducibility) section below.

## 📁 Repository Structure
```
.
├── exp2.ipynb          # Main notebook with the full experiment pipeline
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Experiment Overview

| **Data**        | BigQuery public GA4 e‑commerce sample (`bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`) |
|-----------------|---------------------------------------------------------------------------------------------------------|
| **Sessions**    | 360,129 sessions defined by `(user_pseudo_id, ga_session_id)`                                           |
| **Classes**     | Buyer / Intent / Researcher / Browser (rule‑based hierarchy)                                             |
| **Task**        | Given the first `t` events of a session, predict the final session type (prefix‑level classification)  |
| **Input**       | Event tokens + **Δt** (inter‑event intervals) + **global context** (device, geo, traffic source)       |
| **Models**      | • Markov‑1,‑2,‑3 (step‑wise backoff)<br>• LightGBM (engineered features, SHAP analysed)<br>• SASRec (transformer) with ablations (Base / +Time / +Context / +Time+Context) |
| **Split**       | Temporal 70/15/15 (based on `session_end_ts`)                                                           |
| **Metric**      | Macro‑F1 (unweighted average over the four classes) computed on **all prefixes** of the test set       |

##  Results Summary (test set)

| Model              | Macro‑F1 | Buyer F1 | Intent F1 | Researcher F1 | Browser F1 |
|--------------------|----------|----------|-----------|---------------|------------|
| LightGBM           | **0.5431** | 0.236    | 0.504     | 0.556         | 0.877      |
| SASRec Base        | 0.4570   | 0.209    | 0.460     | 0.450         | 0.710      |
| Markov‑3           | 0.4221   | 0.111    | 0.284     | 0.442         | 0.852      |

**SASRec ablation results** (val Macro‑F1, mean ± std over 5 runs):  
| Variant         | Val Macro‑F1 |
|-----------------|--------------|
| Base            | 0.4494 ± 0.0045 |
| +Time           | 0.4484 ± 0.0063 |
| +Context        | 0.4328 ± 0.0070 |
| +Time+Context   | 0.4310 ± 0.0095 |

*Detailed per‑class scores, confusion matrices, and prefix‑length analyses are available in the notebook and the artefacts linked below.*

## Key Findings

* **LightGBM** with engineered temporal and contextual features achieves the highest Macro‑F1 (**0.5431**), outperforming both Markov‑3 and SASRec. SHAP analysis confirms that product‑interaction features (`count_view_item`, `count_add_to_cart`) are the primary drivers.
* **Adding time intervals and global context did *not* improve SASRec** – the base model (tokens only) performed best among the transformer variants. The extra signals introduced training instability and slightly degraded overall performance.
* **Buyer‑Intent ambiguity** remains the hardest challenge for all models, especially on short prefixes (`t < 10`). LightGBM shows Buyer→Intent as the largest long‑prefix error flow; SASRec over‑predicts Buyer.
* **Error analysis** reveals distinct behavioural biases:
  * Markov‑3 collapses minority classes into **Browser** (lowest error rate on short prefixes, but severe degradation after `t ≈ 10`).
  * LightGBM balances precision/recall well, but Buyer‑Intent confusion is its dominant long‑prefix error.
  * SASRec over‑predicts **Buyer**, achieving higher recall for the minority class at the cost of low precision.
* **Label‑support shift** with prefix length strongly influences performance dynamics: Browser dominates short prefixes (≈89% at `t ≤ 5`), while longer prefixes are dominated by Buyer/Intent/Researcher. This shift explains much of the Macro‑F1 increase with `t`.

##  Reproducibility

###  Pre‑computed artefacts

All data, trained models, and prediction files are available in a Google Drive folder:

👉 **[Link to Google Drive folder](https://drive.google.com/drive/folders/18h8f1za8S3TEbUOnJHn3jLKWjyjV57rr?usp=sharing)** 👈

The folder contains:
- Session‑level dataset (`sessions.parquet`)
- Vocabulary and preprocessing artefacts
- Trained LightGBM and SASRec models
- Test‑set predictions for all models
- Ablation summaries and confusion matrices

Download the entire folder or individual files as needed.

### ⚙️ Key versioning parameters

The following constants are fixed for Experiment 2 to ensure reproducibility:

| Parameter                 | Value                  | Description                              |
|---------------------------|------------------------|------------------------------------------|
| `T_MAX`                   | 43                     | Maximum prefix length (p95 train length) |
| `SEED`                    | 42                     | Base seed for SASRec multi‑run (5 runs with seeds 42,43,44,45,46) |
| `TPESampler(seed=42)`     | 42                     | Optuna sampler seed                       |
| `Optuna n_jobs`           | 1                      | Sequential trials for reproducibility     |
| `LightGBM n_jobs`         | 1                      | Deterministic training                    |

Class labels: **Buyer=0**, **Intent=1**, **Researcher=2**, **Browser=3**.

###  Running the notebook

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download required artefacts from the Google Drive link above into a local folder (e.g., `./exp2_artifacts/`).
4. Open `exp2.ipynb` in Jupyter / Colab.
5. **For a full re‑run** (≈6‑8 hours, GPU needed for SASRec), remove the two “STOP” cells (before Optuna tuning and before the SASRec training loop).  
   If you only want to evaluate, you can load the pre‑computed predictions and skip the training sections.

## Dependencies

Key packages (full list in `requirements.txt`):
- `torch` ≈ 2.0+
- `scikit‑learn` ≈ 1.2+
- `lightgbm` ≈ 4.0+
- `pandas` ≈ 2.0+
- `numpy` ≈ 1.23+
- `matplotlib` ≈ 3.6+
- `google‑cloud‑bigquery` ≈ 3.0+
- `google‑cloud‑storage` ≈ 2.0+
- `optuna` ≈ 3.0+

##  License

[MIT](LICENSE)


* Google Cloud for the public GA4 sample dataset.
* The open‑source community for the incredible tools used in this work.
