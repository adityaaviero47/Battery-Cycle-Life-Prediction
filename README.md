# Battery Cycle-Life Prediction from Early Cycles

![ci](https://github.com/adityaaviero47/Battery-Cycle-Life-Prediction/actions/workflows/ci.yml/badge.svg)

**The problem:** cycling a lithium-ion cell to end of life takes months, which slows cell R&D, fast-charge protocol design and manufacturing quality checks. Can a cell's lifetime be predicted from its **first 100 cycles**, before its capacity has visibly faded?

**The data:** 124 commercial A123 LFP/graphite cells (APR18650M1A, 1.1 Ah) from Severson et al., *Nature Energy* 4, 383 (2019), MIT/Stanford/Toyota Research Institute.
- The cells were cycled at 30 °C under 72 fast-charge policies, with a 4C discharge.
- **Cycle life** means cycles until capacity drops to 80% of nominal. It ranges from **148 to 2,237 cycles**, averaging 802.
- The per-cell early-life features (ΔQ₁₀₀₋₁₀(V) statistics, fade slopes, internal resistance, charge time, temperature) come from a public extraction of the raw data. It's pinned to commit `41a614e` of `DivyanshuKekatpure/battery` and copied into `data/`.

## Findings

**1. The shape of the discharge curve predicts life; early capacity doesn't.**
- At cycle 100 most cells still hold almost all their capacity, so capacity alone barely relates to lifetime: **r = 0.27** with log life.
- The *variance* of the change in the discharge curve between cycles 10 and 100, ΔQ₁₀₀₋₁₀(V), relates strongly: **r = −0.93**.
- The likely physics: that curve change reflects how the electrodes are degrading, before any capacity loss shows.

**2. The first 100 cycles are enough to predict life to about 10%.**
- **Split:** cells from batches 1–2 are interleaved into **train (42)** and **primary test (42)**. Batch 3, cycled about a year later, is the **secondary test (40)**, a true out-of-batch check.
- **Model choice:** made by 5-fold cross-validation repeated 10 times, **on the training cells only**.
- **Clipping:** each input is clipped to its training range. One cell (b1c0) has an obviously bad capacity value, and without clipping a linear model predicts 0 cycles for it.

| Model | Train CV error | Primary test error | Secondary test error | Secondary RMSE (cycles) |
|---|---|---|---|---|
| Naive: predict the training mean | – | 40.4% | 31.1% | 470 |
| Capacity at cycle 100 + fade slope | 28.5% | 29.9% | 40.3% | 535 |
| ΔQ(V) variance only (1 feature) | 16.3% | 14.6% | 11.4% | 199 |
| Discharge features (elastic net) | 11.7% | 12.8% | 13.2% | 190 |
| Full features (elastic net) | 10.0% | 10.1% | 11.0% | 184 |
| **Full features (gradient boosting), chosen by train CV** | **9.8%** | **9.2%** | **10.6%** | **181** |

Errors are mean absolute percentage error. For reference, the original paper reports 9.1% on its primary test.

**3. Screening weak cells early.**
- A single threshold on the variance feature, set on the training cells, catches **all 22 weak cells** (life under 550 cycles) among the 82 test cells, at **71% precision**.
- It needs only 100 cycles instead of cycling to end of life, which is **87.5% less test time** on this set.

## Run it
```bash
pip install -r requirements.txt
python src/model.py        # results/results.json, results/test_predictions.csv
python -m pytest -q        # split integrity, no leakage, beats baselines
```

## Limits
- **One chemistry:** one cell type at one temperature with a fixed discharge, so the models need re-fitting for other chemistries (such as NMC) or real-world drive cycles.
- **Small test sets:** with 42 and 40 cells, one or two cells move the error by about a point.
- **Secondary set is harder:** batch 3 errors are larger in cycles because its cells live longer.
- **Borrowed features:** they come from a third-party extraction of the raw data, which this repo doesn't re-run. Spot checks match the paper: the r = −0.93 correlation and the 124-cell set.
