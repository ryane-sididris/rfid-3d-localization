# Chronological Tensor Options for RFID Localization

This note summarizes the discussion around `data/ds_power_tx_rx_freq.csv`, the raw RFID logs, and the kinds of models that are plausible if we want to move beyond a flat tabular baseline.

## 1. Current Situation

The file [data/ds_power_tx_rx_freq.csv](/Users/melchior/RFID_PROJECT/rfid-repo/data/ds_power_tx_rx_freq.csv) was extracted from [notebooks/RTLS_SF_1tower.ipynb](/Users/melchior/RFID_PROJECT/rfid-repo/notebooks/RTLS_SF_1tower.ipynb) by pivoting RSSI values over:

- `txPower_dBm`: 3 values
- `Tx`: 8 values
- `rx`: 4 values
- `freq_MHz`: 50 values

This gives:

- `3 x 8 x 4 x 50 = 4800` RSSI feature cells per sample

The CSV currently has:

- `4411` rows
- `4800` RSSI feature columns
- extra non-feature columns such as `EPC`, `xyz`, `x`, `y`, `z`, `polar`, `nearestAnt`, `nearestD`, `x_pred`, `y_pred`, `z_pred`, `err`, and distance-like columns tied to antennas

Important caveats:

- The pivot uses `aggfunc=max`, so each cell keeps only the maximum RSSI seen for one `(power, Tx, Rx, freq)` slot.
- Missing values were filled with `0`. Those zeros are mostly "no detection / no observation", not true RSSI values.
- The CSV no longer contains `timestamp` or `round`, so exact event chronology is not preserved there.

## 2. Is the CSV Chronological?

Not in the strict temporal sense.

The feature columns are ordered by pivot layout:

- first by power
- then by Tx
- then by Rx
- then by frequency

So the CSV can be reshaped into a structured tensor, but that tensor is:

- ordered by scan dimensions
- not ordered by actual event time

This distinction matters.

If we reshape the CSV directly, the meaningful local ordering is along the frequency axis. That is useful for a frequency-aware model, but it is not a true time series over acquisition time.

Why strict chronology is lost in the CSV:

- the value kept in each cell is a `max` over the run
- if the same slot was observed multiple times, the CSV does not tell us which round produced the max

So from the CSV alone we can reconstruct:

- a canonical scan tensor

but not:

- the exact time at which the stored max happened

## 3. What the Raw Logs Still Contain

The raw `.txt` files contain two event types:

- `RoundStart`
- `TagReadData`

The key point is that `TagReadData` already carries the exact `round` identifier of the `RoundStart` that generated it. That means we do not have to align detections to rounds using approximate nearest timestamps. We can use `round` directly.

In one inspected sample file:

- `timestamp` was monotonic
- `round` was monotonic
- per-tag detections were monotonic in `round`

Across the raw dataset:

- there are `243` `pgm=AEP_OK` files
- round counts vary from about `157` to `311` per file
- the median is around `208` rounds per file

That variability is manageable for padded sequence models.

## 4. Dataset Representations We Discussed

### A. Flat tabular baseline

This is the current ExtraTrees-style representation:

- input shape: `(N, 4800)`
- `N = 4411`

Use when:

- keeping a strong non-neural baseline
- prioritizing robustness over structured modeling

Main drawback:

- throws away the fact that frequency bins are ordered

### B. Frequency-structured tensor from the CSV

This keeps exactly the same information as `ds_power_tx_rx_freq.csv`, but in a more meaningful shape:

- canonical shape: `(N, 3, 8, 4, 50)`

Axis meanings:

- `3`: power levels
- `8`: Tx choices
- `4`: Rx choices
- `50`: frequency bins

This can also be reshaped to:

- `(N, 96, 50)` where `96 = 3 x 8 x 4`

In that representation:

- each of the `96` channels is one fixed `(power, Tx, Rx)` slice
- the `50` values are ordered by frequency

We called one fixed `(power, Tx, Rx)` slice a **path**.

So:

- number of paths: `96`
- path shape: `(50,)`

Recommended companion tensors:

- `X_val`: RSSI values
- `X_mask`: same shape, `1` if observed, `0` if filled by missingness

Optionally, if rebuilt from raw rather than CSV:

- `X_count`: number of detections contributing to that cell
- `X_mean`
- `X_argmax_round`: normalized round index at which the max occurred

This representation is not strictly chronological, but it is ordered in a physically meaningful way along frequency.

### C. Rich fixed-size tensor rebuilt from raw
C'est la proposition de codex mais jsp si c'est pertinent

This is the most attractive compromise if we want:

- fixed-size tensors
- no detection explosion
- a little chronology

Instead of storing only `max` per `(power, Tx, Rx, freq)`, rebuild the tensor from raw and store several channels per cell:

- `max_rssi_dBm`
- `mean_rssi_dBm`
- `count_reads`
- `mask`
- `argmax_round_norm`
- optionally `first_round_norm` and `last_round_norm`

Shape:

- `(N, 3, 8, 4, 50, C)` where `C` is the number of channels

or after folding the path axes:

- `(N, 96, 50, C)`

This keeps the fixed-size simplicity of the CSV while injecting some timing information.

### D. True chronological per-round sequence

This is the real sequential dataset.

Build one sequence per sample:

- sample = one `(run, EPC)` or one `(xyz, EPC, power)`
- step = one `RoundStart`, ordered by `round`

For each round, aggregate detections for the target EPC into fixed per-step features such as:

- `freq_MHz`
- `Tx`
- `rx1_ant`, `rx2_ant`
- `has_read1`, `has_read2`
- `rssi1_max_dBm`, `rssi2_max_dBm`
- `rssi1_mean_dBm`, `rssi2_mean_dBm`
- `count_reads`
- optionally `delta_t`

Typical padded shape:

- `(N_seq, L_max, F_step)`

where:

- `L_max` is max number of rounds across sequences, about `311` in the current data
- `F_step` is the number of per-round features

If all three powers must be retained jointly, another shape is:

- `(N, 3, L_max, F_step)`

This representation preserves actual acquisition order at the round level.

### E. Every-detection event sequence

This is possible, but was not the preferred first option.

Representation:

- one step = one `TagReadData`
- sort by `(round, timestamp)`
- include `delta_t` explicitly

This handles irregular timing honestly, but has drawbacks:

- variable sequence length can become large
- much noisier than round-level aggregation
- harder to compare fairly across runs

The concern about uneven timing is valid. A sequence model does not require uniform spacing if:

- event order is preserved
- and time deltas are provided as input features

Still, round-level aggregation is cleaner for a first pass.

## 5. Model Options

### ExtraTrees baseline

This remains the strongest baseline to beat.

Why keep it:

- handles sparse high-dimensional tabular data well
- low tuning burden
- likely robust with only `4411` samples

### Conv1D over frequency plus dense fusion

This is the first neural candidate for the CSV-based tensor.

Idea:

- reshape the input to `(N, 96, 50)` or equivalent
- run a small `Conv1D` over the `50` frequency bins
- use a dense head to combine all paths and predict `(x, y, z)`

Important clarification:

- `Conv1D` is only the frequency encoder
- it is not the whole model

### Shared Conv1D per path plus cross-path fusion

This is the more principled neural design.

Pipeline:

1. Take each path `(power, Tx, Rx)` independently as a length-50 sequence.
2. Apply the same small `Conv1D` encoder to every path.
3. Obtain one embedding vector per path.
4. Fuse the `96` path embeddings.
5. Predict `(x, y, z)`.

Possible fusion mechanisms:

- simplest: concatenate embeddings and use an MLP
- richer: self-attention across path embeddings

So "cross-path fusion" is not a separate magical model. It is the stage after the per-path frequency encoder, and it is typically:

- either an MLP
- or an attention block

For this dataset size, the recommended first version is:

- shared `Conv1D` per path
- MLP fusion head

### TCN over round sequences

This is the preferred model for the true chronological dataset.

Pipeline:

- input: `(N_seq, L_max, F_step)`
- several `Conv1D` / dilated temporal blocks over the round axis
- pooling over time
- MLP regression head

Why TCN first:

- simpler than LSTM/Transformer
- handles ordered sequences well
- usually easier to train on medium-sized data

### GRU / LSTM

These remain possible for round sequences or event sequences, but were not the recommended first option.

Why not first:

- more parameter-sensitive
- not obviously better than a TCN here
- the data is sparse and not extremely long

## 6. Main Drawbacks and Risks

- The current CSV is not truly chronological.
- Aggregating with `max` may lose stability information and timing information.
- Flattening all `4800` values into one long pseudo-sequence creates artificial neighbors between unrelated blocks.
- Zeros are structural missingness and should not be treated as real signal values.
- Previous neural attempts on coarse aggregated Tx/Rx features underperformed, so a neural win is not guaranteed.
- The dataset is small enough that ExtraTrees may still outperform neural models.

Evaluation risk:

- many rows share the same `xyz`
- random row-level train/test splits may leak location-specific structure

So validation should be grouped by:

- `xyz`
- or `run`
- depending on the intended deployment setup

## 7. Recommended Next Steps

1. Keep a grouped ExtraTrees baseline on the current flat features.

2. Build a clean CSV-to-tensor extractor for the canonical frequency tensor:

- values: `(N, 96, 50)`
- mask: `(N, 96, 50)`

3. Train the first neural baseline:

- shared `Conv1D` per path
- MLP fusion head

4. If neural performance is not competitive, rebuild the richer fixed-size tensor from raw with:

- `max`
- `mean`
- `count`
- `mask`
- `argmax_round_norm`

5. If the real goal is chronology rather than frequency structure, build the per-round sequence dataset and test:

- TCN over rounds

## 8. Practical Recommendation

The most pragmatic order of work is:

- first: frequency-structured tensor from the current CSV
- second: richer fixed-size tensor from raw
- third: true chronological per-round sequence dataset

This keeps the work incremental:

- no data explosion
- minimal first implementation cost
- direct comparison against ExtraTrees

In short:

- if the goal is to exploit the existing 4800-feature extraction better, use a frequency-structured tensor and a `Conv1D`-based encoder with MLP fusion
- if the goal is to preserve actual scan order, rebuild from raw using `round` and feed a per-round sequence into a TCN
