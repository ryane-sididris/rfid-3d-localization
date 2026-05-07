Explanation video:
https://drive.google.com/drive/folders/1EUJafUOZl9oDjFGUfmzG_RWv7Po1pbLr?usp=sharing



This branch contains multiple experiments sharing files

General structure:

Expirements are in `experiments/`.

Scripts to reproduce the results are generally named `run_*.py` and use definitions from `src/`.

Explanations and summaries of results are available in `experiments/results`



## Main folders

- [`experiments/`](experiments/) contains scripts to run.
- [`experiments/results/`](experiments/results/) contains result notes, tables, text outputs, and plots.
- [`src/loaders/`](src/loaders/) reads CSV files and prepares inputs and targets.
- [`src/models/`](src/models/) contains the models.
- [`src/evaluation/`](src/evaluation/) contains shared evaluation code. 
The main metrics are in [`src/evaluation/metrics.py`](src/evaluation/metrics.py) and conventions are clarified for fair comparisons.
- [`src/validation/cross_validation.py`](src/validation/cross_validation.py) contains the shared validation code.
- [`src/config/antennas.py`](src/config/antennas.py) contains antenna names and positions for the datasets used throughout the project.
- [`data/`](data/) is where datasets are expected. In the current repo, only `.gitkeep` is tracked there.

## Experiments summaries in markdown

Experiments that have a markdown note are listed here. The matching model or helper code is usually in [`src/models/`](src/models/).

### Direct position vs distance-first

Scripts: [`experiments/run_position.py`](experiments/run_position.py) and [`experiments/run_distance.py`](experiments/run_distance.py)

Summary: [`experiments/results/direct_and_trilat.md`](experiments/results/direct_and_trilat.md)

This compares two simple setups. One predicts `(x, y, z)` directly from the input features. The other predicts distances to antennas first, then uses trilateration to yield a final position.

### Tx/Rx delta regression

Scripts: [`experiments/run_tx_rx_global_regression.py`](experiments/run_tx_rx_global_regression.py) and [`experiments/run_tx_rx_per_antenna_regression.py`](experiments/run_tx_rx_per_antenna_regression.py)

Summary: [`experiments/results/tx_rx_delta_regression.md`](experiments/results/tx_rx_delta_regression.md)

This predicts position offsets from the antennas instead of predicting the final position directly. One version uses one model for all antennas together. The other trains one model per antenna.

### Classification then regression

Script: [`experiments/run_classif_then_regression.py`](experiments/run_classif_then_regression.py)

Summary: [`experiments/results/classif_then_regression.md`](experiments/results/classif_then_regression.md)

This first predicts a zone, then uses that information to predict offsets and yield the final position.

### Keras direct position

Script: [`experiments/run_keras_position.py`](experiments/run_keras_position.py)

Summary: [`experiments/results/keras_position.md`](experiments/results/keras_position.md)

This tries a few Keras models that predict `(x, y, z)` directly.

### Keras convolutional network

Script: [`experiments/run_keras_freq_position.py`](experiments/run_keras_freq_position.py)

Summary: [`experiments/results/keras_freq_position.md`](experiments/results/keras_freq_position.md)

This reshapes the frequency data into many paths, runs the same Conv1D on each path, then combines everything to predict `(x, y, z)`.

### Ratio position solver

Scripts: [`experiments/run_ratio_position_solver_linear.py`](experiments/run_ratio_position_solver_linear.py) and [`experiments/run_ratio_position_solver_no_fusion_linear.py`](experiments/run_ratio_position_solver_no_fusion_linear.py)

Summary: [`experiments/results/ratio_position_solver_summary.md`](experiments/results/ratio_position_solver_summary.md)

This uses ratios of signals between Rx antennas and a least squares solver to estimate position. The two versions differ in how they combine pairs of ratio directions.

### Tx barycenter heuristic

Script: [`experiments/run_tx_barycenter_position.py`](experiments/run_tx_barycenter_position.py)

Summary: [`experiments/results/tx_barycenter_heuristic.md`](experiments/results/tx_barycenter_heuristic.md)

This is a very simple baseline. It builds scores per Tx using RSSI and power, uses barycenters them to estimate `x` and `y`, and predicts `z` separately with a small extra model.

### Zone regression

Script: [`experiments/run_zone_regression.py`](experiments/run_zone_regression.py)

Summary: [`experiments/results/zone_regression.md`](experiments/results/zone_regression.md)

This uses a global model to assign the tag to an area first, then uses the predictions of regressors trained only on samples in their area. The goal is to see if local models help once the sample is routed to the right place.
