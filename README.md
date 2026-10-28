# TabStyle: Tabular-Conditioned Affine Modulation for Echocardiography Video Analysis

TabStyle combines echocardiography videos and clinical tabular features for binary classification and auxiliary regression. Clinical embeddings generate channel-wise scaling and shifting parameters that modulate video features, followed by spatio-temporal self-attention and task-specific prediction heads.

## Study overview

The accompanying study evaluates TabStyle on the FASTRHAC cohort to predict cardiac rhythm two years after baseline examination, distinguishing atrial fibrillation from sinus rhythm, with auxiliary myocardial deformation prediction. Across four random seeds, the study reports an AUROC of **0.87 ± 0.10** and accuracy of **0.72 ± 0.09**. Saliency maps and architectural ablations complement the evaluation. These results describe the study dataset and experiments; running on a different dataset requires adapting the configuration and training a model.

This repository includes preprocessing, training, evaluation, and saliency code. The pretrained checkpoints are also available.

## Installation

Create a dedicated Python environment, then install the dependencies from the repository root:

```bash
conda create -n tabstyle python=3.10
conda activate tabstyle
pip install -r requirements.txt
```

## Prepare your data

Provide a CSV containing identifiers, numerical tabular features, and target columns, plus a folder of `.avi` videos:

```text
datasets/
├── metadata.csv
└── videos/
    ├── case001.avi
    └── case002.avi
```

Each CSV row must correspond to one video file:

| Column          | Requirement                                                                     |
| --------------- | ------------------------------------------------------------------------------- |
| `video_id`      | Unique string matching the AVI filename without `.avi`, such as `case001`.      |
| `video_name`    | Group identifier used to keep related clips in the same split.                  |
| Feature columns | Numerical inputs listed in `data.tabular_features`, in the desired input order. |
| Target columns  | Prediction targets listed after `video_id` in `data.task_cols`.                 |

For example, a CSV for two clinical features and one binary target could contain:

```csv
video_id,video_name,age,sex,outcome
case001,patient001,65,0,1
case002,patient002,58,1,0
```

Use consistent identifier spelling and case between the CSV and filenames. Rows without matching videos are filtered out. Encode categorical features numerically before training; binary inputs and targets should use `0` and `1`. Task types are inferred from observed values: columns containing only binary values are treated as binary tasks, and other numerical targets as regression tasks. General multiclass classification is not implemented.

### Convert DICOM cine loops to AVI

Skip this step if your videos are already prepared. For selected echo DICOM cine loops:

```bash
python preprocessing/prepare_echo.py --input-dir datasets/dicom --output-dir datasets/prepared --remove-ecg
```

Replace the input folder with your own path and use a new or empty output directory. The tool exports AVI files into `datasets/prepared/videos`, review images into `masks/` and `previews/`, and successful identifiers into `manifest.csv`. ECG removal is optional; omit `--remove-ecg` to disable it.

Review the masks and previews before training. Join the manifest to your clinical table by identifier, add the required features and targets, and set `data.video_root` to `datasets/prepared/videos`. The manifest alone is not a training dataset. Masking does not guarantee removal of burned-in identifiers. Use `python preprocessing/prepare_echo.py --help` to see all options. The [preprocessing script](preprocessing/prepare_echo.py) accepts RGB and monochrome DICOM cine loops, preserves available frame timing, and pads frames to square before resizing. If timing is missing, supply `--fps` explicitly; this sets playback timing without resampling. Compressed DICOM may require additional decoder packages. Use `--no-motion-mask` when automatic motion masking is unsuitable.

## Configure TabStyle

Edit the root [config.yaml](config.yaml) before running. The supplied file is a template: its feature names are placeholders, while its target names describe the original study.

| Setting                   | What to configure                                    |
| ------------------------- | ---------------------------------------------------- |
| `data.csv_path`           | Path to your CSV.                                    |
| `data.video_root`         | Folder containing the matching AVI files.            |
| `data.tabular_features`   | Ordered list of your clinical feature columns.       |
| `data.task_cols`          | `video_id` followed by your target columns.          |
| `model.tabular.input_dim` | Number of columns in `data.tabular_features`.        |
| `balance.mode`            | `none` or `video_global`.                            |
| `balance.target_col`      | Binary target used when global balancing is enabled. |
| `train.loss_weights`      | Weights keyed by your target column names.           |
| `train.out_dir`           | A separate output directory for each experiment.     |
| `seed`                    | Random seed for splitting and training.              |

For the example CSV above, update the corresponding fields in the existing configuration:

```yaml
data:
  csv_path: datasets/metadata.csv
  video_root: datasets/videos
  tabular_features: [age, sex]
  task_cols: [video_id, outcome]

model:
  tabular:
    input_dim: 2

balance:
  mode: none
  target_col: outcome

train:
  out_dir: runs/experiment
  loss_weights:
    outcome: 1.0
```

This snippet shows overrides, not a complete configuration; retain the other model and training settings. In particular, the supplied config currently lists **two feature placeholders but sets `model.tabular.input_dim` to 55**. Make these consistent with your dataset.

### Splitting and normalization

The loader splits unique `video_name` groups into training, validation, and test sets. The default validation and test fractions are 0.15 each. Assign related clips the same group; when several videos belong to one patient, use a patient-level group to keep that patient's data together. The current loader does **not** use `data.split_col` or read predefined split assignments from the CSV.

Continuous features and regression targets use mean imputation and min–max scaling fitted on the training split. Binary features are used directly; provide complete binary inputs. Dataset video normalization also derives statistics from training videos. With `balance.mode: video_global`, balancing happens **before** splitting and therefore also changes the validation and test populations. Use `none` to retain their natural distribution.

### Video and model settings

The default configuration uses grayscale videos resized to 128×128 and samples 32 frames per input. Short videos are repeated to reach the clip length. Keep `model.stem.in_channels` consistent with the grayscale setting.

The main multimodal model currently supports `custom3dnet` as its video backbone. For a video-only experiment, set `model.name: videoonly` and `model.use_tabular: false`, keeping `model.use_video: true`. Changing the spatial size, clip length, backbone, or attention pooling can change the representation size; adjust `model.predhead.input_dim` accordingly. Set `sanity_check: true` to check a batch before training.

## Train

With `phase: train` in the configuration, run:

```bash
python train.py --config config.yaml
```

Training selects the best checkpoint using validation performance, with validation loss and minimization as the defaults. Configure the batch size, epochs, learning rate, and early stopping in `train`. Keep `train.save_checkpoints: true` for subsequent evaluation. With `train.eval_after_train: true`, training is followed by evaluation on the test split using the best checkpoint.

Set `train.resume: true` to resume from `last.pt` in the same output directory. Use a different `train.out_dir` for each independent experiment to avoid overwriting checkpoints.

## Evaluate

Set the phase and an explicit checkpoint path in `config.yaml`:

```yaml
phase: test # Use val for validation evaluation.
eval:
  checkpoint: runs/experiment/best.pt
```

Retain the other evaluation settings, then run the same command:

```bash
python train.py --config config.yaml
```

Use an existing checkpoint: `checkpoint: null` currently fails during standalone evaluation, and a nonexistent path causes evaluation of the current model weights instead of loading trained weights.

Keep the dataset, feature order, targets, architecture, seed, and splitting settings consistent with training. The current entry point rebuilds splits and refits preprocessing from the supplied dataset for every phase; it is not a standalone inference interface for a new unlabeled dataset.

`eval.threshold` controls binary decisions. `eval.metric_level` selects clip- or video-level reporting, with video-level predictions grouped by `video_name`.

## Generate saliency maps

Set `phase: saliency`, provide an existing `eval.checkpoint`, and set `saliency.enable: true`. Choose a target in `saliency.saliency_task`; set `saliency.saliency_batches: 1` for an initial small run. Then run:

```bash
python train.py --config config.yaml
```

The saliency phase processes the train, validation, and test loaders. The active evaluation implementation uses Captum-based saliency; options are configured in the `saliency` section.

## Outputs

| Location                  | Contents                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `train.out_dir/best.pt`   | Best validation-selected checkpoint.                                                                    |
| `train.out_dir/last.pt`   | Most recent checkpoint for resuming.                                                                    |
| `train.out_dir/metrics/`  | Overall and per-task evaluation metrics as JSON.                                                        |
| `train.out_dir/saliency/` | Saliency outputs, organized by split, when enabled.                                                     |
| `splits/`                 | Generated group split CSVs, named by balancing mode and seed.                                           |
| Repository root           | `target_normalizer.joblib` and `tabular_feature_normalizer.joblib` when continuous columns are present. |

Normalizer files and split files can be overwritten by later runs with matching names; keep copies with the corresponding experiment.

## Experiment tracking and sweeps

W&B logging is disabled by default. To enable it, authenticate with `wandb login` or `WANDB_API_KEY`, set `wandb.enable: true`, and choose your own project name. Keep credentials outside configuration files.

The `run_sweep/` directory contains experimental sweep scripts. Their paths still refer to the previous repository layout, so they require updates before use. The supported training command above does not depend on them.

## Repository layout

```text
config.yaml        Dataset, model, training, evaluation, and saliency settings
train.py           Main entry point
preprocessing/     DICOM-to-AVI conversion
data/              Dataset loading and balancing
models/            Video/tabular encoders, affine modulation, and attention
training/          Data loaders, training loop, and evaluation
saliency/          Attribution methods and visualization helpers
utils/             Video transforms, losses, task inference, and split utilities
run_sweep/         Experimental W&B sweep scripts
splits/            Generated split assignments
```
