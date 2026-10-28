# Forest Fire, Smoke and Non-Fire Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-0A66C2)](#dataset-and-split)
[![Transfer Learning](https://img.shields.io/badge/Transfer-Learning-6F42C1)](#training-protocol)
[![Models](https://img.shields.io/badge/Models-CNN%20%7C%20ViT%20%7C%20Hybrid-00897B)](#models)

This repository contains a reproducible transfer-learning benchmark for the **Deep Neural Networks (HNS)** course at STU FEI. Five architectures are compared on three image classes: `Fire`, `Smoke`, and `Non_Fire`.

## Assignment summary

The assignment requires a comparison of a conventional CNN, an attention-enhanced CNN, an optimized CNN, a visual transformer, and a hybrid convolution-attention model using transfer learning. The dataset must be randomly split into training, validation, and test subsets in the ratio 80%/10%/10%. Each model must be trained three times and statistically evaluated.

For the best checkpoint of every model type, the assignment also requires learning curves for training and validation, test-set Accuracy/Precision/Recall, a confusion matrix, and CNN decision visualizations such as Grad-CAM. Finally, the effect of augmentation and progressive layer unfreezing must be demonstrated for at least one model.

---

## Table of contents

- [Objectives](#objectives)
- [Dataset and split](#dataset-and-split)
- [Models](#models)
- [Training protocol](#training-protocol)
- [Results](#results)
- [Best checkpoint from each model type](#best-checkpoint-from-each-model-type)
- [Detailed results by model](#detailed-results-by-model)
- [Additional M1 experiments](#additional-m1-experiments-augmentation-and-unfreezing)
- [Grad-CAM visual explanations](#grad-cam-visual-explanations)
- [Supplementary evaluation on the complete original test set](#supplementary-evaluation-on-the-complete-original-test-set)
- [Outputs](#outputs)
- [Installation](#installation)
- [Run](#run)
- [Authors](#authors)

---

### List of figures

1. [Figure 1. Learning curves of M1 - ResNet50](#fig-m1-history)
2. [Figure 2. Test confusion matrix of M1 - ResNet50](#fig-m1-cm)
3. [Figure 3. Learning curves of M2 - ResNet50 + CBAM](#fig-m2-history)
4. [Figure 4. Test confusion matrix of M2 - ResNet50 + CBAM](#fig-m2-cm)
5. [Figure 5. Learning curves of M3 - ConvNeXt-Tiny](#fig-m3-history)
6. [Figure 6. Test confusion matrix of M3 - ConvNeXt-Tiny](#fig-m3-cm)
7. [Figure 7. Learning curves of M4 - ViT-Tiny](#fig-m4-history)
8. [Figure 8. Test confusion matrix of M4 - ViT-Tiny](#fig-m4-cm)
9. [Figure 9. Learning curves of M5 - CoAtNet-0](#fig-m5-history)
10. [Figure 10. Test confusion matrix of M5 - CoAtNet-0](#fig-m5-cm)
11. [Figure 11. M1 without supplemental augmentation](#fig-m1-no-augmentation)
12. [Figure 12. M1 with frozen backbone](#fig-m1-frozen)
13. [Figure 13. Grad-CAM: correct M1 prediction](#fig-gradcam-correct)
14. [Figure 14. Grad-CAM: incorrect M1 prediction](#fig-gradcam-incorrect)
15. [Figure 15. M5 on the complete original test set](#fig-m5-full-test-cm)

---

### List of tables

1. [Table 1. Fixed stratified split](#tbl-split)
2. [Table 2. Shared transfer-learning configuration](#tbl-configuration)
3. [Table 3. Mean test metrics over three repeated runs](#tbl-average-results)
4. [Table 4. Checkpoints selected by validation loss](#tbl-best-checkpoints)
5. [Table 5. M1 detailed results](#tbl-m1)
6. [Table 6. M2 detailed results](#tbl-m2)
7. [Table 7. M3 detailed results](#tbl-m3)
8. [Table 8. M4 detailed results](#tbl-m4)
9. [Table 9. M5 detailed results](#tbl-m5)
10. [Table 10. Additional M1 experiments](#tbl-additional-m1)
11. [Table 11. Complete original test-set evaluation](#tbl-full-test)

---

## Objectives

The project:

1. creates one reproducible stratified 80%/10%/10% train/validation/test split;
2. compares a CNN, CNN with attention, optimized CNN, vision transformer, and hybrid architecture;
3. repeats the main experiment three times per model and reports mean and standard deviation;
4. evaluates held-out test data with Accuracy, macro Precision, macro Recall, macro F1-score, and confusion matrices;
5. exports learning curves and Grad-CAM explanations;
6. investigates augmentation and fine-tuning for the baseline CNN.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Dataset and split

Dataset: [Forest Fire Smoke and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)

The downloaded source data was an image dataset organised in folders for `Fire`, `Smoke`, and `Non_Fire`/`non fire`. It contained the image formats JPG/JPEG, PNG, BMP, WEBP and TIFF. The original dataset was already large (more than 14,000 images per class across its supplied folders). Its original `train` and `test` folder names were not used as the final experimental split; the script collected all image files and created one new split required by the assignment.

For a balanced and computationally feasible benchmark, we selected a fixed random sample of **1,000 images from each class**. This follows the concrete 1,000-per-class example in the assignment. It also makes the experiment practical on CPU hardware, keeps every class equally represented, and allows the full five-model comparison with three runs each. The selection and split use seed `42` and are stored in `splits/manifest.csv`.

After the main benchmark, the selected M5 checkpoint was additionally evaluated on all **10,500 images** from the original dataset's supplied `test` folders (3,500 images per class). This is reported as a supplementary experiment at the end of this document. It is not used for model selection or for the primary comparison.

<a id="tbl-split"></a>

**Table 1. Fixed stratified split used by every model and run.**

| Split | Fire | Smoke | Non_Fire | Total |
|---|---:|---:|---:|---:|
| Train | 800 | 800 | 800 | 2,400 |
| Validation | 100 | 100 | 100 | 300 |
| Test | 100 | 100 | 100 | 300 |

The exact same split is used by every model and every repeated run. Training seeds affect the random initialization of the new classification head, batch order, and training augmentation only; they do not change the data split.

The core of the split creation is shown below. `groupby(...).sample(...)` randomly selects the same number of examples from each label, and the two stratified calls preserve class proportions in all subsets.

```python
frame = frame.groupby("label", group_keys=False).sample(
    n=samples_per_class, random_state=seed
).reset_index(drop=True)

train, holdout = train_test_split(
    frame, test_size=0.2, random_state=seed, stratify=frame.label
)
valid, test = train_test_split(
    holdout, test_size=0.5, random_state=seed, stratify=holdout.label
)
```

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Models

| ID | Model | Category |
|---|---|---|
| M1 | ResNet50 | CNN baseline |
| M2 | ResNet50 + CBAM | CNN with channel and spatial attention |
| M3 | ConvNeXt-Tiny | Optimized CNN |
| M4 | ViT-Tiny | Vision transformer |
| M5 | CoAtNet-0 | Hybrid convolution-attention model |

All models use ImageNet pre-trained weights. The original classification layer is replaced by a new three-class head.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Training protocol

<a id="tbl-configuration"></a>

**Table 2. Shared transfer-learning configuration.**

| Parameter | Reported setting |
|---|---|
| Input image size | 224 x 224 pixels |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Batch size | 8 |
| Maximum epochs | 15 |
| Early stopping | Validation-loss patience = 4 |
| Main runs per model | 3 |

Initially, only the new classification head is trainable. With the 15-epoch configuration, all layers are unfrozen before epoch 5, and the learning rate is reduced from `0.0003` to `0.00003`. The best checkpoint is selected only by the lowest validation loss. Test data is used only for the final evaluation.

### Preprocessing and augmentation

Training images are resized to 256 x 256 and then processed with:

- `RandomResizedCrop(224, scale=(0.75, 1.0))`;
- horizontal flip with probability 0.5;
- random rotation from -12 to +12 degrees;
- `ColorJitter(0.15, 0.15, 0.1)` for brightness, contrast, and saturation;
- tensor conversion and ImageNet normalization.

Validation and test images use deterministic preprocessing only: resize, center crop, tensor conversion, and normalization. No random augmentation is applied to validation or test data.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Results

The table shows the mean +- standard deviation across the three main runs. Macro metrics calculate a metric separately for `Fire`, `Smoke`, and `Non_Fire`, then give each class equal weight.

<a id="tbl-average-results"></a>

**Table 3. Mean test metrics over three repeated runs.**

| Model | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| M1 - ResNet50 | 97.89% +- 0.51% | 97.93% +- 0.50% | 97.89% +- 0.51% | 97.89% +- 0.51% |
| M2 - ResNet50 + CBAM | 97.67% +- 0.88% | 97.75% +- 0.78% | 97.67% +- 0.88% | 97.67% +- 0.87% |
| M3 - ConvNeXt-Tiny | 99.00% +- 0.33% | 99.01% +- 0.33% | 99.00% +- 0.33% | 99.00% +- 0.33% |
| M4 - ViT-Tiny | 97.89% +- 0.84% | 97.92% +- 0.84% | 97.89% +- 0.84% | 97.89% +- 0.84% |
| M5 - CoAtNet-0 | **99.33% +- 0.67%** | **99.34% +- 0.65%** | **99.33% +- 0.67%** | **99.33% +- 0.67%** |

### Conclusions

- **CoAtNet-0 (M5)** achieved the best mean test accuracy: **99.33%**.
- **ConvNeXt-Tiny (M3)** was the most stable model, with the lowest standard deviation: **0.33%**.
- The best individual M5 run classified all 300 test images correctly. This is one run and does not guarantee 100% real-world performance.
- The M1 run with a frozen backbone achieved 94.33%, while the main M1 configuration with backbone fine-tuning achieved 97.89% on average. This indicates that fine-tuning pre-trained layers was important for this task.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Best checkpoint from each model type

The best checkpoint is selected by **lowest validation loss**, before inspecting the final test metrics. It is therefore possible for a selected run to have lower test accuracy than another run of the same model; this is methodologically correct because test data must not be used to select a model.

<a id="tbl-best-checkpoints"></a>

**Table 4. Checkpoints selected by the lowest validation loss.**

| Model | Selected run | Test accuracy | Precision macro | Recall macro | F1 macro | Best validation loss | Completed epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 - ResNet50 | 3 | 98.00% | 98.03% | 98.00% | 98.00% | 0.05314 | 14 |
| M2 - ResNet50 + CBAM | 3 | 98.00% | 98.03% | 98.00% | 98.00% | 0.06747 | 12 |
| M3 - ConvNeXt-Tiny | 3 | 99.33% | 99.33% | 99.33% | 99.33% | 0.06546 | 10 |
| M4 - ViT-Tiny | 2 | 97.00% | 97.02% | 97.00% | 97.00% | 0.04082 | 15 |
| M5 - CoAtNet-0 | 3 | 100.00% | 100.00% | 100.00% | 100.00% | 0.03852 | 10 |

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Detailed results by model

All values in the following tables are measured on the same held-out test set of 300 images. `Validation loss` is the lowest validation loss reached during a run; it is the criterion used for checkpoint selection. The main runs all use augmentation and backbone unfreezing.

### M1 - ResNet50

<a id="tbl-m1"></a>

**Table 5. Test results of individual ResNet50 runs.**

| Run | Accuracy | Precision macro | Recall macro | F1 macro | Validation loss | Epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.33% | 98.38% | 98.33% | 98.33% | 0.05687 | 11 |
| 2 | 97.33% | 97.38% | 97.33% | 97.33% | 0.07382 | 15 |
| 3 | 98.00% | 98.03% | 98.00% | 98.00% | 0.05314 | 14 |
| **Mean ± std** | **97.89% ± 0.51%** | **97.93% ± 0.50%** | **97.89% ± 0.51%** | **97.89% ± 0.51%** | - | - |

<a id="fig-m1-history"></a>

![Learning curves of M1 - ResNet50](outputs/20261020_125517/M1_run3/history.png)

*Figure 1. Training and validation loss and accuracy of the selected ResNet50 run (M1, run 3).*

<a id="fig-m1-cm"></a>

![Test confusion matrix of M1 - ResNet50](outputs/20261020_125517/M1_run3/confusion_matrix.png)

*Figure 2. Test-set confusion matrix of the selected ResNet50 run (M1, run 3).*

### M2 - ResNet50 + CBAM

<a id="tbl-m2"></a>

**Table 6. Test results of individual ResNet50 + CBAM runs.**

| Run | Accuracy | Precision macro | Recall macro | F1 macro | Validation loss | Epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96.67% | 96.87% | 96.67% | 96.69% | 0.08093 | 12 |
| 2 | 98.33% | 98.35% | 98.33% | 98.33% | 0.07108 | 12 |
| 3 | 98.00% | 98.03% | 98.00% | 98.00% | 0.06747 | 12 |
| **Mean ± std** | **97.67% ± 0.88%** | **97.75% ± 0.78%** | **97.67% ± 0.88%** | **97.67% ± 0.87%** | - | - |

<a id="fig-m2-history"></a>

![Learning curves of M2 - ResNet50 with CBAM](outputs/20261020_125517/M2_run3/history.png)

*Figure 3. Training and validation loss and accuracy of the selected ResNet50 + CBAM run (M2, run 3).*

<a id="fig-m2-cm"></a>

![Test confusion matrix of M2 - ResNet50 with CBAM](outputs/20261020_125517/M2_run3/confusion_matrix.png)

*Figure 4. Test-set confusion matrix of the selected ResNet50 + CBAM run (M2, run 3).*

### M3 - ConvNeXt-Tiny

<a id="tbl-m3"></a>

**Table 7. Test results of individual ConvNeXt-Tiny runs.**

| Run | Accuracy | Precision macro | Recall macro | F1 macro | Validation loss | Epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 99.00% | 99.00% | 99.00% | 99.00% | 0.06756 | 12 |
| 2 | 98.67% | 98.68% | 98.67% | 98.67% | 0.07283 | 10 |
| 3 | 99.33% | 99.33% | 99.33% | 99.33% | 0.06546 | 10 |
| **Mean ± std** | **99.00% ± 0.33%** | **99.01% ± 0.33%** | **99.00% ± 0.33%** | **99.00% ± 0.33%** | - | - |

<a id="fig-m3-history"></a>

![Learning curves of M3 - ConvNeXt-Tiny](outputs/20261020_125517/M3_run3/history.png)

*Figure 5. Training and validation loss and accuracy of the selected ConvNeXt-Tiny run (M3, run 3).*

<a id="fig-m3-cm"></a>

![Test confusion matrix of M3 - ConvNeXt-Tiny](outputs/20261020_125517/M3_run3/confusion_matrix.png)

*Figure 6. Test-set confusion matrix of the selected ConvNeXt-Tiny run (M3, run 3).*

### M4 - ViT-Tiny

<a id="tbl-m4"></a>

**Table 8. Test results of individual ViT-Tiny runs.**

| Run | Accuracy | Precision macro | Recall macro | F1 macro | Validation loss | Epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.00% | 98.06% | 98.00% | 98.01% | 0.05506 | 12 |
| 2 | 97.00% | 97.02% | 97.00% | 97.00% | 0.04082 | 15 |
| 3 | 98.67% | 98.67% | 98.67% | 98.67% | 0.05865 | 13 |
| **Mean ± std** | **97.89% ± 0.84%** | **97.92% ± 0.84%** | **97.89% ± 0.84%** | **97.89% ± 0.84%** | - | - |

<a id="fig-m4-history"></a>

![Learning curves of M4 - ViT-Tiny](outputs/20261020_125517/M4_run2/history.png)

*Figure 7. Training and validation loss and accuracy of the selected ViT-Tiny run (M4, run 2).*

<a id="fig-m4-cm"></a>

![Test confusion matrix of M4 - ViT-Tiny](outputs/20261020_125517/M4_run2/confusion_matrix.png)

*Figure 8. Test-set confusion matrix of the selected ViT-Tiny run (M4, run 2).*

### M5 - CoAtNet-0

<a id="tbl-m5"></a>

**Table 9. Test results of individual CoAtNet-0 runs.**

| Run | Accuracy | Precision macro | Recall macro | F1 macro | Validation loss | Epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.67% | 98.69% | 98.67% | 98.67% | 0.06188 | 9 |
| 2 | 99.33% | 99.34% | 99.33% | 99.33% | 0.05054 | 11 |
| 3 | 100.00% | 100.00% | 100.00% | 100.00% | 0.03852 | 10 |
| **Mean ± std** | **99.33% ± 0.67%** | **99.34% ± 0.65%** | **99.33% ± 0.67%** | **99.33% ± 0.67%** | - | - |

<a id="fig-m5-history"></a>

![Learning curves of M5 - CoAtNet-0](outputs/20261020_125517/M5_run3/history.png)

*Figure 9. Training and validation loss and accuracy of the selected CoAtNet-0 run (M5, run 3).*

<a id="fig-m5-cm"></a>

![Test confusion matrix of M5 - CoAtNet-0](outputs/20261020_125517/M5_run3/confusion_matrix.png)

*Figure 10. Test-set confusion matrix of the selected CoAtNet-0 run (M5, run 3).*

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Additional M1 experiments: augmentation and unfreezing

The baseline M1 configuration uses supplemental augmentation and progressive backbone unfreezing. Two additional single runs were executed to inspect these choices.

<a id="tbl-additional-m1"></a>

**Table 10. Diagnostic M1 experiments for augmentation and backbone fine-tuning.**

| M1 configuration | Augmentation flag | Backbone unfreezing | Test accuracy | F1 macro | Validation loss | Epochs |
|---|---|---|---:|---:|---:|---:|
| Main M1 benchmark | Flip, rotation, color jitter and random crop | Yes | 97.89% mean | 97.89% mean | - | - |
| Additional run 91 | No flip, rotation, or color jitter; random crop remains | Yes | 97.67% | 97.67% | 0.08099 | 11 |
| Additional run 92 | Full training augmentation | No | 94.33% | 94.34% | 0.11916 | 15 |

The comparison with run 92 shows a clear decrease when the ImageNet backbone remains frozen. In the augmentation comparison, the no-supplemental-augmentation run is close to the main M1 mean. It still retains `RandomResizedCrop`, so it must not be called a completely augmentation-free configuration. The additional configurations are diagnostic single runs, not a full statistical ablation study.

<a id="fig-m1-no-augmentation"></a>

![M1 without supplemental augmentation](outputs/20261020_125517/M1_run91/history.png)

*Figure 11. Learning curves for M1 without supplemental flip, rotation, and color-jitter augmentation.*

<a id="fig-m1-frozen"></a>

![M1 with frozen backbone](outputs/20261020_125517/M1_run92/history.png)

*Figure 12. Learning curves for M1 with a frozen pre-trained backbone.*

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Grad-CAM visual explanations

Grad-CAM was generated from the final convolutional layer of each applicable selected model. For one correctly and, where available, one incorrectly classified test image, it creates a heatmap over the original image.

- Red/yellow regions have the strongest positive influence on the predicted class.
- Green regions have medium influence.
- Blue regions have low influence.

Grad-CAM is **not** a segmentation mask. It does not precisely mark every smoke or fire pixel; it shows which image areas were most important for the model's decision. The project saves `gradcam_correct.png` and, when an error exists, `gradcam_incorrect.png`. M5 run 3 has no incorrect Grad-CAM image because it made no test-set errors.

<a id="fig-gradcam-correct"></a>

![Grad-CAM for a correct M1 prediction](outputs/20261020_125517/M1_run3/gradcam_correct.png)

*Figure 13. Grad-CAM of a correctly classified test image for ResNet50 (M1, run 3).*

<a id="fig-gradcam-incorrect"></a>

![Grad-CAM for an incorrect M1 prediction](outputs/20261020_125517/M1_run3/gradcam_incorrect.png)

*Figure 14. Grad-CAM of an incorrectly classified test image for ResNet50 (M1, run 3).*

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Supplementary evaluation on the complete original test set

The best M5 checkpoint from the main comparison (`M5_run3/weights.pt`) was evaluated without additional training on all original images in the supplied dataset test folders.

<a id="tbl-full-test"></a>

**Table 11. Supplementary M5 evaluation on the complete supplied test folders.**

| Property | Value |
|---|---:|
| Model | M5 - CoAtNet-0, run 3 |
| Number of images | 10,500 |
| Fire | 3,500 |
| Smoke | 3,500 |
| Non_Fire | 3,500 |
| Loss | 0.0545 |
| Accuracy | 98.33% |
| Precision macro | 98.34% |
| Recall macro | 98.33% |
| F1 macro | 98.33% |

The corresponding files are in `outputs/M5_full_test_final/`: `metrics.json`, `predictions.csv`, `original_test_manifest.csv`, and `confusion_matrix.png`.

### Important interpretation note

This result is a useful large-scale supplementary check, but it must **not** be presented as a fully independent external-test result. The main split was created by scanning all raw folders, including the supplied original `test` folders. Consequently, 733 images from those original test folders were already present in the primary 3,000-image sample: 574 in main training, 72 in validation, and 87 in the main held-out test subset. The primary 300-image held-out test results remain the official comparison results because every model used the same fixed protocol. A strictly independent evaluation on the original test folders would require rebuilding the primary split from the original `train` folders only and retraining the models.

<a id="fig-m5-full-test-cm"></a>

![M5 confusion matrix on the complete original test set](outputs/M5_full_test_final/confusion_matrix.png)

*Figure 15. Confusion matrix of M5 on all 10,500 images from the supplied original test folders.*

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Outputs

Every full execution creates a timestamped folder under `outputs/`:

```text
outputs/<timestamp>/
├── runs.csv                 # metrics for all runs
├── summary.csv              # mean and standard deviation for M1-M5
├── best_runs.csv            # lowest-validation-loss run for each model
├── report.md                # generated experiment report
├── M1_run3/
│   ├── history.png          # train/validation loss and accuracy
│   ├── confusion_matrix.png # test-set confusion matrix
│   ├── test_predictions.csv
│   ├── gradcam_correct.png
│   ├── gradcam_incorrect.png
│   └── weights.pt
└── ...
```

If a selected run has no incorrect prediction, no `gradcam_incorrect.png` is generated.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Installation

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download and extract the dataset under `raw_data/`. The script searches recursively for class folders named `Fire`, `Smoke`, and `Non_Fire`; the original Kaggle `train` and `test` folders may remain unchanged.

```text
Forest-Fire-Classification/
├── raw_data/
│   └── FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/
│       ├── train/
│       └── test/
└── experiment.py
```

Supported formats: JPG, JPEG, PNG, BMP, WEBP, TIF, and TIFF.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Run

Create only the split:

```powershell
python experiment.py --prepare-only --samples-per-class 1000 --seed 42
```

Run the reported benchmark:

```powershell
python experiment.py --models M1 M2 M3 M4 M5 --runs 3 --epochs 15 --batch-size 8 --samples-per-class 1000
```

Run a quick environment check:

```powershell
python experiment.py --models M1 --runs 1 --epochs 2 --batch-size 8 --samples-per-class 1000
```

To use every available image instead of the balanced 1,000-per-class sample, use `--samples-per-class 0`.

<p align="right">
<a href="#table-of-contents">Back to contents</a>
</p>

---

## Authors

**Ievgeniia Ievgrafova**
**Yehor Kiriienko**

HNS course project 

Faculty of Electrical Engineering and Information Technology

Slovak University of Technology in Bratislava

2026
