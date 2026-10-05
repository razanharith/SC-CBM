# SC-CBM: A Self-Auditing Concept Bottleneck Model for Interpretable Skin Lesion Diagnosis

## Overview

SC-CBM is a trained concept bottleneck for dermoscopic melanoma diagnosis built on **one quantity used twice**. The fraction of a concept's activation energy that falls inside the lesion, its *coverage*, acts during training as an asymmetric regulariser that penalises only out-of-lesion evidence, and at inference as a per-case audit score that marks individual concept assertions as reliable or suspicious. The training objective and the deployed monitor are not two mechanisms kept in sync; they are one expression evaluated at two times.

The second role does not follow from the first, and we show this rather than assert it: under background corruption every configuration degrades alike, so spatial faithfulness in the weights is not robustness. Faithfulness that holds on average over a training distribution is not a guarantee for one patient, which is why the evidence has to be reported case by case.

**Key result:** in-lesion concept coherence rises from 50.9% to 89.3% on PH² at no significant diagnosis cost (p = 0.68), and the audit score separates correct from incorrect concept assertions with an AUROC of 0.85, against 0.67 for the same model trained without the spatial term.

**8,218 trainable parameters | 8.54M frozen | 4.5 ms/image | 223 img/s | no language model**

## Pipeline

SC-CBM keeps the standard concept-bottleneck factorisation x → c → y and constrains where each concept reads its evidence, then reports that constraint at test time:

```
Input (224×224×3)
    ↓  Frozen ResNet-50 (to layer3)           → F  [1024, 14, 14]
    ↓  1×1 conv concept layer                 → A  [8, 14, 14]   concept-logit maps
    ↓  Attention pooling + sigmoid            → c  [8]           concept probabilities
    ↓  Linear head (polarity-initialised)     → ŷ  [2]           melanoma / nevus

    Lesion mask M ──┬── training : L_spa penalises out-of-lesion energy of present concepts
                    └── inference: a_k = coverage of concept k  →  per-case audit score
```

The mask comes from ground truth (PH²) or from SAM (Derm7pt, and PH² in the fully automatic setting), produced independently of the concept scores it constrains. Only the concept layer and the linear head are trained; the backbone stays frozen.

## Objective

```
L = L_cls + λ_c · L_concept + λ_s · L_spa
```

`L_cls` is a class-weighted cross-entropy on the diagnosis, `L_concept` a per-concept BCE with per-concept positive weights, and `L_spa` the asymmetric coverage term. For each ground-truth-present concept k:

```
L_spa = mean_k ( 1 − Σ ReLU(A_k) ⊙ M / Σ ReLU(A_k) )
```

The penalty is one-sided. A concept is never forced to cover the lesion, only to stop reading evidence from outside it. Defaults: λ_c = 1, λ_s = 0.5.

## Clinical Concept Vocabulary

Eight dermoscopic concepts from the 7-Point Checklist and the ABCD rule, organised as semantic opposites so each malignancy indicator is paired with the benign pattern it must be told apart from. The linear head is initialised from these polarities, so each concept votes for its associated class before any training.

| k | Concept | Polarity |
|---|---|---|
| 1 | Typical Pigment Network | Benign |
| 2 | Atypical Pigment Network | Malignant |
| 3 | Blue-Whitish Veil | Malignant |
| 4 | Irregular Streaks | Malignant |
| 5 | Regular Streaks | Benign |
| 6 | Regular Dots & Globules | Benign |
| 7 | Irregular Dots & Globules | Malignant |
| 8 | Regression Structures | Malignant |

## Results

All figures are means over 15 runs (3 seeds × 5 folds) with Wilcoxon signed-rank tests on matched folds, unless marked single-seed.

### Main Results

| Dataset | Configuration | BAcc | Sens | Spec | F1 | cF1 | Spatial coh. |
|---|---|---|---|---|---|---|---|
| PH² | Attention + GT masks | **88.0** | 87.1 | 89.0 | 75.0 | 47.2 | 95.9 |
| PH² | Attention + SAM masks | 85.8 | 80.3 | 91.3 | 73.6 | 40.3 | 84.1 |
| PH² | GAP + GT masks | 82.4 | 76.2 | 88.5 | 68.3 | 34.9 | 89.3 |
| Derm7pt | Attention + SAM masks | 76.8 | 79.2 | — | 61.9 | 54.5 | — |

Attention pooling, not the spatial term, is where the accuracy is won: +5.6 points on PH² (p = 0.017) and +3.0 on Derm7pt (p = 0.035) over global-average pooling.

### Comparison with Prior Work

| Method | Training | PH² | Derm7pt |
|---|---|---|---|
| Two-Step (Patrício et al.) | Training-free | 85.05 | **79.1** |
| SC-CBM, diagnosis loss only | Trained CBM | 84.0 | 73.3 |
| SC-CBM, attention, GT masks | Trained CBM | **88.0** | 76.8 |
| SC-CBM, attention, SAM masks | Trained CBM | 85.8 | 76.8 |

We report the Derm7pt gap openly. The intervention analysis locates it in concept extraction rather than in the reasoning above it: supplying ground-truth concepts lifts Derm7pt from 76.8% to 83.3%, a +6.5-point ceiling the head is not responsible for.

### Training-Signal Ladder (PH², GAP, GT masks)

| Configuration | BAcc | cF1 | Spatial coh. |
|---|---|---|---|
| Diagnosis loss only | 84.0 | 43.2 | 61.4 |
| + concept supervision | 82.0 | **43.9** | 50.9 |
| + spatial coherence (full) | 82.4 | 34.9 | **89.3** |

Concept supervision alone lowers coherence to 50.9% as channels spread energy to compete on the diagnosis. The spatial term lifts it to 89.3% with balanced accuracy statistically unchanged (p = 0.68 vs diagnosis-only). The cF1 drop is a thresholding artefact that calibration recovers.

### Per-Case Audit

| Model | AUROC | Mean coverage (correct) | Mean coverage (wrong) |
|---|---|---|---|
| SC-CBM (spatial term) | **0.85** | 0.92 | 0.46 |
| Concept supervision only | 0.67 | — | — |

At the validation-fitted threshold the flag keeps specificity 0.85 at sensitivity 0.69, so roughly seven concept errors in ten are caught while nine correct assertions in ten are spared. Assertion correctness here is scored against the same concept annotations the loss supervises, so this is the coupled measurement; an independent wrongness criterion is the next step.

### Calibration

| Metric | Uncalibrated | Calibrated |
|---|---|---|
| Per-concept F1 (PH²) | 41.6 | **54.0** |
| Expected calibration error (PH²) | 0.127 | **0.087** |

### Mask Fidelity

| Dataset | Configuration | BAcc | cF1 | Spatial coh. |
|---|---|---|---|---|
| PH² | GAP, GT masks | 82.4 | 34.9 | 89.3 |
| PH² | GAP, SAM masks | 81.8 | 28.0 | 76.5 |
| Derm7pt | GAP, no masks | 73.8 | 52.7 | — |
| Derm7pt | GAP, SAM masks | 72.7 | 47.7 | 75.2 |

SAM masks reach IoU 0.67 mean and 0.84 median on PH², and preserve the diagnosis-level effect (p = 0.74 against ground truth). The typical case is segmented well; a tail of hard cases drags the mean. On Derm7pt, 38 of 1011 SAM masks are near-full-field artefacts on non-dermoscopic clinical photographs, and activating the spatial term there costs concept F1 significantly (p = 0.0006). Diagnosis tolerates imperfect masks; thresholded concept detection does not.

### Cross-Dataset Transfer

| Direction | GT masks | SAM masks |
|---|---|---|
| Derm7pt → PH² | 85.0 | 83.4 |
| PH² → Derm7pt | 68.0 | 68.4 |

Single run per direction, train on full source and test on full target. Indicative rather than benchmark figures.

### Spatial-Coherence Weight (PH², single-seed)

| λ_s | BAcc | cF1 | Spatial coh. |
|---|---|---|---|
| 0 | 85.6 | **48.7** | 52.5 |
| 0.1 | 85.5 | 46.7 | 70.0 |
| 0.25 | 80.3 | 39.8 | 79.4 |
| 0.5 | **87.9** | 36.9 | 89.5 |
| 0.75 | 83.3 | 28.2 | 90.4 |
| 1.0 | 83.9 | 25.1 | **93.6** |

The coherence-versus-detection trade is smooth and controllable in λ_s.

### Backbone Choice (single-seed)

| Dataset | Backbone | BAcc | cF1 | Spatial coh. |
|---|---|---|---|---|
| PH² | ResNet-18 | 73.6 | 5.9 | 78.5 |
| PH² | ResNet-50 | **87.9** | 36.9 | 89.5 |
| PH² | ResNet-101 | 84.0 | **43.9** | 82.3 |
| PH² | DenseNet-201 | 73.0 | 3.5 | 70.2 |
| Derm7pt | ResNet-18 | 50.5 | 28.9 | — |
| Derm7pt | ResNet-50 | **74.7** | **53.1** | — |
| Derm7pt | ResNet-101 | 71.9 | 49.7 | — |
| Derm7pt | DenseNet-201 | 70.4 | 45.4 | — |

Under a frozen encoder the bottleneck's ceiling is set by mid-level feature quality, which is why ResNet-50 survives while ResNet-18 and DenseNet-201 collapse the concept layer.

### Computational Cost

| Component | Value |
|---|---|
| Trainable parameters | 8,218 |
| Frozen parameters (backbone) | 8,543,296 |
| Trainable share of the network | < 0.1% |
| Inference time | 4.5 ms / image |
| Throughput | 223 images / s |
| Hardware | Apple M3 Pro, PyTorch MPS backend |

The concept layer is a 1×1 convolution from 1024 feature channels to 8 concept channels and the head is an 8×2 linear map. Attention pooling adds no parameters, since its weights are a softmax over each channel's own activations. The audit score is free at test time: it reuses activation maps the forward pass has already produced, and its only extra input is the mask the segmenter supplies.

### Negative Result

Background swaps and synthetic out-of-lesion distractors (hair, ruler marks) degrade every configuration by between 0.03 and 0.10 balanced accuracy, spatially regularised or not. Under a frozen encoder no concept-level regulariser can remove background signal from the features the concept channels read. Training-time coherence buys faithful evidence, not corruption immunity. This is what motivates the audit role rather than undermining the method.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- Segment Anything (ViT-B) for automatic masks
- NumPy, SciPy, scikit-learn, Matplotlib, Pillow

```
pip install torch torchvision numpy scipy scikit-learn matplotlib pillow segment-anything
```

The experiments in the paper were run on an Apple M3 Pro with the PyTorch MPS backend. A CUDA GPU works unchanged; the frozen backbone keeps memory requirements modest.

## Datasets

| Dataset | Images | Masks | Concepts | Folds |
|---|---|---|---|---|
| PH² | 200 | Ground truth | 8, expert-annotated | 5 (stratified) |
| Derm7pt | 1,011 | None (SAM-generated) | 7-point criteria | 5 (stratified) |

Concept order throughout the code is `[TPN, APN, BWV, ISTR, RSTR, RDG, IDG, RS]`.

Dataset loaders and evaluation metrics are shared with the companion [GroundDerm](https://github.com/razanharith/GroundDerm) codebase. Point the two environment variables at your local copies before running:

```
export GROUNDERM_ROOT=/path/to/GroundDerm/main-code
export DATASETS_ROOT=/path/to/datasets      # expects PH2/ and derm7pt/ inside
```

## Training

```
python SC-CBM/train.py --dataset ph2 --epochs 30 --tag sccbm --save-ckpt
```

Full multi-seed protocol (3 seeds × 5 folds per configuration):

```
bash SC-CBM/run_multiseed.sh
```

Ablation ladder, λ_s sweep, and backbone sweep:

```
bash SC-CBM/run_all.sh
bash SC-CBM/sweep_lambda.sh
```

## Evaluation

```
python SC-CBM/calibrate.py          # per-concept thresholds + diagnosis temperature
python SC-CBM/verify_case.py        # per-case audit score, AUROC, operating point
python SC-CBM/deconfound.py         # background-corruption stress test
python SC-CBM/cross_dataset.py      # transfer between PH2 and Derm7pt
python SC-CBM/make_tables.py        # result JSONs → LaTeX / markdown tables
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alharith2026sccbm,
  title   = {{SC-CBM}: A Self-Auditing Concept Bottleneck Model for Interpretable
             Skin Lesion Diagnosis},
  author  = {Alharith, Razan},
  year    = {2026},
  url     = {https://github.com/razanharith/SC-CBM}
}
```

## Contact

For questions about this research, contact Razan Alharith at razanalharith@my.swjtu.edu.cn.
