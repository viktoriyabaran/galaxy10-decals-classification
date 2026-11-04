# Model 1: Random Forest

The simple baseline: a classic machine learning model that sees each image as a flat list of pixel values.
Code: `src/model1_random_forest.py` (shared code in `src/dataset.py` and `src/evaluate.py`).

## Data

- Source: `Galaxy10_DECals_NoDuplicated.h5`, cleaned by `src/prepare_data.py`
  (1 blank image and 118 duplicates of the same object within 30" removed, 17,556 images left).
- Split, stratified by class, seed 42: train 12,289 / validation 2,633 / test 2,634.

## Preprocessing

1. **Shrink 256x256 to 64x64** by averaging each 4x4 square of pixels (`load_images(64)`).
   - No cropping: we measured that a centre crop of 128x128 would cut off part of the galaxy in 15% of
     images, and in 61% of "Edge-on without Bulge" and 31% of "Merging" galaxies, exactly the parts
     that define those classes.
   - At 64x64, large galaxies keep visible features (bars, arms, thin disks); small galaxies become a few pixels.
2. **Flatten** each 64x64x3 image into one row of 12,288 numbers and **scale** to 0-1 (divide by 255).
   - Scaling does not matter for a Random Forest (trees only compare values with thresholds),
     it is done so that all three models get identical input.

Weakness of this representation: the forest has no idea which pixels are neighbours, and the same shape
at a different position or angle looks like completely different numbers.

## Model settings

| Setting | Value | Meaning |
|---|---|---|
| `n_estimators` | 300 | number of trees that vote |
| `max_features` | `"sqrt"` | each question picks from sqrt(12,288) ~ 110 random pixels |
| `min_samples_leaf` | 1 | trees may keep splitting until every training image sits alone |
| `class_weight` | `"balanced"` | mistakes on rare classes count more |
| `random_state` | 42 | same forest every run |

Training takes ~16 s on an Apple M4 (10 cores). Trees are 24-49 questions deep with ~2,372 leaves each.

## Results (validation set)

- Training accuracy: **100%**: the trees memorise the training images.
- Validation accuracy: **58.6%**, macro F1: **0.541** (random guessing: ~10-15%).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Disturbed | 0.493 | 0.204 | 0.288 |
| Merging | 0.637 | 0.511 | 0.567 |
| Round Smooth | 0.718 | 0.858 | 0.782 |
| In-between Round Smooth | 0.633 | 0.660 | 0.646 |
| Cigar Shaped Smooth | 0.714 | 0.200 | 0.312 |
| Barred Spiral | 0.472 | 0.363 | 0.410 |
| Unbarred Tight Spiral | 0.438 | 0.546 | 0.486 |
| Unbarred Loose Spiral | 0.433 | 0.442 | 0.438 |
| Edge-on without Bulge | 0.695 | 0.858 | 0.768 |
| Edge-on with Bulge | 0.667 | 0.760 | 0.710 |

Confusion matrix: `results/model1_rf/confusion_val.png`.

### What the confusion matrix shows

- **Easy:** Round Smooth (86%) and Edge-on without Bulge (86%), classes with a consistent look.
- **The three spirals are confused with each other:** only 36% of Barred Spirals are recognised
  (23% called Tight, 24% Loose). Bars and arm tightness are fine details.
- **Disturbed: 20%**, 31% are called Unbarred Loose Spiral. Irregular shapes have no fixed pixel pattern.
- **Cigar Shaped Smooth: 20%**, 38% called In-between Round Smooth, 24% Edge-on with Bulge.
  Few training examples (233), and the tilt angle varies.
- **Edge-on with / without Bulge** swap ~10% each way: the bulge is a small detail.

## Comparison with the literature

You (2025, *Evaluation of Galaxy Morphology Classification with ML and DL*) report Random Forest 38.2%
and SVM 43.7% on 64x64 images compressed with PCA-256, on the original dataset with a random split.

### Does PCA explain the difference?

Same forest and split, with PCA in front (validation set):

| Input | Variance kept | Accuracy | Macro F1 |
|---|---|---|---|
| raw pixels (12,288 columns) | 100% | 58.6% | 0.541 |
| PCA-256 | 80% | 53.1% | 0.478 |
| PCA-50 | 65% | 57.4% | 0.532 |

- PCA-256 lowers accuracy by 5.5 points, so PCA explains part of the gap, but not all of it.
- PCA-50 keeps less information but scores higher than PCA-256. Likely reason: the later components are
  mostly noise, and with `max_features="sqrt"` a tree picks from only 16 random columns out of 256,
  most of them noise. With 50 components, most candidates are useful.
- Compression is not good or bad by itself: it has to be measured on the validation set.

## Experiment (step 6, validation set)

`src/model1_experiments.py`: each variant changes one thing compared with the baseline.
Results also in `results/model1_rf/experiments.csv`.

| Variant | Image size | PCA | `min_samples_leaf` | Columns | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| baseline | 64 | none | 1 | 12,288 | 58.6% | 0.541 |
| tiny images | 16 | none | 1 | 768 | 56.0% | 0.516 |
| smaller images | 32 | none | 1 | 3,072 | 58.9% | 0.537 |
| bigger images | 128 | none | 1 | 49,152 | 59.0% | 0.541 |
| PCA-50 | 64 | 50 | 1 | 50 | 57.4% | 0.532 |
| PCA-256 | 64 | 256 | 1 | 256 | 53.1% | 0.478 |
| less memorising | 64 | none | 5 | 12,288 | 58.0% | 0.546 |

- With 2,633 validation images, accuracy has a natural noise of about +/-1 point, so differences
  below ~1-2 points are not real improvements.
- Image size: accuracy levels off from 32x32 upwards (58.9% / 58.6% / 59.0% for 32 / 64 / 128).
  Only 16x16 is slightly worse (56.0%), where small galaxies shrink to 1-3 pixels.
  The forest cannot use finer detail, so 128x128 only makes training 5x slower (77 s vs 16 s).
- Leaf size (`min_samples_leaf` 1 vs 5) makes no measurable difference.
- PCA-256 is the only clear effect: -5.5 points accuracy.
- PCA is fitted on the training images only, then applied to validation (otherwise information leaks).
- **Chosen for the test: the baseline.** It was chosen before the experiment, and picking the variant
  that is +0.3 higher by chance would make the result optimistic. It also matches the 64x64 input of the CNNs.
## Final result (step 7, test set, run once)

| | Validation | **Test** |
|---|---|---|
| Accuracy | 58.6% | **58.1%** |
| Macro F1 | 0.541 | **0.527** |

| Class | Precision | Recall | F1 | Validation F1 |
|---|---|---|---|---|
| Disturbed | 0.464 | 0.198 | 0.277 | 0.288 |
| Merging | 0.701 | 0.561 | 0.623 | 0.567 |
| Round Smooth | 0.678 | 0.818 | 0.741 | 0.782 |
| In-between Round Smooth | 0.577 | 0.696 | 0.631 | 0.646 |
| Cigar Shaped Smooth | 0.500 | 0.120 | 0.194 | 0.312 |
| Barred Spiral | 0.491 | 0.350 | 0.408 | 0.410 |
| Unbarred Tight Spiral | 0.476 | 0.612 | 0.535 | 0.486 |
| Unbarred Loose Spiral | 0.432 | 0.411 | 0.422 | 0.438 |
| Edge-on without Bulge | 0.682 | 0.853 | 0.758 | 0.768 |
| Edge-on with Bulge | 0.664 | 0.700 | 0.682 | 0.710 |

Confusion matrix: `results/model1_rf/confusion_test.png`.

- Test and validation agree within the noise (-0.5 points), as expected from a split without leakage.
- The same pattern as on validation: Round Smooth and Edge-on without Bulge are easy, the three spirals
  are confused with each other, Disturbed is mostly called Unbarred Loose Spiral (30%).
- Cigar Shaped Smooth is the weakest class: only 6 of 50 found (12%). 38% are called Edge-on with Bulge
  and 26% In-between Round Smooth. With only 50 test images, its per-class numbers jump a lot between
  validation and test (10 vs 6 found), so they are less reliable than the others.

**Model 1 result for the comparison: test accuracy 58.1%, macro F1 0.527.**

### Time (Apple M4, 10 cores, 32 GB)

| Part | Images | Time | Per image |
|---|---|---|---|
| Training (`fit`) | 12,289 | 15.86 s | 1.290 ms |
| Validation (`predict`) | 2,633 | 0.07 s | 0.027 ms |
| Test (`predict`) | 2,634 | 0.07 s | 0.028 ms |

Source: `results/model1_rf/timings.csv`, written by `src/model1_random_forest.py`.

- Per image, training is ~47x slower than predicting (1.290 vs 0.028 ms).
- Predicting one galaxy takes ~0.03 ms, about 36,000 galaxies per second.
