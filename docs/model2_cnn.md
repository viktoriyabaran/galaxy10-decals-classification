# Model 2: Small CNN

A convolutional neural network trained from scratch: it learns its own image filters, looks at
neighbouring pixels together and recognises the same pattern anywhere in the image.
Code: `src/model2_cnn.py`, network in `src/small_cnn.py`, shared PyTorch code in `src/training.py`
(data from `src/dataset.py`, metrics from `src/evaluate.py`, the same as model 1).

## Data

- The same cleaned data and split as model 1: train 12,289 / validation 2,633 / test 2,634.
- The same 64x64 images (`load_images(64)`, 4x4 averaging, no cropping).
- Converted to PyTorch tensors of shape (colour, height, width) = (3, 64, 64), scaled to 0-1.
- **Normalised** per colour channel to mean 0 and std 1, using the mean and std of the training images only
  (R, G, B mean 0.167 / 0.162 / 0.159, std 0.114 / 0.108 / 0.100). Neural networks learn much better
  with inputs centred around 0; validation and test are normalised with the training numbers.
- Grouped into **batches** of 64 images: 193 training batches per epoch.
- Training batches are shuffled every epoch (seeded). This matters because the file is sorted
  by class: without shuffling, the first batches would contain only one class.

## Augmentation

- Every training image gets a random orientation each time it is used: rotated by 0 / 90 / 180 / 270
  degrees and mirrored or not, 8 possible versions (`augment()` in `src/training.py`).
- Galaxies have no "up" in space, so every version shows the same galaxy type.
- Only the pixel positions change, never the brightness values.
- Rules: training images only (validation and test show galaxies as they are), created on the fly
  (no copies on disk), and after the split (otherwise versions of one galaxy could end up in train and test).
- Only 90-degree steps: exact and lossless, other angles would need interpolation and leave empty corners.
- Example: `results/model2_cnn/augmentation.png`.

## Network

Four convolution blocks, each: 3x3 convolution -> batch normalisation -> ReLU -> 2x2 max pooling.

| Part | Output shape | What it learns |
|---|---|---|
| input | 3 x 64 x 64 | |
| block 1 | 32 x 32 x 32 | simple patterns: edges, bright spots |
| block 2 | 64 x 16 x 16 | combinations: blobs, curves |
| block 3 | 128 x 8 x 8 | larger shapes |
| block 4 | 256 x 4 x 4 | whole structures: arms, bars, disks |
| average pooling + flatten | 256 | one number per feature map |
| dropout (30%, training only) | 256 | less memorising |
| linear | 10 | one score per class |

- **391,946 learnable parameters**: 389,376 in the convolution blocks, 2,570 in the final layer.
- A convolution filter is a small grid of numbers that slides over the image and lights up where its
  pattern appears. The filters start random and are learned during training.
- Average pooling at the end makes the network work for any image size (used in the 128x128 experiment).
- Untrained, the network gives every class ~10% probability.

## Training

| Setting | Value | Meaning |
|---|---|---|
| loss | cross-entropy with class weights | -log(probability given to the true class) |
| class weights | from 0.67 (Round Smooth) to 5.27 (Cigar Shaped) | same formula as `class_weight="balanced"` in model 1 |
| optimizer | Adam | adjusts every parameter a little after each batch |
| learning rate | 0.001, cosine schedule to ~0 | size of the adjustment steps |
| batch size | 64 | images per learning step |
| epochs | up to 50 | passes over all training images |
| early stopping | patience 10 | stop if validation macro F1 has not improved for 10 epochs |
| best epoch | highest validation macro F1 | saved to `models/model2_cnn.pt` |
| seed | 42 | same start values and shuffling |

- One learning step: guess for a batch -> loss -> backpropagation (how each parameter should change)
  -> optimizer step. The untrained loss is ~2.3 = -log(0.1), the loss of guessing 10% for every class.
- Runs on the Mac's GPU (`mps`). Some GPU operations are not perfectly repeatable, so a new training run
  can give slightly different numbers (typically within ~1 point).

### Learning-rate schedule

| Run | Learning rate | Epochs | Validation accuracy | Validation macro F1 |
|---|---|---|---|---|
| first run | fixed 0.001 | 30 | 74.2% (epoch 30) | 0.713 |
| final run | 0.001, cosine to ~0 | 50 | **79.8%** (epoch 50) | **0.778** |

(The first run's numbers come from its console output; its history.csv was replaced by the final run.)

- With a fixed learning rate, validation scores jumped a lot between epochs (e.g. 26% at epoch 3,
  58.7% at epoch 15 right after 72.8%): the steps were too big to settle on a good solution.
- With the cosine schedule, the first half is still jumpy, but from epoch ~25 the curves become smooth:
  small steps at the end let the network settle.
- The best epoch was the last one (50). Early stopping did not trigger.
- No overfitting: training loss (0.50) and validation loss (0.59) fell together until the end.

Learning curves: `results/model2_cnn/learning_curves.png`, every epoch in `results/model2_cnn/history.csv`.

## Results (validation set)

- Validation accuracy: **79.8%**, macro F1: **0.778** (model 1: 58.6% / 0.541).

| Class | Precision | Recall | F1 | Model 1 recall | Model 1 F1 |
|---|---|---|---|---|---|
| Disturbed | 0.461 | 0.475 | 0.468 | 0.204 | 0.288 |
| Merging | 0.809 | 0.848 | 0.828 | 0.511 | 0.567 |
| Round Smooth | 0.926 | 0.922 | 0.924 | 0.858 | 0.782 |
| In-between Round Smooth | 0.914 | 0.881 | 0.897 | 0.660 | 0.646 |
| Cigar Shaped Smooth | 0.614 | 0.860 | 0.717 | 0.200 | 0.312 |
| Barred Spiral | 0.816 | 0.810 | 0.813 | 0.363 | 0.410 |
| Unbarred Tight Spiral | 0.669 | 0.784 | 0.722 | 0.546 | 0.486 |
| Unbarred Loose Spiral | 0.715 | 0.542 | 0.617 | 0.442 | 0.438 |
| Edge-on without Bulge | 0.847 | 0.939 | 0.890 | 0.858 | 0.768 |
| Edge-on with Bulge | 0.907 | 0.910 | 0.909 | 0.760 | 0.710 |

Confusion matrix: `results/model2_cnn/confusion_val.png`.

### What the CNN fixed

The CNN is better than the forest in every class; the forest's weakest classes improved the most.

- **Cigar Shaped Smooth: 20% -> 86% recall.** The elongated shape is recognised at any angle
  (filters + rotation augmentation), and the class weight 5.27 helps.
- **Barred Spiral: 36% -> 81%.** The filters can detect the bright bar through the centre.
- **Merging: 51% -> 85%.** Two bright cores close together are a local pattern single pixels cannot express.
- **Edge-on with / without Bulge** are confused ~3% each way instead of ~10%: the bulge is now visible.

### What is still hard

- **Unbarred Loose Spiral: 54%**, 18% called Tight Spiral and 11% Disturbed. Loose vs tight arms is a matter
  of degree, and at 64x64 the arms of small galaxies are only a few pixels wide.
- **Disturbed: 48%**, mostly confused with Loose Spiral (14%) and In-between Round Smooth (10%).
  "Irregular" is not one shape.
- **Cigar Shaped precision 0.61**: the large class weight makes the network say "Cigar" when unsure,
  so it finds most cigars but also calls some other galaxies cigars.

## Experiment (step 8, validation set)

`src/model2_experiments.py`: each variant changes one thing compared with the baseline (the final run above,
whose numbers are read from its history.csv instead of training it again).
Results in `results/model2_cnn/experiments.csv`, learning curves in `results/model2_cnn/experiments/`.

| Variant | Image size | Augmentation | Epochs trained | Best epoch | Accuracy | Macro F1 | Training time |
|---|---|---|---|---|---|---|---|
| baseline | 64 | yes | 50 | 50 | 79.8% | 0.778 | 354 s |
| no augmentation | 64 | no | 26 (early stop) | 16 | 69.7% | 0.652 | 166 s |
| 128x128 images | 128 | yes | 50 | 44 | 78.1% | 0.759 | 1,382 s |

### No augmentation: overfitting

- Training loss fell to 0.23 (baseline: 0.50), while validation loss stayed above ~1.0 and jumped up to 2.26
  (baseline: 0.59). The growing gap between the two curves is the sign of memorising: without rotations,
  the network learns each galaxy in one orientation only.
- Early stopping ended training at epoch 26 (best epoch 16).
- Augmentation is worth **~10 points of accuracy** (+0.13 macro F1), the biggest single effect so far,
  and it costs nothing at test time.
- Caveat: early stopping ended the run while the learning rate was still ~0.0005, and the last epochs were
  rising again (67.8% at epoch 26). With all 50 epochs the gap might be smaller, but the overfitting is clear.

### 128x128 images: no gain

- 78.1% / 0.759: no improvement (the difference to the baseline is within the noise).
- 3.9x slower: 26.5 s per epoch instead of 7.1 s.
- The curves are smooth and do not overfit, but training loss ended higher (0.63 vs 0.50): the network learned
  more slowly at this size.
- Likely reason: the network was designed for 64x64. After 4 blocks, the last feature maps are 8x8 instead of
  4x4, so each position sees only half as much of the galaxy: more detail, less overview. Using 128x128 would
  probably need a fifth block (not tested).
- Neither model gains from 128x128, for different reasons: the forest cannot use detail at all,
  this CNN is not built for that image size.

**Chosen for the test: the baseline** (64x64 with augmentation): best on validation, chosen before the
experiments, and 4x faster to train than 128x128.

## Final result

| | Validation | **Test** |
|---|---|---|
| Accuracy | 79.8% | **79.5%** |
| Macro F1 | 0.778 | **0.779** |

| Class | Precision | Recall | F1 | Model 1 recall | Model 1 F1 |
|---|---|---|---|---|---|
| Disturbed | 0.404 | 0.444 | 0.424 | 0.198 | 0.277 |
| Merging | 0.833 | 0.833 | 0.833 | 0.561 | 0.623 |
| Round Smooth | 0.918 | 0.907 | 0.912 | 0.818 | 0.741 |
| In-between Round Smooth | 0.891 | 0.888 | 0.889 | 0.696 | 0.631 |
| Cigar Shaped Smooth | 0.644 | 0.940 | 0.764 | 0.120 | 0.194 |
| Barred Spiral | 0.785 | 0.810 | 0.797 | 0.350 | 0.408 |
| Unbarred Tight Spiral | 0.693 | 0.817 | 0.750 | 0.612 | 0.535 |
| Unbarred Loose Spiral | 0.717 | 0.522 | 0.604 | 0.411 | 0.422 |
| Edge-on without Bulge | 0.890 | 0.886 | 0.888 | 0.853 | 0.758 |
| Edge-on with Bulge | 0.902 | 0.950 | 0.925 | 0.700 | 0.682 |

Confusion matrix: `results/model2_cnn/confusion_test.png`.

- Test and validation agree within the noise (-0.3 points accuracy, same macro F1): the choice of the best
  epoch and of the setting on validation did not make the result optimistic.
- The same pattern as on validation. Cigar Shaped Smooth reaches 94% recall (model 1: 12%).
- The main remaining confusion goes both ways between **Disturbed and Unbarred Loose Spiral**:
  22% of Disturbed are called Loose Spiral, 14% of Loose Spirals are called Disturbed;
  16% of Loose Spirals are called Tight Spiral.

**Model 2 result for the comparison: test accuracy 79.5%, macro F1 0.779.**

### Time (Apple M4, GPU `mps`)

| Part | Images | Time | Per image |
|---|---|---|---|
| Training (`fit`, 50 epochs) | 12,289 | 354.43 s | 28.841 ms |
| Training, one epoch (average) | 12,289 | 7.09 s | 0.577 ms |
| Validation (`predict`) | 2,633 | 0.46 s | 0.176 ms |
| Test (`predict`) | 2,634 | 0.49 s | 0.184 ms |

Source: `results/model2_cnn/timings.csv`, written by `src/model2_cnn.py`. Training time is the sum of the
epoch times in `history.csv` (from the training run); prediction times are measured after a GPU warm-up,
only the model's own work is timed.

## Comparison with model 1 (test set)

| | Model 1: Random Forest | **Model 2: Small CNN** |
|---|---|---|
| Accuracy | 58.1% | **79.5%** (+21.4 points) |
| Macro F1 | 0.527 | **0.779** (+0.252) |
| Training | 15.9 s | 354 s (~22x longer) |
| Predicting all test images | 0.07 s | 0.49 s (~7x longer) |
| Per test image | 0.028 ms | 0.184 ms (still ~5,400 galaxies per second) |

## Comparison with the literature

- You (2025, *Evaluation of Galaxy Morphology Classification with ML and DL*): CNN ~64-68% on 64x64 images,
  original dataset, random split, no augmentation reported. Model 2 is ~11-15 points higher at the same image
  size, with augmentation, class weights and a learning-rate schedule, on a stricter deduplicated split.
- Astroformer (Dagli 2023): 94.86% top-1 accuracy, with a much larger transformer-based model, larger images
  and heavy augmentation. Model 3 (a pretrained network) is the step towards closing that gap.
