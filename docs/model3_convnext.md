# Model 3: ConvNeXt-Tiny (pretrained, fine-tuned)

A large modern convolutional network that was already trained on ImageNet (1.2 million everyday photos).
It does not learn from zero: it already knows edges, blobs, curves and textures, and only adapts them to
galaxies ("transfer learning" / "fine-tuning").
Code: `src/model3_convnext.py`, run table built by `src/model3_summary.py`, shared code in `src/training.py`
(data from `src/dataset.py`, metrics from `src/evaluate.py`, the same as models 1 and 2).

## Why a pretrained network

- Training a large network from scratch was measured to be too slow on this Mac: e.g. WaveMix
  (the ~95% result in the literature) at 256x256 would take ~55 min per epoch, ~4 days for 100 epochs.
- Fine-tuning a pretrained network needs ~15 epochs instead of 50-100+, so a run takes under an hour at 128x128.
- ConvNeXt (2022) instead of ResNet: a newer design that borrows ideas from vision transformers
  (large 7x7 filters, LayerNorm, GELU), available in `torchvision` with ImageNet weights.

## Data

- The same cleaned data and split as models 1 and 2: train 12,289 / validation 2,633 / test 2,634.
- **128x128 images** (`load_images(128)`, 2x2 averaging, no cropping). A 256x256 run is in the experiments.
- Normalised with the **ImageNet** mean and std (0.485 / 0.456 / 0.406, 0.229 / 0.224 / 0.225), not the
  galaxy statistics: the pretrained filters expect inputs on the same scale as during pretraining.
- Images stay in memory as uint8 (0-255) and are converted to floats one batch at a time
  (`GalaxyImages`). Converting all of them up front, as in model 2, would need ~14 GB at 256x256.
- The same augmentation as model 2: random 90-degree rotation + mirroring, training images only.
- Batches of 32 (model 2: 64, this network needs much more memory per image): 385 training batches per epoch.

## Network

- `torchvision.models.convnext_tiny` with ImageNet weights (`IMAGENET1K_V1`, ~110 MB download, cached).
- The last layer (768 -> 1000 ImageNet classes) is replaced by a new one (768 -> 10 galaxy classes),
  which starts random. Everything else keeps its pretrained values and is fine-tuned.
- **27,827,818 learnable parameters**, ~71x more than model 2 (391,946).

## Training

| Setting | Value | Meaning |
|---|---|---|
| loss | cross-entropy with class weights | the same as model 2 |
| optimizer | AdamW, weight decay 0.05 | Adam that also gently pulls weights towards 0; standard for fine-tuning |
| learning rate | 0.0001, cosine schedule to ~0 | 10x smaller than model 2, so the pretrained knowledge is not destroyed |
| batch size | 32 | images per learning step |
| epochs | 15 | a pretrained network needs far fewer passes |
| early stopping | none | see the experiment below |
| best epoch | highest validation macro F1 | saved to `models/model3_convnext_size_128.pt` |
| seed | 42 | same start values of the new layer and same shuffling |

- **Fast start:** 75.6% after one epoch, and 81.5% after two, already above model 2's final 79.8%.
- Training is repeatable: the two runs below are identical, number for number, for the 9 epochs they share.
- Unlike model 2, the network does fit the training images better than new ones: at the end training loss is
  0.12 and validation loss 0.43. Validation loss is still at its lowest level (0.42-0.43 from epoch 12),
  so this does not hurt yet, but more epochs would probably start to overfit.

Learning curves: `results/model3_convnext/size_128/learning_curves.png`, every epoch in
`results/model3_convnext/size_128/history.csv`.

## Results (validation set)

- Validation accuracy: **87.2%**, macro F1: **0.860** (model 2: 79.8% / 0.778, model 1: 58.6% / 0.541).

| Class | Precision | Recall | F1 | Model 2 recall | Model 2 F1 |
|---|---|---|---|---|---|
| Disturbed | 0.632 | 0.531 | 0.577 | 0.475 | 0.468 |
| Merging | 0.896 | 0.917 | 0.906 | 0.848 | 0.828 |
| Round Smooth | 0.959 | 0.944 | 0.952 | 0.922 | 0.924 |
| In-between Round Smooth | 0.925 | 0.931 | 0.928 | 0.881 | 0.897 |
| Cigar Shaped Smooth | 0.814 | 0.960 | 0.881 | 0.860 | 0.717 |
| Barred Spiral | 0.905 | 0.869 | 0.887 | 0.810 | 0.813 |
| Unbarred Tight Spiral | 0.763 | 0.861 | 0.809 | 0.784 | 0.722 |
| Unbarred Loose Spiral | 0.805 | 0.756 | 0.780 | 0.542 | 0.617 |
| Edge-on without Bulge | 0.913 | 0.943 | 0.928 | 0.939 | 0.890 |
| Edge-on with Bulge | 0.938 | 0.968 | 0.952 | 0.910 | 0.909 |

Confusion matrix: `results/model3_convnext/size_128/confusion_val.png`.

### What improved over model 2

F1 is higher in every class.

- **Unbarred Loose Spiral: 54% -> 76% recall**, the largest gain. Loose vs tight arms is a fine detail that
  the 128x128 input and the pretrained texture filters can see.
- **Cigar Shaped Smooth: precision 0.61 -> 0.81** at the same high recall (96%): the network no longer
  calls uncertain galaxies "Cigar" as often.
- **Barred Spiral: 81% -> 87%**, **Merging: 85% -> 92%**.

### What is still hard

- **Disturbed: 53%**, still clearly the weakest class: 15% are called Unbarred Loose Spiral and
  10% In-between Round Smooth. "Irregular" is not one shape, and the volunteer labels for this class are the
  least consistent, so part of this is likely label noise rather than a model problem.
- **The spirals among each other:** 11% of Loose Spirals are called Tight, 8% of Tight Spirals are called
  Loose, 6% of Loose Spirals are called Disturbed.

## Experiments (validation set)

Each run is one call of `src/model3_convnext.py`; `src/model3_summary.py` collects them into
`results/model3_convnext/experiments.csv`. Each run's files are in its own folder in `results/model3_convnext/`.

| Variant | Image size | Early stopping | Epochs trained | Best epoch | Accuracy | Macro F1 | Training time |
|---|---|---|---|---|---|---|---|
| early stopping | 128 | patience 4 | 9 (early stop) | 5 | 85.9% | 0.843 | 1,370 s |
| **baseline (full schedule)** | 128 | none | 15 | 15 | **87.2%** | **0.860** | 2,544 s |
| 256x256 images | 256 | none | 15 | 12 | 87.5% | 0.862 | 9,493 s |

### Early stopping: stopped too soon

- The first run used patience 4, as model 2 used patience 10. Epoch 5 jumped to 85.9% / 0.843, epochs 6-9
  were slightly below it in macro F1, and training stopped at epoch 9.
- At that point the learning rate was still ~45% of its starting value. Epoch 9 already had the lowest
  validation loss of that run (0.443), so the network was still improving: epoch 5 was a lucky high point
  (validation scores move by about +/-1 point between epochs).
- With the full 15-epoch schedule, the low-learning-rate epochs at the end gave the best results
  (epochs 12-15: 86.8-87.4%). Gain: **+1.3 points accuracy, +0.017 macro F1** for ~20 more minutes.
- Because both runs are identical up to epoch 9, the difference comes only from stopping, not from chance.
- Lesson: with a short cosine schedule, early stopping is not needed. The best epoch is saved anyway,
  and the last epochs are where the network settles.

### 256x256 images: no real gain

- 87.5% / 0.862 (best epoch 12) vs 87.2% / 0.860 at 128x128: +0.3 points, within the noise of +/-1 point.
- **3.7x slower:** ~620 s per epoch instead of ~170 s, 158 min for the run instead of 42 min.
- The learning curves have the same shape: a faster start (84.0% after 2 epochs vs 81.5%), then both
  end at the same level.
- Per class, the differences are small and go both ways: Disturbed F1 0.611 vs 0.577, Cigar Shaped
  0.855 vs 0.881 (only 50 and 162 validation images, so each misclassified galaxy moves these by 0.6-2 points).
- Likely reason: at 128x128 the features that decide most classes (bars, arm shape, bulge, cores) are already
  visible, and the remaining errors are the ambiguous classes (Disturbed vs Loose Spiral, Tight vs Loose),
  where the labels themselves disagree, not a lack of detail.

**Chosen for the test: the baseline** (128x128, full schedule): the same score as 256x256 within the noise,
3.7x faster to train and ~4x faster to predict.

## Final result

| | Validation | **Test** |
|---|---|---|
| Accuracy | 87.2% | **87.3%** |
| Macro F1 | 0.860 | **0.857** |

| Class | Precision | Recall | F1 | Model 2 recall | Model 2 F1 |
|---|---|---|---|---|---|
| Disturbed | 0.595 | 0.580 | 0.588 | 0.444 | 0.424 |
| Merging | 0.906 | 0.917 | 0.911 | 0.833 | 0.833 |
| Round Smooth | 0.964 | 0.939 | 0.951 | 0.907 | 0.912 |
| In-between Round Smooth | 0.938 | 0.950 | 0.944 | 0.888 | 0.889 |
| Cigar Shaped Smooth | 0.772 | 0.880 | 0.822 | 0.940 | 0.764 |
| Barred Spiral | 0.863 | 0.886 | 0.874 | 0.810 | 0.797 |
| Unbarred Tight Spiral | 0.806 | 0.868 | 0.836 | 0.817 | 0.750 |
| Unbarred Loose Spiral | 0.810 | 0.746 | 0.776 | 0.522 | 0.604 |
| Edge-on without Bulge | 0.933 | 0.919 | 0.926 | 0.886 | 0.888 |
| Edge-on with Bulge | 0.937 | 0.954 | 0.945 | 0.950 | 0.925 |

Confusion matrix: `results/model3_convnext/size_128/confusion_test.png`.
Run: `src/model3_convnext.py --size 128 --eval-only` (loads the saved network, no training).

- Test and validation agree within the noise (+0.1 points accuracy, -0.003 macro F1): choosing the run and the
  best epoch on validation did not make the result optimistic.
- F1 is higher than model 2 in every class. Cigar Shaped recall is lower (44 vs 47 of 50 found), but with
  far fewer false alarms (precision 0.77 vs 0.64).
- The remaining confusions are the same as on validation: 15% of Disturbed are called Unbarred Loose Spiral;
  25% of Loose Spirals are missed (9% called Tight Spiral, 8% Disturbed); 9% of Tight Spirals are called Loose.

**Model 3 result for the comparison: test accuracy 87.3%, macro F1 0.857.**

### Time (Apple M4 MacBook Air, GPU `mps`)

| Part | Images | Time | Per image |
|---|---|---|---|
| Training (`fit`, 15 epochs) | 12,289 | 2,543.61 s (42 min) | 206.983 ms |
| Training, one epoch (average) | 12,289 | 169.57 s | 13.799 ms |
| Validation (`predict`) | 2,633 | 6.24 s | 2.372 ms |
| Test (`predict`) | 2,634 | 6.38 s | 2.423 ms |

Source: `results/model3_convnext/size_128/timings.csv`, written by `src/model3_convnext.py --eval-only`.
Training time is the sum of the epoch times in `history.csv`; prediction times are measured after a GPU
warm-up and include converting each batch from uint8.

- The MacBook Air has no fan and slows down when hot: the first epoch took 139 s, the rest ~170 s.
- Predicting is still fast: ~410 galaxies per second.

## Comparison with models 1 and 2 (test set)

| | Model 1: Random Forest | Model 2: Small CNN | **Model 3: ConvNeXt-Tiny** |
|---|---|---|---|
| Accuracy | 58.1% | 79.5% | **87.3%** (+7.8 points vs model 2) |
| Macro F1 | 0.527 | 0.779 | **0.857** (+0.078) |
| Parameters | (300 trees) | 391,946 | 27,827,818 |
| Image size | 64x64 | 64x64 | 128x128 |
| Training | 15.9 s | 354 s | 2,544 s (~7x model 2) |
| Predicting all test images | 0.07 s | 0.49 s | 6.4 s (~13x model 2) |
| Per test image | 0.028 ms | 0.184 ms | 2.423 ms |

Tables for all three models: `results/comparison/overall.csv` and `per_class.csv` (`src/compare_models.py`).

## Comparison with the literature

- Astroformer (Dagli 2023): 94.86%, WaveMix (Jeevan et al.): ~95%. Both train large networks from scratch at
  full 256x256 resolution with heavy augmentation and far more compute (measured here: WaveMix at 256x256 would
  take ~4 days for 100 epochs on this Mac). Their splits are not the same as ours: here duplicates of the same
  object are removed first, so the same galaxy cannot appear in both training and test, so the numbers are
  not directly comparable.
- Model 3 closes about half of the gap between model 2 (79.5%) and those results, in 42 minutes of training
  on a laptop.
