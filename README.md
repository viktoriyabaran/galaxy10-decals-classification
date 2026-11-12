# Galaxy10 DECaLS

Three models classify galaxy images from [Galaxy10 DECaLS](https://github.com/henrysky/Galaxy10) into 10 morphological classes.

## Results

| Model | Input | Parameters | Accuracy | Macro F1 | Training time | Test prediction |
|---|---|---|---|---|---|---|
| Random Forest | 64×64 | — | 58.1% | 0.527 | 16 s | 0.07 s |
| Small CNN | 64×64 | 392 K | 79.5% | 0.779 | 6 min | 0.49 s |
| ConvNeXt-Tiny | 128×128 | 27.8 M | **87.3%** | **0.858** | 42 min | 6.4 s |

Measured on an Apple M4.

More details: [`results/report.pdf`](results/report.pdf)
