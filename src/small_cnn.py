from torch import nn


def conv_block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),  # channels_out filters of 3x3
        nn.BatchNorm2d(channels_out),   # keeps the numbers in a stable range -> faster, steadier training
        nn.ReLU(),                      # negative -> 0: "pattern not here"
        nn.MaxPool2d(2),                # keep the strongest answer in each 2x2 square: 64 -> 32 -> ...
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 64x64 -> 32x32: edges, bright spots
            conv_block(32, 64),    # 32x32 -> 16x16: blobs, curves
            conv_block(64, 128),   # 16x16 -> 8x8:   larger shapes
            conv_block(128, 256),  # 8x8   -> 4x4:   whole structures (arms, bars, disks)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # average each of the 256 feature maps to one number
            nn.Flatten(),              # (batch, 256, 1, 1) -> (batch, 256)
            nn.Dropout(0.3),           # during training, randomly switch off 30% of them (less memorising)
            nn.Linear(256, n_classes), # 256 numbers -> 10 scores, one per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))
