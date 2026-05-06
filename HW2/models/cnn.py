import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", use_batchnorm=False):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=False))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class STL10CNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        channels=None,
        activation="relu",
        pool_type="max",
        use_batchnorm=False,
        dropout=0.0,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        blocks = []
        in_c = 3
        for out_c in channels:
            blocks.append(ConvBlock(in_c, out_c, activation, use_batchnorm))
            if pool_type == "max":
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pool_type == "avg":
                blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")
            in_c = out_c

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """返回最后一个卷积块中的卷积层(在最终池化之前)，获得更高分辨率的特征"""
        # 遍历features中的层，找到最后一个ConvBlock
        last_conv_block = None
        for layer in self.features:
            if isinstance(layer, ConvBlock):
                last_conv_block = layer
        if last_conv_block is None:
            raise RuntimeError("No convolutional layer found")
        # 返回该ConvBlock中的卷积层
        return last_conv_block.block[0]
