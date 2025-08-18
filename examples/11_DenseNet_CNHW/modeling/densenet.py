import numpy as np
from aitemplate.frontend import nn
from aitemplate.testing import detect_target
from aitemplate.compiler import ops
from aitemplate.frontend.nn import batch_norm
# ---------------------------------------------------------------------------
# Base class (similar to ResNet’s CNNBlockBase)
# ---------------------------------------------------------------------------
class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of the `forward()` method must be NHWC tensors.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

# ---------------------------------------------------------------------------
# Stem (same style as ResNet’s BasicStem)
# ---------------------------------------------------------------------------
class BasicStem(CNNBlockBase):
    """
    The standard stem: a convolution with a 7x7 kernel and stride 2, followed by max pooling.
    """
    def __init__(self, in_channels=3, out_channels=64, norm="BN", activation="ReLU"):
        super().__init__(in_channels, out_channels, 4)
        if detect_target().name() == "cuda":
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasReluFewChannels
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswishFewChannels
            else:
                raise NotImplementedError
        else:
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasRelu
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswish
            else:
                raise NotImplementedError
        self.conv0 = conv_op(in_channels, out_channels, 7, 2, 3, dtype="float")
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool(x)
        return x

# ---------------------------------------------------------------------------
# DenseNet‑specific blocks
# ---------------------------------------------------------------------------
class DenseLayer(nn.Module):
    """
    A single DenseLayer. It first applies a 1×1 bottleneck conv (reducing dimensionality)
    followed by a 3×3 conv. Its output is concatenated with the input along the channel axis.
    """
    def __init__(self, num_blocks, in_channels, growth_rate, bn_size=4, norm="BN", activation="ReLU"):
        super().__init__()
        mid_channels = bn_size * growth_rate
        if activation == "ReLU":
            if num_blocks < 2:
                conv1_op = nn.Conv2dBiasRelu
                conv2_op = nn.Conv2dBiasRelu
            else:
                conv1_op = nn.Conv2dCNHWBiasRelu
                conv2_op = nn.Conv2dCNHWBiasRelu
        else:
            raise NotImplementedError
        self.conv1 = conv1_op(in_channels, mid_channels, 1, 1, 0, dtype="float")
        self.conv2 = conv2_op(mid_channels, growth_rate, 3, 1, 1, dtype="float")
        if num_blocks < 2:
            self.concat = ops.concatenate()
        else:
            self.concat = ops.concatenate(cnhw=True)

    def forward(self, x):
        new_features = self.conv1(x)
        new_features = self.conv2(new_features)
        # Concatenate along the channel dimension (assuming NHWC, so dim=3)
        return self.concat([x, new_features], dim=3)

class DenseBlock(nn.Module):
    """
    A dense block is a sequential stack of DenseLayers.
    """
    def __init__(self, num_blocks, num_layers, in_channels, growth_rate, bn_size=4, norm="BN", activation="ReLU"):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(num_blocks, channels, growth_rate, bn_size, norm, activation)
            layers.append(layer)
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    """
    A transition layer downsamples the feature maps and reduces the number of channels.
    Typically, it performs a 1×1 conv followed by a 2×2 average pooling.
    """
    def __init__(self, in_channels, out_channels, num_blocks, norm="BN", activation="ReLU"):
        super().__init__()
        if activation == "ReLU":
            if num_blocks < 2:
                conv_op = nn.Conv2dBiasRelu
            else:
                conv_op = nn.Conv2dCNHWBiasRelu
        else:
            raise NotImplementedError
        self.conv = conv_op(in_channels, out_channels, 1, 1, 0, dtype="float")
        if num_blocks == 0:
            self.pool = nn.AvgPool2d(2, 2, 0)
        elif num_blocks == 1:
            self.pool = nn.AvgPool2dTranspose(2, 2, 0)
        elif num_blocks == 2:
            self.pool = nn.AvgPool2dCNHW(2, 2, 0)
        else:
            self.pool = nn.AvgPool2dCNHWTranspose(2, 2, 0)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DenseNet(nn.Module):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): the initial stem module.
            stages (list[list[nn.Module]]): the sequential stages (dense blocks and transitions).
            num_classes (int or None): if not None, a linear classifier is added.
            out_features (list[str] or None): names of intermediate outputs to return.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names = []
        self.stages = []

        if out_features is not None:
            # Only keep up to the maximum requested stage.
            num_stages = max([{"block1": 1, "block2": 2, "block3": 3, "block4": 4}.get(f, 0)
                              for f in out_features])
            stages = stages[:num_stages]

        # Assemble each stage
        cur_denseblock = 0
        cur_transision = 0
        for i, blocks in enumerate(stages):
            if i % 2 == 1:
                name = "transition" + str(cur_transision + 1)
                cur_transision = cur_transision + 1
            else:
                name = "denseblock" + str(cur_denseblock + 1)
                cur_denseblock = cur_denseblock + 1
            stage = nn.Sequential(*blocks)
            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)
            current_stride = int(current_stride * 2)
            self._out_feature_strides[name] = current_stride
            # Assume the last block in the stage has the attribute out_channels.
            self._out_feature_channels[name] = blocks[-1].out_channels if hasattr(blocks[-1], "out_channels") else None

        self.stage_names = tuple(self.stage_names)

        if num_classes is not None:
            # Global average pooling (assuming input features yield a 7x7 feature map)
            # final_channels = self._out_feature_channels[self.stage_names[-1]]
            # self.norm5 = getattr(batch_norm, "BatchNorm2d")(
            #     final_channels, eps=1e-05, permute_input_output=False
            # )
            self.avgpool = nn.AvgPool2dCNHWTranspose(7, 1, 0)
            self.fc = nn.Linear(self._out_feature_channels[self.stage_names[-1]], num_classes, dtype="float")

        if out_features is None:
            out_features = [self.stage_names[-1]]
        self._out_features = out_features
        self.reshape = nn.Reshape()

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            # x = self.norm5(x)
            x = self.avgpool(x)
            x = self.fc(x)
            if x._rank() == 2:
                x = self.reshape(x, [x._size(0), 1, 1, x._size(1)])
            return x
        return outputs

# ---------------------------------------------------------------------------
# Build DenseNet‑121 Backbone
# ---------------------------------------------------------------------------
def build_densenet_backbone():
    norm = "BN"
    activation = "ReLU"
    # DenseNet-121 configuration: 4 dense blocks with (6, 12, 24, 16) layers
    block_config = (6, 12, 24, 16)
    growth_rate = 32
    num_init_features = 64

    # Use the same stem as in ResNet.
    stem = BasicStem(in_channels=3, out_channels=num_init_features, norm=norm, activation=activation)

    stages = []
    num_features = num_init_features
    # For each dense block, create a DenseBlock and, if not the last block, add a Transition.
    for i, num_layers in enumerate(block_config):
        dense_block = DenseBlock(i, num_layers, num_features, growth_rate, bn_size=4, norm=norm, activation=activation)
        stages.append([dense_block])
        num_features = dense_block.out_channels
        if i != len(block_config) - 1:
            # Transition reduces channels by 50%.
            trans = Transition(num_features, num_features // 2, i,norm=norm, activation=activation)
            stages.append([trans])
            num_features = num_features // 2

    # Wrap all stages in a DenseNet instance with a final classifier.
    return DenseNet(stem, stages, num_classes=1000)

if __name__ == "__main__":
    model = build_densenet_backbone()
    # Traverse all submodules
    for module in model.modules():
        print(module)
