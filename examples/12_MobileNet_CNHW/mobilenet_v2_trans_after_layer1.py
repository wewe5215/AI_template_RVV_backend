import numpy as np
from aitemplate.frontend import nn
from aitemplate.testing import detect_target
from aitemplate.compiler import ops
from aitemplate.compiler.ops.common.epilogue import FuncEnum

# -----------------------------------------------------------------------------
# Base class (same as provided)
# -----------------------------------------------------------------------------
class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels, and a stride.
    The input and output of the `forward()` method must be NHWC tensors.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

# -----------------------------------------------------------------------------
# MobileNetV2 Stem
# -----------------------------------------------------------------------------
class MobileNetV2Stem(CNNBlockBase):
    """
    The stem of MobileNetV2 consists of a single 3×3 convolution with stride 2
    and padding 1 followed by a non-linear activation.
    """
    def __init__(self, in_channels=3, out_channels=32, norm="BN", activation="ReLU6"):
        # For a 3x3 conv with stride 2, no extra pooling is needed.
        # We pass an effective stride value (here 2) to the base.
        super().__init__(in_channels, out_channels, 2)
        conv_op = None
        if activation == "ReLU6":
            conv_op = nn.Conv2dBiasRelu6Transpose
        elif activation == "ReLU":
            conv_op = nn.Conv2dBiasReluTranspose
        else:
            raise NotImplementedError
        # Use kernel size 3, stride 2, padding 1.
        self.conv = conv_op(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dtype="float")

    def forward(self, x):
        # x is assumed to be NHWC.
        return self.conv(x)

# -----------------------------------------------------------------------------
# MobileNetV2 Inverted Residual Block
# -----------------------------------------------------------------------------
class MobileInvertedResidual(nn.Module):
    """
    An inverted residual block for MobileNetV2.
    
    The block structure is:
      1. (Optional) Expansion 1x1 convolution: expands channels from in_channels
         to in_channels * expansion_factor, with non-linearity.
      2. Depthwise 3x3 convolution: groups equal to expanded channels,
         with stride and padding 1.
      3. Projection 1x1 convolution: projects back to out_channels.
    
    A residual connection is used if stride == 1 and in_channels == out_channels.
    """
    def __init__(self, in_channels, out_channels, stride, expansion_factor, block_idx, num_blocks, norm="BN", activation="ReLU6"):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion_factor

        self.layers = []  # to collect layers sequentially

        # If expansion_factor != 1, add expansion conv (1x1)
        # print(f'block_idx = {block_idx}, num_blocks = {num_blocks}')
        if expansion_factor != 1:
            if activation == "ReLU6":
                conv_op = nn.Conv2dCNHWBiasRelu6
            elif activation == "ReLU":
                conv_op = nn.Conv2dCNHWBiasRelu
            self.expansion_conv = conv_op(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, dtype="float")
        else:
            self.expansion_conv = None

        # In depthwise conv, the number of groups equals the number of input channels (hidden_dim).
        # We assume this operator accepts an extra parameter "groups".
        if activation == "ReLU6":
            self.depthwise_conv = nn.Conv2dCNHWDepthwiseBiasRelu6(
                hidden_dim if self.expansion_conv is not None else in_channels,
                hidden_dim if self.expansion_conv is not None else in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=(hidden_dim if self.expansion_conv is not None else in_channels),
                dtype="float"
            )
        elif activation == "ReLU":
            self.depthwise_conv = nn.Conv2dCNHWDepthwiseBiasRelu(
                hidden_dim if self.expansion_conv is not None else in_channels,
                hidden_dim if self.expansion_conv is not None else in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=(hidden_dim if self.expansion_conv is not None else in_channels),
                dtype="float"
            )
        if self.use_res_connect:
            self.projection_conv = nn.Conv2dCNHWBiasAdd(
                hidden_dim if self.expansion_conv is not None else in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype="float")
        else:
            self.projection_conv = nn.Conv2dCNHWBias(
                hidden_dim if self.expansion_conv is not None else in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype="float")
    

    def forward(self, x):
        # x is NHWC.
        identity = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)
        x = self.depthwise_conv(x)
        if self.use_res_connect:
            x = self.projection_conv(x, identity)
        else:
            x = self.projection_conv(x)
        return x

# -----------------------------------------------------------------------------
# make_mobile_stage Function
# -----------------------------------------------------------------------------
def make_mobile_stage(idx, num_blocks, in_channels, out_channels, expansion_factor, stride, norm="BN", activation="ReLU6"):
    """
    Creates a stage by stacking 'num_blocks' MobileInvertedResidual blocks.
    
    Args:
        num_blocks (int): Number of MobileInvertedResidual blocks to stack.
        in_channels (int): Number of input channels for the stage.
        out_channels (int): Number of output channels (after projection) for each block.
        expansion_factor (int): Expansion factor for the inverted residual block.
        stride (int): Stride for the first block in the stage; subsequent blocks have stride 1.
        norm (str): Normalization method.
        activation (str): Activation function.
        
    Returns:
        nn.Sequential: A sequential container of MobileInvertedResidual blocks.
    """
    blocks = []
    for i in range(num_blocks):
        block_stride = stride if i == 0 else 1
        block = MobileInvertedResidual(in_channels, out_channels, block_stride,
                                       expansion_factor=expansion_factor, block_idx=idx, num_blocks=i,
                                       norm=norm, activation=activation)
        blocks.append(block)
        in_channels = out_channels
    return nn.Sequential(*blocks)

# -----------------------------------------------------------------------------
# MobileNetV2 Backbone
# -----------------------------------------------------------------------------
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, norm="BN", activation="ReLU6"):
        """
        Builds the MobileNetV2 backbone.
        
        Args:
            num_classes (int): Number of classes for classification.
            width_mult (float): Width multiplier to adjust channel counts.
            norm (str): Normalization type.
            activation (str): Activation function to use.
        """
        super().__init__()
        # Stem: 3x3 conv, stride 2.
        input_channel = int(32 * width_mult)
        self.stem = MobileNetV2Stem(in_channels=3, out_channels=input_channel, norm=norm, activation=activation)
        
        # Inverted residual block configuration:
        # (expansion_factor, output_channels, num_blocks, stride)
        inverted_residual_settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        blocks = []
        # Set current input channel from the stem.
        in_channels = input_channel
        idx = 0
        for t, c, n, s in inverted_residual_settings:
            out_channels = int(c * width_mult)
            # Here you could use the make_mobile_stage function instead of the inline loop.
            # For demonstration, we use the function to create each stage.
            stage = make_mobile_stage(idx = idx, num_blocks=n,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      expansion_factor=t,
                                      stride=s,
                                      norm=norm,
                                      activation=activation)
            blocks.append(stage)
            in_channels = out_channels
            idx = idx + 1
        self.blocks = nn.Sequential(*blocks)
        
        # Final convolution layer: 1x1 conv to expand to last_channel.
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        if activation == "ReLU6":
            conv_op = nn.Conv2dCNHWBiasRelu6
        elif activation == "ReLU":
            conv_op = nn.Conv2dCNHWBiasRelu
        self.final_conv = conv_op(in_channels, last_channel, kernel_size=1, stride=1, padding=0, dtype="float")
        
        # Optionally add global average pooling and a classifier.
        if num_classes is not None:
            self.avgpool = nn.AvgPool2dCNHWTranspose(7, 1, 0)
            self.fc = nn.Linear(last_channel, num_classes, dtype="float")
        
        # For consistency with other backbones, store output features if needed.
        self._out_features = ["final"]
        self.reshape = nn.Reshape()

    def forward(self, x):
        # x is NHWC.
        x = self.stem(x)         # Output reduced spatially.
        x = self.blocks(x)     # Apply inverted residual blocks.
        x = self.final_conv(x)   # Final 1×1 conv.
        # If classification is desired, apply pooling and fc.
        if hasattr(self, "fc"):
            x = self.avgpool(x)
            x = self.fc(x)
            if x._rank() == 2:
                x = self.reshape(x, [x._size(0), 1, 1, x._size(1)])
        return x

# -----------------------------------------------------------------------------
# Builder Function
# -----------------------------------------------------------------------------
def build_mobilenetv2_backbone(num_classes=1000, width_mult=1.0, norm="BN", activation="ReLU6"):
    """
    Create a MobileNetV2 backbone.
    
    Args:
        num_classes (int): Number of classes for the classifier.
        width_mult (float): Width multiplier.
        norm (str): Normalization method.
        activation (str): Activation function.
        
    Returns:
        MobileNetV2 instance.
    """
    model = MobileNetV2(num_classes=num_classes, width_mult=width_mult, norm=norm, activation=activation)
    return model

# -----------------------------------------------------------------------------
# Example Main (for testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a MobileNetV2 model instance.
    model = build_mobilenetv2_backbone(num_classes=1000, width_mult=1.0, norm="BN", activation="ReLU6")
    # Print a summary of modules to inspect structure.
    for module in model.modules():
        print(module)
