import numpy as np
from aitemplate.frontend import nn
from aitemplate.testing import detect_target
from aitemplate.compiler import ops

# Assume FireModule is defined as before.
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, activation="ReLU"):
        super().__init__()
        if detect_target().name() == "cuda":
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasReluFewChannels
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswishFewChannels
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")
        else:
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasRelu
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswish
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")
        self.squeeze = conv_op(in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, dtype="float")
        self.expand1x1 = conv_op(squeeze_channels, expand1x1_channels, kernel_size=1, stride=1, padding=0, dtype="float")
        self.expand3x3 = conv_op(squeeze_channels, expand3x3_channels, kernel_size=3, stride=1, padding=1, dtype="float")

    def forward(self, x):
        x = self.squeeze(x)
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        # Concatenate along the channel dimension (axis=3 in NHWC format)
        return ops.concatenate()([out1, out3], dim=3)

def identical_elem_tuple_to_int(param):
    """
    Convert tuples with all the same int elem to a single int 
    (ex. (3, 3, 3) --> 3)
    """
    if isinstance(param, int):
        return param
    if not isinstance(param, (list, tuple)) or not all(x == param[0] for x in param):
        raise RuntimeError(f"AIT supports square param values only, but got {param}")
    return param[0]

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # Convert to NHWC as expected by the converter.
        x_nhwc = x
        # Assume static shape is available in the tensor attributes.
        shape = [var._attrs["values"][0] for var in x_nhwc._attrs["shape"]]
        print(f'shape in adaptiveavgpool2d is {shape}')
        HI, WI, CI = shape[1], shape[2], shape[3]
        if CI % 2 != 0:
            raise RuntimeError(
                f"AIT avg_pool2d expects input channel dim to align w/ a multiple of 2 but got {CI}"
            )
        if HI != WI:
            raise RuntimeError(
                f"adaptive_avg_pool2d currently only supports square input H/W but got H: {HI} and W: {WI}"
            )
        # Compute stride and kernel size to reduce HI x WI to output_size x output_size.
        stride = HI // self.output_size
        kernel_size = HI - (self.output_size - 1) * stride
        # Use the plain AvgPool2d with computed parameters.
        result = nn.AvgPool2d(kernel_size, stride, 0)(x_nhwc)
        # Convert the result back to NCHW if needed.
        return result

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000, activation="ReLU"):
        super().__init__()
        self.num_classes = num_classes
        # Select convolution operator based on target and activation.
        if detect_target().name() == "cuda":
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasReluFewChannels
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswishFewChannels
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")
        else:
            if activation == "ReLU":
                conv_op = nn.Conv2dBiasRelu
            elif activation == "Hardswish":
                conv_op = nn.Conv2dBiasHardswish
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")
        
        # Initial conv layer: 7x7 conv, stride 2, padding 3, output channels=96.
        self.conv1 = conv_op(3, 96, 7, 2, 3, dtype="float")
        self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)
        
        # Fire modules
        self.fire2 = FireModule(in_channels=96,  squeeze_channels=16, expand1x1_channels=64,  expand3x3_channels=64,  activation=activation)
        self.fire3 = FireModule(in_channels=128, squeeze_channels=16, expand1x1_channels=64,  expand3x3_channels=64,  activation=activation)
        self.fire4 = FireModule(in_channels=128, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128, activation=activation)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire5 = FireModule(in_channels=256, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128, activation=activation)
        self.fire6 = FireModule(in_channels=256, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192, activation=activation)
        self.fire7 = FireModule(in_channels=384, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192, activation=activation)
        self.fire8 = FireModule(in_channels=384, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256, activation=activation)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire9 = FireModule(in_channels=512, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256, activation=activation)
        
        # Final conv layer (classifier conv).
        # In the original SqueezeNet, conv10 maps from 512 channels to num_classes.
        # Here we choose to map to 1000 channels and then use an FC to allow flexibility.
        self.dropout = nn.Dropout(0.5)
        self.conv10 = nn.Conv2dBiasRelu(512, 1000, kernel_size=1, stride=1, padding=0, dtype="float")
        
        # Final classification layers (like in ResNet)
        if num_classes is not None:
            # Assume the spatial size of conv10 output is 8x8 for a 224x224 input.
            # self.avgpool = nn.AvgPool2d(14, 1, 0)  # fixed kernel to reduce 8x8 to 1x1
            self.avgpool = AdaptiveAvgPool2d(1)
            # self.reshape = nn.Reshape()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = self.dropout(x)
        x = self.conv10(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            # Flatten the tensor: reshape from (N, 1000, 1, 1) to (N, 1000)
            # x = self.reshape(x, [x._size(0), -1])
        return x


def build_squeezenet_backbone(activation="ReLU", num_classes=1000):
    """
    Create a SqueezeNet backbone instance.
    
    Args:
        activation (str): The activation function to use ("ReLU" or "Hardswish").
        num_classes (int): Number of classes for the final classification layer.
            Set to None to use SqueezeNet as a feature extractor.
            
    Returns:
        SqueezeNet: an instance of the SqueezeNet backbone.
    """
    backbone = SqueezeNet(num_classes=num_classes, activation=activation)
    return backbone
