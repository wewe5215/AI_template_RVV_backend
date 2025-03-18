#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import unittest
import torch

from aitemplate.compiler import compile_model
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target

from modeling.SqueezeNet import build_squeezenet_backbone
from weight_utils import timm_export


def mark_output(y):
    """Mark output tensors for AITemplate optimization."""
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        # print("output_{} shape: {}".format(i, y_shape))


class SqueezeNetVerification(unittest.TestCase):
    def test_squeezenet(self):
        target = detect_target()
        batch_size = 1
        torch_dtype = torch.float32
        ait_dtype = "float32"
        # Create input tensor with NHWC layout
        x = Tensor(
            shape=[batch_size, 224, 224, 3],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        # Build SqueezeNet backbone
        model = build_squeezenet_backbone(activation="ReLU")
        model.name_parameter_tensor()  # mark parameter names for export
        # Forward input tensor through the model to get output tensor(s)
        y = model(x)
        mark_output(y)

        # Export weights from timm SqueezeNet model to AITemplate format.
        timm_exporter = timm_export("squeezenet1_0", pretrained=False)
        ait_params = timm_exporter.export_model(half=False)
        # print("ait_params")
        # for name, param in ait_params.items():
        #     print(f'name = {name}, param.shape = {param.shape}')

        pt_model = timm_exporter.pt_model.to(dtype=torch_dtype, device="cpu")
        pt_model.eval()

        # Compile the AITemplate model.
        module = compile_model(y, target, "./tmp", "squeezenet")
        for name, param in ait_params.items():
            module.set_constant_with_tensor(name, param)

        # Prepare input tensor for AITemplate (NHWC) and for torch (NCHW)
        x_ait = torch.rand([batch_size, 224, 224, 3], dtype=torch_dtype, device="cpu")
        # Normalize input as done in training
        x_ait -= torch.tensor([0.485, 0.456, 0.406])
        x_ait /= torch.tensor([0.229, 0.224, 0.225])
        x_pt = torch.transpose(x_ait, 1, 3).contiguous()

        with torch.no_grad():
            y_pt = pt_model(x_pt)
        # AITemplate compiled model produces output in NHWC: [N, 1, 1, num_classes]
        y_ait = torch.zeros([batch_size, 1, 1, 1000], dtype=torch_dtype, device="cpu")
        module.run_with_tensors([x_ait], [y_ait])

        torch.testing.assert_close(
            y_pt, y_ait.reshape([batch_size, 1000]), rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
