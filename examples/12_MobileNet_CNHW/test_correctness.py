import unittest
import torch
import numpy as np
import os
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target
import subprocess

# Import your backbone builder and weight converter for MobileNetV2.
from mobilenet_v2 import build_mobilenetv2_backbone
from mobilenet_v2_trans_after_layer1 import build_mobilenetv2_backbone as backbone_trans_after_layer1
from weight_utils import export_mobilenet
from aitemplate.utils.remote_send_receive_files import (
    transfer_folder, 
    check_remote_file_exists, 
    retrieve_confirmation_file, 
    poll_for_confirmation,
    TARGET_USER,
    TARGET_IP,
    remote_run_program_send_back_result
)
target_dir  = f"/home/{TARGET_USER}/Desktop/AITemplate_Benchmark_on_XNNPACK" # Target directory to store files
def mark_output(y):
    """
    Mark the output tensors for AITemplate optimization.
    
    Parameters
    ----------
    y : List[Tensor] or Tensor
        The output tensor(s) to be marked.
    """
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % i
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


class MobileNetV2Verification(unittest.TestCase):
    def test_mobilenetv2(self):
        target = detect_target()
        batch_size = 1
        torch_dtype = torch.float32
        ait_dtype = "float32"

        # Create input tensor (AITemplate expects NHWC)
        x = Tensor(
            shape=[batch_size, 224, 224, 3],
            dtype=ait_dtype,
            name="input0",
            is_input=True,
        )
        
        # Build MobileNetV2 backbone (you need to implement build_mobilenetv2_backbone accordingly)
        # model = build_mobilenetv2_backbone()
        model = backbone_trans_after_layer1()
        model.name_parameter_tensor()
        # Forward the input to the model
        y = model(x)
        mark_output(y)
        model_name = "cnhw_mobilenetv2_trans_after_layer1"
        module = compile_model(y, target, "./tmp", f"{model_name}_{batch_size}", remote_compile = True)

        # # Use the MobileNetV2 converter; this exporter should be implemented to support MobileNetV2.
        weight_exporter = export_mobilenet("mobilenetv2", pretrained=True)
        ait_params = weight_exporter.export_model(half=False)
        np_weights = {}
        for k, v in ait_params.items():
            np_weights[k] = v.detach().cpu().numpy().astype(np.float32)

        metadata_folder = f"metadata_{model_name}_{batch_size}"
        os.makedirs(metadata_folder, exist_ok=True)
        weights_file = f"{metadata_folder}/weights_file_{batch_size}.npz"
        io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"

        np.savez_compressed(weights_file, **np_weights)
        # ait model expects NHWC format
        x_ait = torch.rand([batch_size, 224, 224, 3], dtype=torch_dtype, device="cpu")
        # center the input wrt the training data for numerical stability
        x_ait -= torch.tensor([0.485, 0.456, 0.406])
        x_ait /= torch.tensor([0.229, 0.224, 0.225])
        y_ait = torch.zeros([batch_size, 1, 1, 1000], dtype=torch_dtype, device="cpu")
        x_input_np = x_ait.cpu().detach().numpy().astype(np.float32)
        y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
        io_data = {"x_input": x_input_np, "y_output": y_output_np}
        np.savez_compressed(io_file, **io_data)
        transfer_folder(metadata_folder, TARGET_USER, TARGET_IP, target_dir)
        remote_run_program_send_back_result(target_dir, "static/test_correctness_on_riscv.py", model_name, batch_size)

        pt_model = weight_exporter.pt_model.to(dtype=torch_dtype, device="cpu")
        pt_model.eval()
        # torch model expects NCHW format
        x_pt = torch.transpose(x_ait, 1, 3).contiguous()
        with torch.no_grad():
            y_pt = pt_model(x_pt)

        output_file = f"output_file_{model_name}_{batch_size}.npz"
        output_np = np.load(output_file, allow_pickle=True)
        y_ait = torch.from_numpy(output_np["y_output"].reshape([batch_size, 1000]))
        torch.testing.assert_close(
            y_pt, y_ait.reshape([batch_size, 1000]), rtol=1, atol=1e-1
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
