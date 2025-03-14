import os

import click
import pickle
import subprocess
from model import Model
from FakeTorchTensor import FakeTorchTensor
import numpy as np
import sys

# Workaround: map 'numpy._core' to 'numpy.core'
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core


def load_data(batch_size):
    weights_file = f"static/weights_file_pruned_{batch_size}.npz"
    io_file = f"static/io_tensors_{batch_size}.npz"
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weight file not found: {weights_file}")
    if not os.path.exists(io_file):
        raise FileNotFoundError(f"Input tensor file not found: {io_file}")
    
    weight_data = np.load(weights_file, allow_pickle=True)
    # The data is accessible via its keys:
    weights = {key: FakeTorchTensor(weight_data[key]) for key in weight_data.files}
    
    # Depending on your shared library, you might need to convert tensors to numpy arrays.
    data = np.load(io_file, allow_pickle=True)
    x_input = FakeTorchTensor(data["x_input"])
    y_output = FakeTorchTensor(data["y_output"])

    
    print(f"[Target] Loaded weights, input, and output data")
    return weights, x_input, y_output

def transfer_file(file: str, target_user: str, target_ip: str, target_dir: str):
    """
    
    Parameters:
        file (str): The path to the file to be transferred.
        target_user (str): The username on the target machine.
        target_ip (str): The IP address of the target machine.
        target_dir (str): The destination directory on the target machine.
    """
    subprocess.run(
        ["scp", file, f"{target_user}@{target_ip}:{target_dir}"],
        check=True
    )
    print("[Host] file transferred successfully.")

def run(model_name, batch_size, mod=None, graph_mode=True):
    model_name = f"{model_name}_{batch_size}"
    mod = Model(os.path.join(f"./{model_name}", "test.so"))
    weights, x_input, y_output = load_data(batch_size)
    for name, param in weights.items():
            mod.set_constant_with_tensor(name, param)


    mod.fold_constants(sync=True)
    mod.run_with_tensors([x_input], [y_output])

    output_file = f"output_file_pruned_{model_name}.npz"
    y_output_np = y_output.cpu().detach().numpy().astype(np.float32)
    np.savez_compressed(output_file, y_output=y_output_np)
@click.command()
@click.option("--model-name", type=str, default="resnet50", help="Model name to use")
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=False,
    help="Whether to use FP16 for accumulation (similar to TensorRT)",
)
@click.option("--use-graph", type=bool, default=True, help="Whether to use CUDA graph")
@click.option("--batch-size", type=int, default=0, help="Batch size")
def main(model_name, use_fp16_acc=False, use_graph=True, batch_size=0):
    use_graph = False  # This seems redundant, but keeping it as per original logic

    if batch_size < 1:
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            # compile_module(model_name, bs, use_fp16_acc=use_fp16_acc)
            run(model_name, bs, graph_mode=use_graph)
    else:
        # compile_module(model_name, batch_size, use_fp16_acc=use_fp16_acc)
        run(model_name, batch_size, graph_mode=use_graph)

if __name__ == "__main__":
    main()