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


def load_data(model_name, batch_size):
    weights_file = f"static/mobilenetv2_weights_file.npz"
    io_file = f"static/mobilenetv2_io_tensors_{batch_size}.npz"
    
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

def benchmark(model_name, batch_size, mod=None, graph_mode=True):
    weights, x_input, y_output = load_data(model_name, batch_size)
    model_name = f"{model_name}_{batch_size}"
    mod = Model(os.path.join(f"./{model_name}", "test.so"))
    mod.set_many_constants_with_tensors(weights)
    mod.fold_constants(sync=True)

    t, _, __ = mod.benchmark_with_tensors(
        [x_input],
        [y_output],
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    # benchmark
    t, _, __ = mod.benchmark_with_tensors(
        [x_input],
        [y_output],
        count=100,
        repeat=4,
        graph_mode=graph_mode,
    )
    print(f"batch_size: {batch_size}, latency: {t}")
    dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    dev_flag = dev_flag.replace(",", "_")
    benchmark_out = f"{model_name}_ait_benchmark_dev_{dev_flag}.txt"
    with open(benchmark_out, "a") as f:
        f.write(f"batch_size: {batch_size}, latency: {t}\n")
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
    use_graph = False
    if batch_size < 1:
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            # compile_module("resnet50", bs, use_fp16_acc=use_fp16_acc)
            benchmark(model_name, bs, graph_mode=use_graph)
    else:
        # compile_module("resnet50", batch_size, use_fp16_acc=use_fp16_acc)
        benchmark(model_name, batch_size, graph_mode=use_graph)


if __name__ == "__main__":
    main()