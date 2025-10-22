import os

import click
import pickle
import subprocess
from model import Model
from FakeTorchTensor import FakeTorchTensor
import numpy as np
import sys
import re
# Workaround: map 'numpy._core' to 'numpy.core'
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core


def load_data(batch_size, model_name):
    folder = f"metadata_{model_name}_{batch_size}"
    io_file = f"{folder}/io_tensors_{batch_size}.npz"

    data = np.load(io_file, allow_pickle=True)
    inputs = {}
    inputs["X"] = FakeTorchTensor(data["X"])
    if "gamma" in data.files:
        inputs["gamma"] = FakeTorchTensor(data["gamma"])
    if "beta" in data.files:
        inputs["beta"] = FakeTorchTensor(data["beta"])

    x4 = FakeTorchTensor(data["x4"])

    return inputs, x4

def run(model_name, batch_size, mod=None, graph_mode=True):
    inputs, x4 = load_data(batch_size, model_name)
    mod = Model(os.path.join(f"./{model_name}", "test.so"))
    mod.run_with_tensors(inputs, [x4])

    output_file = f"output_file_{model_name}_{batch_size}.npz"
    y_output_np = x4.cpu().detach().numpy().astype(np.float32)
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

    run(model_name, batch_size, graph_mode=use_graph)

if __name__ == "__main__":
    main()
