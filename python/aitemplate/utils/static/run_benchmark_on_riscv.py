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

def load_data_bert(batch_size, model_name):
    folder = f"metadata_{model_name}_{batch_size}"
    weights_file = f"{folder}/weights_file_{batch_size}.npz"
    io_file = f"{folder}/io_tensors_{batch_size}.npz"
    
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weight file not found: {weights_file}")
    if not os.path.exists(io_file):
        raise FileNotFoundError(f"Input tensor file not found: {io_file}")
    
    weight_data = np.load(weights_file, allow_pickle=True)
    # The data is accessible via its keys:
    weights = {key: FakeTorchTensor(weight_data[key]) for key in weight_data.files}
    
    # Depending on your shared library, you might need to convert tensors to numpy arrays.
    data = np.load(io_file, allow_pickle=True)
    x_input_obj = data["x_input"]
    x_input = x_input_obj.item() if getattr(x_input_obj, "dtype", None) == object else x_input_obj
    x_input = {k: FakeTorchTensor(v) for k, v in x_input.items()}
    y_output = FakeTorchTensor(data["y_output"])

    print(f"[Target] Loaded weights, input, and output data")
    return weights, x_input, y_output

def load_data(model_name, batch_size):
    metadata_folder = f"metadata_{model_name}_{batch_size}"
    weights_file = f"{metadata_folder}/weights_file_{batch_size}.npz"
    io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"
    
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

def benchmark(model_name, batch_size, mod=None, graph_mode=True):
    if "BERT" in model_name:
        weights, x_input, y_output = load_data_bert(batch_size, model_name)
        mod = Model(os.path.join(f"./{model_name}", "test.so"))
        mod.set_many_constants_with_tensors(weights)
        mod.fold_constants(sync=True)
        model_name = f"{model_name}_{batch_size}"
        t, _, __ = mod.benchmark_with_tensors(
            x_input,
            [y_output],
            count=10,
            repeat=1,
            graph_mode=graph_mode,
        )
        # benchmark
        t, _, __ = mod.benchmark_with_tensors(
            x_input,
            [y_output],
            count=10,
            repeat=1,
            graph_mode=graph_mode,
        )
    else:
        weights, x_input, y_output = load_data(model_name, batch_size)
        mod = Model(os.path.join(f"./{model_name}_{batch_size}", "test.so"))
        mod.set_many_constants_with_tensors(weights)
        mod.fold_constants(sync=True)

        t, _, __ = mod.benchmark_with_tensors(
            [x_input],
            [y_output],
            count=10,
            repeat=4,
            graph_mode=graph_mode,
        )
        # benchmark
        t, _, __ = mod.benchmark_with_tensors(
            [x_input],
            [y_output],
            count=10,
            repeat=4,
            graph_mode=graph_mode,
        )
    print(f"batch_size: {batch_size}, latency: {t}")
    benchmark_out = f"{model_name}_ait_benchmark.txt"
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
            benchmark(model_name, bs, graph_mode=use_graph)
    else:
        benchmark(model_name, batch_size, graph_mode=use_graph)


if __name__ == "__main__":
    main()