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
import os

from typing import Dict, List

import click
import numpy as np
import torch
from aitemplate.compiler import compile_model, Model

from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from modeling.bert import BertBaseEncodersOnly, BertBaseUncased
from modeling.torch_model import BertBaseUncased as BertPt
import importlib
dt = importlib.import_module("aitemplate.testing.detect_target")
dt.IS_CPU_BACKEND = True
dt = importlib.import_module("aitemplate.compiler.compiler")
dt.IS_REMOTE_COMPILE = True
from aitemplate.utils.remote_send_receive_files import (
    transfer_folder, 
    check_remote_file_exists, 
    retrieve_confirmation_file, 
    poll_for_confirmation,
    TARGET_USER,
    TARGET_IP,
    remote_run_program_send_back_result
)
def mark_output(y: Tensor) -> None:
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def create_bert_inputs(
    batch_size: int, seq_length: int, dtype: str = "int64"
) -> List[Tensor]:
    input_ids = Tensor(
        shape=[batch_size, seq_length],
        name="input_ids",
        dtype=dtype,
        is_input=True,
    )
    token_type_ids = Tensor(
        shape=[batch_size, seq_length],
        name="token_type_ids",
        dtype=dtype,
        is_input=True,
    )
    position_ids = Tensor(
        shape=[batch_size, seq_length],
        name="position_ids",
        dtype=dtype,
        is_input=True,
    )
    return [input_ids, token_type_ids, position_ids]


def create_bert_encoders_input(
    batch_size: int, seq_length: int, hidden: int, dtype: str = "float32"
):
    encoder_input = Tensor(
        shape=[batch_size, seq_length, hidden],
        name="input",
        dtype=dtype,
        is_input=True,
    )
    return [encoder_input]


def create_bert_inputs_pt(
    batch_size: int, seq_length: int, dtype: torch.dtype = torch.int64
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randn(batch_size, seq_length).to(dtype)
    token_type_ids = torch.randn(batch_size, seq_length).to(dtype)
    position_ids = torch.randn(batch_size, seq_length).to(dtype)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def create_bert_encoders_inputs_pt(
    batch_size: int, seq_length: int, hidden_size: int
) -> Dict[str, torch.Tensor]:
    encoder_input = torch.randn([batch_size, seq_length, hidden_size])
    return {"input": encoder_input}


def map_pt_params(
    ait_bert, pt_bert, batch_size: int, seq_length: int
) -> Dict[str, torch.Tensor]:
    pt_params = dict(pt_bert.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_bert.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            mapped_pt_params[ait_name] = pt_params[name]
            continue

        if name.endswith("self.qkv.weight"):
            prefix = name[: -len("qkv.weight")]
            q_weight = pt_params[prefix + "query.weight"]
            k_weight = pt_params[prefix + "key.weight"]
            v_weight = pt_params[prefix + "value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_pt_params[ait_name] = qkv_weight
        elif name.endswith("self.qkv.bias"):
            prefix = name[: -len("qkv.bias")]
            q_bias = pt_params[prefix + "query.bias"]
            k_bias = pt_params[prefix + "key.bias"]
            v_bias = pt_params[prefix + "value.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            mapped_pt_params[ait_name] = qkv_bias
        elif name.endswith("self.proj.weight"):
            prefix = name[: -len("self.proj.weight")]
            pt_name = prefix + "output.dense.weight"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("self.proj.bias"):
            prefix = name[: -len("self.proj.bias")]
            pt_name = prefix + "output.dense.bias"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("cu_length"):
            cu_len = np.cumsum([0] + [seq_length] * batch_size).astype("int32")
            mapped_pt_params[ait_name] = torch.from_numpy(cu_len)
        else:
            if "weight_indice" in name:
                continue
            print(name)
            pt_param = pt_bert.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param

    return mapped_pt_params


def benchmark(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    mod: Model,
    graph_mode: bool,
    encoders_only: bool,
    activation: str,
):
    if encoders_only:
        inputs = create_bert_encoders_inputs_pt(batch_size, seq_length, hidden_size)
    else:
        inputs = create_bert_inputs_pt(batch_size, seq_length)
    if dt.IS_REMOTE_COMPILE == False:
        outputs = [torch.empty(mod.get_output_maximum_shape(0))]
        # warm up
        t, _, __ = mod.benchmark_with_tensors(
            inputs,
            outputs,
            count=100,
            repeat=4,
            graph_mode=graph_mode,
        )
        # benchmark
        t, _, __ = mod.benchmark_with_tensors(
            inputs,
            outputs,
            count=100,
            repeat=4,
            graph_mode=graph_mode,
        )
    else:
        model_name = f"BERT_{activation}_{batch_size}_{seq_length}"
        metadata_folder = f"metadata_{model_name}_{batch_size}"
        io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"
        x_input_np = {}
        for name, tensor in inputs.items():
            # Move to CPU and detach if it's a tensor
            if hasattr(tensor, "cpu"):
                array = tensor.detach().cpu().numpy()
            else:
                # already a NumPy array, scalar, or something else
                array = np.array(tensor)

            # Cast to float32 *only* if it’s a floating type
            if np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
            elif np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.int64)   # or np.int64 if your model expects that
            # you can add more cases if needed (bool, etc.)
            elif np.issubdtype(array.dtype, np.bool_):
                array = array.astype(np.bool_)

            x_input_np[name] = array
        y_ait = torch.zeros([batch_size, seq_length, hidden_size], dtype=torch.float32, device="cpu")
        y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
        io_data = {"x_input": x_input_np, "y_output": y_output_np}
        np.savez_compressed(io_file, **io_data)
    print(f"batch_size: {batch_size}, seq_length: {seq_length}, latency: {t}")
    dev_flag = os.environ.get("HIP_VISIBLE_DEVICES", "-1")
    dev_flag = dev_flag.replace(",", "_")
    with open(f"bert_ait_benchmark_dev_{dev_flag}.txt", "a") as f:
        f.write(f"batch_size: {batch_size}, seq_length: {seq_length}, latency: {t}\n")


def compile_module(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    activation: str,
    use_fp16_acc: bool,
    encoders_only: bool,
    pt_model: torch.nn.Module,
    is_remote_compile: bool,
    metadata_folder: str,
    model,
) -> None:
    model_name = f"BERT_{activation}_{batch_size}_{seq_length}"
    if is_remote_compile:
        target = detect_target(use_fp16_acc=use_fp16_acc)
    else:
        target = detect_target(use_fp16_acc=use_fp16_acc, xnnpack_path="/Users/wewe5215/Desktop/XNNPACK", is_remote_compile=is_remote_compile)

    if encoders_only:
        inputs = create_bert_encoders_input(batch_size, seq_length, hidden_size)
    else:
        inputs = create_bert_inputs(batch_size, seq_length)

    # Mark all parameters with name same to PyTorch name convention
    model.name_parameter_tensor()
    # Forward the input tensor to the model, get output tensor
    y = model(*inputs)
    # Mark output tensor
    mark_output(y)

    mod = compile_model(y, target, "./tmp", model_name, remote_compile=is_remote_compile)
    return mod


@click.command()
@click.option("--batch-size", type=int, default=0, help="Inference batch size")
@click.option("--seq-length", type=int, default=0, help="Inference sequence length")
@click.option(
    "--activation",
    type=str,
    default="gelu",
    help="Activation function applied on BERT, currently only support fast_gelu on Rocm. CUDA supports both gelu and fast_gelu. No effect if framework is pt.",
)
@click.option(
    "--graph-mode",
    type=bool,
    default=True,
    help="Use CUDA graph or not. hipGraph is not supported yet. No effect if framework is pt.",
)
@click.option(
    "--use-fp16-acc",
    type=bool,
    default=True,
    help="Use fp16 accumulation or not (TensorRT is using fp16_acc). No effect if framework is pt.",
)
@click.option(
    "--use-pretrained-pt-model",
    type=bool,
    default=True,
    help="Whether or not to use the pretrained BERT model weights.",
)
@click.option(
    "--encoders-only",
    type=bool,
    default=True,
    help="Whether or not to run the BERT benchmark with encoders only. If enabled, only the transformer blocks without BERT embeddings are benchmarked.",
)
def compile_and_benchmark(
    batch_size: int,
    seq_length: int,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    use_pretrained_pt_model: bool,
    encoders_only: bool,
):
    if detect_target().name() == "rocm":
        graph_mode = False
        assert activation in (
            "fast_gelu"
        ), f"Unsupported activation: {activation} on rocm"

    pt_model = BertPt(pretrained=use_pretrained_pt_model)._model
    pt_model.eval()
    hidden_size = pt_model.config.hidden_size

    if batch_size < 1:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        batch_sizes = [batch_size]

    if seq_length < 1:
        seq_lengths = (
            [64, 128, 384, 512, 1024, 4096] if encoders_only else [64, 128, 384, 512]
        )
    else:
        seq_lengths = [seq_length]

    for seq_length in seq_lengths:
        for bs in batch_sizes:
            mod = compile_module(
                bs,
                seq_length,
                hidden_size,
                activation,
                use_fp16_acc,
                encoders_only,
                pt_model,
                is_remote_compile = dt.IS_REMOTE_COMPILE
            )
            benchmark(bs, seq_length, hidden_size, mod, graph_mode, encoders_only, activation)


if __name__ == "__main__":
    torch.manual_seed(4896)
    compile_and_benchmark(batch_size=1, seq_length = 64)
