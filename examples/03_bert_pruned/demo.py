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
import click

import torch
import os
from transformers import BertTokenizer
import importlib
from modeling.bert import BertBaseEncodersOnly, BertBaseUncased
dt = importlib.import_module("aitemplate.testing.detect_target")
dt.IS_CPU_BACKEND = True
dt = importlib.import_module("aitemplate.compiler.compiler")
dt.IS_REMOTE_COMPILE = True
from benchmark_ait import compile_module, map_pt_params, f32_data_pruning_column_wise_with_ratio
from modeling.torch_model import BertBaseUncased as BertPt
import numpy as np
import math
from aitemplate.utils.remote_send_receive_files import (
    transfer_folder, 
    check_remote_file_exists, 
    retrieve_confirmation_file, 
    poll_for_confirmation,
    TARGET_USER,
    TARGET_IP,
    remote_run_program_send_back_result
)
target_dir  = f"/home/{TARGET_USER}/Desktop/AITemplate_Benchmark_on_XNNPACK" 
pruning_ratio = 0.25
def prune_model_weights(np_weights, pruning_ratio):
    """
    Processes a dictionary of model weights (including both kernels and biases). For each weight
    kernel (key containing 'weight' and 'conv'), it prunes the weight column-wise according to the given ratio.
    For weights with dimension 4 (assumed to be of shape
    (output_channel, kernel_height, kernel_width, input_channel)),
    it reshapes them to 2D with shape (output_channel, kernel_height * kernel_width * input_channel).
    The corresponding bias is retained.
    
    Parameters:
      np_weights: dict, keys are layer names and values are numpy arrays (weights or biases)
      mr: block size (number of rows per block) for the pruning routine.
      pruning_ratio: fraction of columns to prune (e.g., 0.5 means prune bottom 50% columns).
      
    Returns:
      new_model: dict, containing:
          - For each weight key: new entries for "layer_weight_pruned" and "layer_weight_indice"
          - For each bias key: the bias is retained unmodified.
    """

    new_model = {}
    for key, value in np_weights.items():
        print(key, value.dtype, type(value))
        if "bert_embeddings" in key or "LayerNorm" in key or "cu_length" in key:
            new_model[key] = value
            continue
        if "weight" in key and "indice" not in key:
            if value.ndim == 2:
                print(f"value.shape = {value.shape}")
                output_channel, input_channel = value.shape
            else:
                raise ValueError(f"Unsupported weight dimension {value.ndim} for key {key}")

            lmul = 2
            nr = lmul * (256 / 32)  # 32 for float32
            mr = 10
            pruned_weight, indices = f32_data_pruning_column_wise_with_ratio(value, nr, mr, pruning_ratio)
            new_model[key] = pruned_weight
            new_model[key + "_indice"] = indices
            for indice in indices:
                if indice >= input_channel:
                    print(f'{indice} out of range')
            part1 = math.ceil((output_channel) / mr)
            part2 = math.ceil(input_channel * (1 - pruning_ratio))
            print(f'{key} is pruned with mr = {mr}, lmul = {lmul}; {{{part1}, {part2}}}, indice shape = {indices.shape}, pruned_weight shape = {pruned_weight.shape}')
            bias_key = key.replace("weight", "bias")
            if bias_key in np_weights:
                new_model[bias_key] = np_weights[bias_key]
    return new_model

def prepare_data(prompt: str, model_path: str):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    result = tokenizer(prompt, return_attention_mask=False, return_tensors="pt")
    target_size = result["input_ids"].size()
    if target_size[1] > 512:
        raise ValueError("Sequence length > 512 is not supported")

    result["position_ids"] = (
        torch.arange(target_size[1], dtype=torch.int64)
        .reshape(result["input_ids"].size())
        .contiguous()
    )
    return result

def handling_tensor_to_numpy(tensor):
    if hasattr(tensor, "cpu"):
        array = tensor.detach().cpu().numpy()
    else:
        # already a NumPy array, scalar, or something else
        array = np.array(tensor)

    if np.issubdtype(array.dtype, np.floating):
        # Keep float32/64 as-is, but you can normalize if needed
        array = array.astype(np.float32)  # optional: promote half to float32
    elif np.issubdtype(array.dtype, np.integer):
        # Preserve exact integer width
        if array.dtype == np.int32:
            array = array.astype(np.int32)
        elif array.dtype == np.int64:
            array = array.astype(np.int64)
        elif array.dtype == np.int16:
            array = array.astype(np.int16)
        elif array.dtype == np.int8:
            array = array.astype(np.int8)
    elif np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.bool_)
    return array

def run_model(
    prompt: str,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    verify: bool,
    model_path="bert-base-uncased",
):
    inputs = prepare_data(prompt, model_path)
    inputs_pt = {name: data for name, data in inputs.items()}
    batch_size, seq_len = inputs["input_ids"].size()

    pt_model = BertPt(model_path=model_path, pretrained=True)._model
    pt_model.eval()
    hidden_size = pt_model.config.hidden_size
    model_name = f"BERT_pruned_{int(pruning_ratio)}_{activation}_{batch_size}_{seq_len}"
    metadata_folder = f"metadata_{model_name}_{batch_size}"
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder, exist_ok=True)
        print(f"Created directory: {metadata_folder}")
    encoders_only = False
    if encoders_only:
        model = BertBaseEncodersOnly(batch_size, seq_len, hidden_act=activation)
    else:
        model = BertBaseUncased(batch_size, seq_len, hidden_act=activation)
    mod = compile_module(
        batch_size, seq_len, hidden_size, activation, use_fp16_acc, False, pt_model, \
        is_remote_compile=dt.IS_REMOTE_COMPILE, metadata_folder=metadata_folder, model=model
    )
    params = map_pt_params(model, pt_model, batch_size, seq_len)

    if dt.IS_REMOTE_COMPILE == False:
        mod.set_many_constants_with_tensors(params)
        mod.fold_constants(sync=True)
        outputs = [torch.empty(mod.get_output_maximum_shape(0))]
        mod.run_with_tensors(inputs_pt, outputs, graph_mode=graph_mode)
    else:
        np_weights = {}
        for name, tensor in params.items():
            array = handling_tensor_to_numpy(tensor)
            np_weights[name] = array
        new_np_weights = prune_model_weights(np_weights, pruning_ratio)
        weights_file = f"{metadata_folder}/weights_file_{batch_size}.npz"
        np.savez_compressed(weights_file, **new_np_weights)
        io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"
        x_input_np = {}
        for name, tensor in inputs_pt.items():
            array = handling_tensor_to_numpy(tensor)
            x_input_np[name] = array
        y_ait = torch.zeros([batch_size, seq_len, hidden_size], dtype=torch.float32, device="cpu")
        y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
        io_data = {"x_input": x_input_np, "y_output": y_output_np}
        np.savez_compressed(io_file, **io_data)
        transfer_folder(metadata_folder, TARGET_USER, TARGET_IP, target_dir)
        # remote_run_program_send_back_result(target_dir, "static/test_correctness_on_riscv.py", model_name, batch_size)
        # output_file = f"output_file_{model_name}_{batch_size}.npz"
        # output_np = np.load(output_file, allow_pickle=True)
        # outputs = torch.from_numpy(output_np["y_output"])
    # print(f"Logits: {outputs[0]}")
    # if verify:
    #     pt_outputs = pt_model.bert(**inputs_pt)
    #     torch.allclose(outputs[0], pt_outputs.last_hidden_state, 1e-1, 1e-1)
    #     print("Verification done!")


@click.command()
@click.option(
    "--prompt",
    type=str,
    default="The quick brown fox jumps over the lazy dog.",
    help="The prompt to give BERT.",
)
@click.option(
    "--activation",
    type=str,
    default="gelu",
    help="Activation function applied on BERT, currently only support gelu and fast_gelu",
)
@click.option(
    "--graph_mode",
    type=bool,
    default=True,
    help="Use CUDA graph or not. (hipGraph is not supported yet)",
)
@click.option(
    "--use_fp16_acc",
    type=bool,
    default=True,
    help="Use fp16 accumulation or not (TensorRT is using fp16_acc)",
)
@click.option(
    "--verify",
    type=bool,
    default=True,
    help="Verify AIT outputs against PT",
)
def run_demo(
    prompt: str,
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    verify: bool,
):
    run_model(prompt, activation, graph_mode, use_fp16_acc, verify)


if __name__ == "__main__":
    torch.manual_seed(4896)
    run_demo()
