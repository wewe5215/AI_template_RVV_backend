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
dt = importlib.import_module("aitemplate.testing.detect_target")
dt.IS_CPU_BACKEND = True
dt = importlib.import_module("aitemplate.compiler.compiler")
dt.IS_REMOTE_COMPILE = True
from benchmark_ait import compile_module
from modeling.torch_model import BertBaseUncased as BertPt
import numpy as np
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
    model_name = f"BERT_{activation}_{batch_size}_{seq_len}"
    metadata_folder = f"metadata_{model_name}_{batch_size}"
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder, exist_ok=True)
        print(f"Created directory: {metadata_folder}")
    mod = compile_module(
        batch_size, seq_len, hidden_size, activation, use_fp16_acc, False, pt_model, \
        is_remote_compile=dt.IS_REMOTE_COMPILE, metadata_folder=metadata_folder
    )

    if dt.IS_REMOTE_COMPILE == False:
        outputs = [torch.empty(mod.get_output_maximum_shape(0))]
        mod.run_with_tensors(inputs_pt, outputs, graph_mode=graph_mode)
    else:
        io_file = f"{metadata_folder}/io_tensors_{batch_size}.npz"
        x_input_np = {}
        for name, tensor in inputs_pt.items():
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
        y_ait = torch.zeros([batch_size, seq_len, hidden_size], dtype=torch.float32, device="cpu")
        y_output_np = y_ait.cpu().detach().numpy().astype(np.float32)
        io_data = {"x_input": x_input_np, "y_output": y_output_np}
        np.savez_compressed(io_file, **io_data)
        transfer_folder(metadata_folder, TARGET_USER, TARGET_IP, target_dir)
        remote_run_program_send_back_result(target_dir, "static/test_correctness_on_riscv.py", model_name, batch_size)
        output_file = f"output_file_{model_name}_{batch_size}.npz"
        output_np = np.load(output_file, allow_pickle=True)
        outputs = torch.from_numpy(output_np["y_output"])
    print(f"Logits: {outputs[0]}")
    if verify:
        pt_outputs = pt_model.bert(**inputs_pt)
        torch.allclose(outputs[0], pt_outputs.last_hidden_state, 1e-1, 1e-1)
        print("Verification done!")


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
