# AI_template_RVV_backend
**Implementation of the paper: [Efficient Column-Wise N:M Pruning on RISC-V CPU](https://arxiv.org/abs/2507.17301)**
## Setup
1. **Open** `python/aitemplate/utils/remote_send_receive_files.py`, and set the following variables: 
  - `TARGET_USER`
  - `TARGET_IP`
  - `REMOTE_PROFILE_DIR`
  - `RUN_DIR` 
  > ⚠️ **Note:** You must manually create the `REMOTE_PROFILE_DIR` and `RUN_DIR` directories on your remote device before proceeding.
2. **Send the folder** : `python/aitemplate/utils/static/` to the `RUN_DIR` directory on your remote device.
3. **Build** `3rdparty/XNNPACK_RVV` **first**
4. **After the build completes,** edit `python/aitemplate/backend/rvv/target_def.py` so that `xnnpack_path` points to your freshly built XNNPACK library.
5. **Warning:** the bare-metal cross-compiler `riscv64-unknown-elf-gcc` ships without `libpthread`, so multi-threading is unavailable; actual thread counts therefore depend on the device at run time. Consequently, AI_template_RVV_backend compiles and runs the program directly on the device.
6. Python 3.11.10 runs without any problems; newer Python versions may have compatibility issues with dependent packages.
7. **Compiler requirement:** compile the generated C++ with **Clang ≥ 17.0.2**; older versions lack several RVV v0.12 intrinsics used by the backend.

## Important Notices
1. There will be four instances of remote access. Please check the content sent to the remote device before entering your password, for the sake of computer security. The text in parentheses indicates the file and location of the code that sends the remote access request:
  - Set up ssh_client (`python/aitemplate/utils/remote_send_receive_files.py`)
  - Send profile code to the remote device via scp (`python/aitemplate/backend/builder.py`, line 1038)
  - Send generated function code to the remote device via scp (`python/aitemplate/backend/builder.py`, line 1086)
  - Send metadata for code execution via scp (in each example folder’s `test_correctness.py` and `benchmark_ait_rvv.py`)
2. If you have any questions, feel free to open an issue. I will respond as soon as possible.
3. Currently, the CPU backend only supports f32. Support for f16 will be added in the future.

## Steps for Replicating the End-to-End Experiment from Our Paper
1. **Complete the Setup**
  - Make sure you have followed all the steps in the [Setup](#setup) section.
2. Navigate to the Example Folder and choose a folder corresponding to the model you want to evaluate::
  - `example/01_resnet-50_pruned_RVV` -> ResNet 18, 34, 50, 101, 152
  - `example/11_DenseNet_pruned` -> for DenseNet121
  - `example/12_MobileNet_pruned` -> for MobileNet-V2
3. Run the Benchmark Script: 
  - Execute the following script with your desired batch size:
  - `benchmark_ait_rvv.py --batch-size {batch_size you want}`
  - This will generate a profile summary and the benchmark result.
4. Retrain the Pruned Model:
  - Use the profile summary to guide retraining of the pruned model.
  - For ResNet models, retraining code is provided in: `example/01_resnet-50_pruned_RVV/retrain_code_resnet`
  - For DenseNet121 models, retraining code is provided in: `example/11_DenseNet_pruned/densenet121_re_train_column_wise_pruning.py`
  - Detailed training recipes and hyperparameters are described in the *Performance Evaluation* Section of our paper.
5. Other Notice: 
  - If you want to use the CPU backend, set the `IS_CPU_BACKEND` flag before compiling or running your model:
    ```python
      import importlib
      dt = importlib.import_module("aitemplate.testing.detect_target")
      dt.IS_CPU_BACKEND = True
    ```
  - **NHWC Layout Support**: The dense CPU backend also supports the NHWC data layout. For the models discussed in the paper, this may result in generated code that calls low-level XNNPACK operators. These operators are compatible with various hardware backends.
  - **Remote Compilation**: To enable remote compilation and execution, pass `remote_compile=True` to the `compile_model`,function. Otherwise, it defaults to `False`
    ```python
    module = compile_model(y, target, "./tmp", model_name, remote_compile=True)
    ```
# AITemplate

[![License](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE) |
[![Documentation](https://github.com/facebookincubator/AITemplate/actions/workflows/docs.yaml/badge.svg)](https://facebookincubator.github.io/AITemplate) |
[![CircleCI](https://circleci.com/gh/facebookincubator/AITemplate.svg?style=svg)](https://app.circleci.com/pipelines/github/facebookincubator/AITemplate)
[![Deploy docs to Pages](https://github.com/facebookincubator/AITemplate/actions/workflows/pages.yaml/badge.svg)](https://github.com/facebookincubator/AITemplate/actions/workflows/pages.yaml)


AITemplate (AIT) is a Python framework that transforms deep neural networks into CUDA (NVIDIA GPU) / HIP (AMD GPU) C++ code for lightning-fast inference serving. AITemplate highlights include:

- High performance: close to roofline fp16 TensorCore (NVIDIA GPU) / MatrixCore (AMD GPU) performance on major models, including ResNet, MaskRCNN, BERT, VisionTransformer, Stable Diffusion, etc.

- Unified, open, and flexible. Seamless fp16 deep neural network models for NVIDIA GPU or AMD GPU. Fully open source, Lego-style easily extendable high-performance primitives for new model support. Supports a significantly more comprehensive range of fusions than existing solutions for both GPU platforms.


## More about AITemplate

### Excellent Backward Capability

AITemplate doesn't depend on third-party libraries or runtimes, such as cuBLAS, cuDNN, rocBLAS, MIOpen, TensorRT, MIGraphX, etc. Each model is compiled into a self-contained portable binary, which can be used on any software environment with the same hardware.

### Horizontal Fusion

AITemplate provides unique advanced horizontal fusion. AITemplate can fuse parallel GEMM, LayerNorm, and other operators with different input shapes into a single GPU kernel.

### Vertical Fusion

AITemplate provides strong vertical fusion. AITemplate can fuse a large range of operations into TensorCore/MatrixCore operations, such as elementwise operations, reductions, and layout permutations. AITemplate also provides back-to-back style TensorCore / MatrixCore operation fusion.

### Memory Fusion

AITemplate provides innovative memory fusions. AITemplate can fuse GEMM, LayerNorm, and other operators, followed by memory operations such as concatenation, split, and slice into a single operator.

### Working w/wo PyTorch

The AITemplate-generated Python runtime can take PyTorch tensors as inputs and outputs without an extra copy. For environments without PyTorch, the AITemplate Python/C++ runtime is self-contained.

### Extensions without suffering

AITemplate provides a straightforward approach for making an extension in codegen. To add a new operator or a new fused kernel into AITemplate, most of the time one only needs to add two Python files: one for a graph node definition and another for the backend codegen. The CUDA/HIP kernel in a text header file can be directly utilized in the codegen.


## FX2AIT

FX2AIT is a Python-based tool that converts PyTorch models into AITemplate (AIT) engine for lightning-fast inference serving. Using FX2AIT's built-in AITLowerer, partial AIT acceleration can be achieved for models with unsupported operators in AITemplate.

Key features of FX2AIT include:

* Easy Conversion: FX2AIT requires only a PyTorch model and input for conversion, generating an "AITModule" output for inference serving.
* Expanded Support: AITemplate does not support all PyTorch operators. FX2AIT's AITLowerer offers a solution for partial AIT conversion for models with unsupported operators. Check the `fx2ait/fx2ait/example/03_lowering_split` for more information.

More info can be found from https://github.com/facebookincubator/AITemplate/tree/main/fx2ait.


## Installation

**Hardware requirements:**

  - **NVIDIA**: AIT is only tested on SM80+ GPUs (Ampere etc). Not all kernels work with old SM75/SM70 (T4/V100) GPUs.
  - **AMD**:  AIT is only tested on CDNA2 (MI-210/250) GPUs. There may be compiler issues for old CDNA1 (MI-100) GPUs.

### Clone the code

When cloning the code, please use the following command to also clone the submodules:
```
git clone --recursive https://github.com/facebookincubator/AITemplate
```

### Docker Image

We highly recommend using AITemplate with Docker to avoid accidentally using a wrong version of NVCC or HIPCC.

- CUDA: `./docker/build.sh cuda`
- ROCM: `DOCKER_BUILDKIT=1 ./docker/build.sh rocm`

This will build a docker image with tag `ait:latest`.

### From Source

The following command will create a Python wheel for AITemplate. Please ensure you have correct CUDA/ROCm compiler installed.

- CUDA: CUDA 11.6
- ROCm: We tested on ROCm 5.2.3 with a customized build HIPCC with the command in docker/Dockerfile.rocm#L87-L96

*Incorrect compiler will lead performance regression.*

**Please check all submodules are cloned correctly before go to next step.**

```
cd python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```

## Getting Started

Check out the [AITemplate Documentation](https://facebookincubator.github.io/AITemplate) for API reference.

There are a few tutorials for onboarding:

- 01: [How to inference a PyTorch model with AIT](https://facebookincubator.github.io/AITemplate/tutorial/how_to_infer_pt.html)
- 02: [How to add an op to AIT codegen](https://facebookincubator.github.io/AITemplate/tutorial/how_to_add_op.html)
- 03: [How to visualize AIT's optimization](https://facebookincubator.github.io/AITemplate/tutorial/how_to_visualize.html)


## Examples & Performance

AITemplate provides the following model templates & reference performance data on A100/MI-250:

- [01_ResNet-50](examples/01_resnet-50/) with PyTorch Image Models (TIMM)
- [02_MaskRCNN-FPN](examples/02_detectron2/) with Detectron2
- [03_BERT](examples/03_bert/) with Hugging Face Transformer
- [04_Vision Transformer](examples/04_vit/) with PyTorch Image Models (TIMM)
- [05_Stable Diffusion](examples/05_stable_diffusion/) with Hugging Face Diffusers


## Release

All current development updates can be seen in the AITemplate repository. Releases are not on a set schedule and will only be tagged for significant feature releases.

Mid-term plan:

- Better dynamic shape support: Focus on the dynamic sequence in Transformers. Add symbolic shape support.
- More automatic graph passes: Relief manual rewrite models to obtain the best performance.
- Quantization: fp8/int8/int4.
- Sparsity pruning for Gemm.
- PT2 integration: Aten2AIT is under active development.

Long-term plan:

- Automatic ONNX, Open-XLA and other format model conversion.
- Composable Kernel CPU extension on AVX2/AVX-512 for AMD Epyc CPU.


## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute to the project.


## The Team

AITemplate is currently maintained by Meta engineers: [Ying Zhang](https://github.com/ipiszy), [Yang Chen](https://github.com/chenyang78), [Terry Chen](https://github.com/terrychenism), [Mu-Chu Lee](https://github.com/muchulee8), [Max Podkorytov](https://github.com/tenpercent), [Adnan Akhundov](https://github.com/aakhundov).

AITemplate is co-created by Meta engineers: [Bing Xu](https://github.com/antinucleon), [Ying Zhang](https://github.com/ipiszy), [Hao Lu](https://github.com/hlu1), [Yang Chen](https://github.com/chenyang78), and [Terry Chen](https://github.com/terrychenism), with major contributions coming from other talented engineers. A non-exhaustive list to mention is Mike Iovine, Mu-Chu Lee, Scott Wolchok, Oleg Khabinov, Shirong Wu, Huamin Li, Hui Guo, Zhijing Li, Max Podkorytov. We also want to thank Andrew Tulloch, Yinghai Lu, Lu Fang for the valuable discussions.

FX2AIT and Aten2AIT are co-created and maintained by Meta engineers: [Wei Wei](https://github.com/frank-wei), [Shirong Wu](https://github.com/wushirong) and [Zhijing Li](https://github.com/tissue3).


## Acknowledgements

AITemplate team works closely with NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass) Team (led by Andrew Kerr, Haicheng Wu) and AMD [Composable Kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) Team (led by Chao Liu, Jing Zhang). We co-designed many advanced GPU optimizations specialized for each platform, and nothing is possible without our close collaboration.


## License

AITemplate is licensed under the [Apache 2.0 License](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE).
