# AI_template_RVV_backend
This is a project forked from [AITemplate](https://github.com/facebookincubator/AITemplate).
## My Contributions
- **Implemented the paper: [Efficient Column-Wise N:M Pruning on RISC-V CPU](https://arxiv.org/abs/2507.17301)**
- Added a CPU backend, which was not previously supported
- Developed custom operations including:
  1. Sparse 2D convolution operators in CNHW layout (generate microkernel functions with different tile sizes and LMUL, and integrated them with functions defined in `static/cpu/include/rvv_utils.h`)
  2. Dense 2D convolution operators in CNHW layout (generate code that utilizes custom XNNPACK neural network operators implemented by myself)
- Enhanced profiling mechanisms to select optimal tile size and RISC-V Vector Length Multiplier (LMUL)
- Extended AITemplate to support remote compilation and code execution on RISC-V devices
## Setup
1. **Open** `python/aitemplate/utils/remote_send_receive_files.py`, and set the following variables: 
  - `TARGET_USER`
  - `TARGET_IP`
  - `REMOTE_PROFILE_DIR`
  - `RUN_DIR` 
  > ⚠️ **Note:** You must manually create the `REMOTE_PROFILE_DIR` and `RUN_DIR` directories on your remote device before proceeding.
2. **Send the folder** : `python/aitemplate/utils/static/` to the `RUN_DIR` directory on your remote device.
3. **Build 3rdParty** `3rdparty/XNNPACK_RVV` **first**
4. **After the build completes,** edit `python/aitemplate/backend/rvv/target_def.py` so that `xnnpack_path` points to your freshly built XNNPACK library.
5. **Warning:** the bare-metal cross-compiler `riscv64-unknown-elf-gcc` ships without `libpthread`, so multi-threading is unavailable; actual thread counts therefore depend on the device at run time. Consequently, AI_template_RVV_backend compiles and runs the program directly on the device.
6. **Build AITemplate** : Please note that Python 3.11.10 runs without any problems; newer Python versions may have compatibility issues with dependent packages.
  - When cloning the code, please use the following command to also clone the submodules:
    ```
    git clone --recursive https://github.com/facebookincubator/AITemplate
    ```
  - build AITemplate:
    ```
    cd python
    python setup.py bdist_wheel
    pip install dist/*.whl --force-reinstall
    ```
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

## License

AITemplate is licensed under the [Apache 2.0 License](https://github.com/facebookincubator/AITemplate/blob/main/LICENSE).
