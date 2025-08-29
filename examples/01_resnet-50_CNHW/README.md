# ResNet-50

In this example, we will demo how to use AITemplate for inference on the ResNet-50 model from PyTorch Image Models (TIMM).

We will demo two usages:
* Using AIT to accelerate PyTorch inference
* Using AIT standalone under RVV environment

## Code structure
```
modeling
    resnet.py              # ResNet definition using AIT's frontend API
weight_utils.py            # Utils to convert TIMM R-50 weights to AIT
infer_with_torch.py        # Example to accelerate PyTorch, and seamlessly use with other PyTorch code
infer_with_numpy.py        # Dump TIMM weights to Numpy and use AIT & Numpy without 3rdparties
benchmark_pt.py            # Benchmark code for PyTorch
benchmark_ait_rvv.py       # Benchmark code for AIT & XNNPACK backend through \
                                sending compiled object files, python scripts to RISC-V board \
                                and retrive the benchmark result from RISC-V board
```
### Note for Performance Results
The correctness of the CPU-backend code generation has been verified with the 01_resnet-50 benchmark. However, if you validate remotely compiled and executed code against results from a different device—for example, comparing PyTorch on an M2 Mac with AIT running on a RISC-V Banana Pi—you may see slight differences in the ResNet-50 outputs. If you instead generate and run the CPU code locally and compare it with the PyTorch result on the same machine, there should be no mismatched elements.