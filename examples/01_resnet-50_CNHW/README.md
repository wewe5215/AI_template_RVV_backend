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
static
    FakeTorchTensor.py     # Fake torch tensor (torch library isn't supported on device)
    model.py               # Python bindings to the AIT runtime on device
    model_utils.py         # utils used in model.py
    remote_send_receive_files.py # functions for remote sending/retriving the files
    run_benchmark_on_riscv.py
    test_correctness_on_riscv.py
```



## Reference Speed vs PyTorch Eager



### Note for Performance Results


