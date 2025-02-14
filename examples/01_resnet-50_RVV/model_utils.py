import os
import platform
import numpy as np
import struct
_DTYPE_TO_ENUM = {
    "float16": 1,
    "float32": 2,
    "float": 2,
    "int": 3,
    "int32": 3,
    "int64": 4,
    "bool": 5,
    "bfloat16": 6,
}

def dtype_str_to_enum(dtype: str) -> int:
    """Returns the AITemplateDtype enum value (defined in model_interface.h) of
    the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        the AITemplateDtype enum value.
    """
    if dtype not in _DTYPE_TO_ENUM:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(_DTYPE_TO_ENUM.keys())}"
        )
    return _DTYPE_TO_ENUM[dtype]

def types_mapping():

    # Define a dummy bfloat16 type for mapping purposes.
    class bfloat16:
        def __repr__(self):
            return "bfloat16"

    yield (np.float16, "float16")
    yield (bfloat16, "bfloat16")
    yield (np.float32, "float32")
    yield (np.int32, "int32")
    yield (np.int64, "int64")
    yield (bool, "bool")

_DTYPE2BYTE = {
    "bool": 1,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
    "bfloat16": 2,
}

def torch_dtype_to_string(dtype):
    for torch_dtype, ait_dtype in types_mapping():
        if dtype == torch_dtype:
            return ait_dtype
    raise ValueError(
        f"Got unsupported input dtype {dtype}! "
        f"Supported dtypes are: {list(types_mapping())}"
    )

def normalize_dtype(dtype: str) -> str:
    """Returns a normalized dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        normalized dtype str.
    """
    if dtype == "int":
        return "int32"
    if dtype == "float":
        return "float32"
    return dtype

def get_dtype_size(dtype: str) -> int:
    """Returns size (in bytes) of the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        Size (in bytes) of this dtype.
    """

    if dtype not in _DTYPE2BYTE:
        raise KeyError(f"Unknown dtype: {dtype}. Expected one of {_DTYPE2BYTE.keys()}")
    return _DTYPE2BYTE[dtype]

def write_tensor_binary(tensor: "torch.Tensor", file_handle) -> None:
    tensor = tensor.detach().cpu().contiguous()
    endianness = "@"  # system endianness
    dtype_str = normalize_dtype(torch_dtype_to_string(tensor.dtype))
    dtype_int = dtype_str_to_enum(dtype_str)
    sizeof_dtype = get_dtype_size(dtype_str)
    num_dims = len(tensor.shape)
    file_handle.write(struct.pack(endianness + "I", dtype_int))  # unsigned int
    file_handle.write(struct.pack(endianness + "I", sizeof_dtype))  # unsigned int
    file_handle.write(struct.pack(endianness + "I", num_dims))  # unsigned int
    total_size = sizeof_dtype
    for dim in tensor.shape:
        file_handle.write(struct.pack(endianness + "N", dim))  # size_t
        total_size *= dim
    file_handle.write(struct.pack(endianness + "N", total_size))  # size_t
    bytedata = tensor.numpy().tobytes()
    # just as a safety check
    if len(bytedata) != total_size:
        raise RuntimeError("Tensor has wrong number of bytes!")
    file_handle.write(bytedata)
def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return os.name == "nt"

def is_macos():
    return platform.system() == "Darwin"
