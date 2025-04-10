import numpy as np
class FakeTorchTensor:
    def __init__(self, array):
        """
        Initialize with a NumPy array.
        """
        self.array = array
        self.dtype = array.dtype
        self.shape = array.shape
        self.strides = array.strides
        self.ndim = array.ndim
        # Expose the __array_interface__ for interoperability with other libraries.
        self.__array_interface__ = array.__array_interface__

    def __getitem__(self, index):
        """
        Enable subscript/slicing operations.
        Delegates to the underlying NumPy array and wraps the result in a FakeTorchTensor if it's an array.
        """
        result = self.array[index]
        if isinstance(result, np.ndarray):
            return FakeTorchTensor(result)
        else:
            return result

    def cpu(self):
        """
        Mimic the torch.Tensor.cpu() method.
        Since this is a CPU-only tensor, return self.
        """
        return self

    def detach(self):
        """
        Mimic the torch.Tensor.detach() method.
        Since this fake tensor doesn't track gradients, simply return self.
        """
        return self

    def numpy(self):
        """
        Mimic the torch.Tensor.numpy() method.
        Returns the underlying NumPy array.
        """
        return self.array

    def reshape(self, *new_shape):
        """
        Mimic the torch.Tensor.reshape() method.
        Returns a new FakeTorchTensor with the array reshaped.
        """
        reshaped_array = self.array.reshape(*new_shape)
        return FakeTorchTensor(reshaped_array)

    def size(self):
        """
        Mimic the torch.Tensor.size() method.
        Returns the shape of the tensor as a tuple.
        """
        return self.array.shape

    def flatten(self):
        """
        Mimic the torch.Tensor.flatten() method.
        Returns a new FakeTorchTensor with a one-dimensional (flattened) version of the array.
        """
        flat_array = self.array.flatten()
        return FakeTorchTensor(flat_array)

    def data_ptr(self):
        """
        Mimic the torch.Tensor.data_ptr() method.
        Returns an integer representing the memory address of the data.
        """
        return self.array.ctypes.data

    @property
    def ptr(self):
        """
        Provide a property alias for data_ptr.
        """
        return self.data_ptr()

    def __repr__(self):
        return f"FakeTorchTensor(shape={self.shape}, dtype={self.dtype})"
