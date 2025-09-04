import torch
from torch.nn.utils.prune import BasePruningMethod


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, mask):
        mask.to(dtype=weight.dtype)
        ctx.mask = mask
        return weight*mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class MyPruningMethod(BasePruningMethod):
    PRUNING_TYPE = "global"

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, module, inputs):
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned"  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = STEFunction.apply(orig, mask)
        setattr(module, self._tensor_name, pruned_tensor)

    def update_mask(self, module, mask):
        """
        if torch.equal(self.mask, mask) is False:
            print(torch.equal(self.mask, mask))
        """

        mask = self.default_mask * mask # default_mask should be all ones.
        self.mask = mask
        setattr(module, self._tensor_name + "_mask", mask)

    def apply_mask(self, module):
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned"  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    def compute_mask(self, t, default_mask):
        assert default_mask.shape == self.mask.shape
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        self.default_mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name, mask):
        return super().apply(module, name, mask=mask)

