import unittest

import torch
import numpy as np
golden_file = "output_file_resnet50_1.npz"
test_file = "output_file_cnhw_resnet50_1.npz"
golden = np.load(golden_file, allow_pickle=True)
y_gold = golden["y_output"]

test = np.load(test_file, allow_pickle=True)
y_test = test["y_output"]
torch.testing.assert_close(
    torch.from_numpy(y_test.reshape([1, 1000])), torch.from_numpy(y_gold.reshape([1, 1000])), rtol=1e-1, atol=1e-1
)

print("Test passed: y_test is close to y_gold!")