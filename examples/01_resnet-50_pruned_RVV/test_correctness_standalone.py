import unittest

import torch
import numpy as np
golden_file = "y_pt.npz"
test_file = "output_file_pruned_cnhw_pruned_resnet50_1.npz"
golden = np.load(golden_file, allow_pickle=True)
y_gold = golden["y"]

test = np.load(test_file, allow_pickle=True)
y_test = test["y_output"]
torch.testing.assert_close(
    torch.from_numpy(y_test.reshape([1, 1000])), torch.from_numpy(y_gold.reshape([1, 1000])), rtol=1.1e-2, atol=1.1e-2
)
print("my answer:")
for i in torch.from_numpy(y_test.reshape([1, 1000])):
    print(f"{i} ")

# print("Test passed: y_test is close to y_gold!")