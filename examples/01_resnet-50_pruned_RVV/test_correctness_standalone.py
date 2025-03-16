import unittest

import torch
import numpy as np
golden_file = "revised_cnhw_pruned_75_resnet50_4_y_pt.npz"
test_file = "output_file_revised_cnhw_pruned_75_resnet50_4.npz"
golden = np.load(golden_file, allow_pickle=True)
y_gold = golden["y"]

test = np.load(test_file, allow_pickle=True)
y_test = test["y_output"]
batch_size = 4
tolerence = 0.012
torch.testing.assert_close(
    torch.from_numpy(y_test.reshape([batch_size, 1000])), torch.from_numpy(y_gold.reshape([batch_size, 1000])), rtol=tolerence, atol=tolerence
)
# print("my answer:")
# for i in torch.from_numpy(y_test.reshape([1, 1000])):
#     print(f"{i} ")

# print("Test passed: y_test is close to y_gold!")