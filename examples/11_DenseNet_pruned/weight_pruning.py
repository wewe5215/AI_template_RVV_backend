import numpy as np
from aitemplate.utils.fetch_Lmul_TileSize import fetch_lmul_and_tile
import math
vlen = 256
def f32_data_pruning_column_wise_with_ratio(weight, nr, mr, pruning_ratio):
    """
    Performs column-wise pruning on a 2D weight array using a given pruning ratio.
    
    Parameters:
      weight: a 2D numpy array of shape (output_channel, input_channel)
      nr: an integer multiplier for the recorded indices
      mr: block size (number of rows per block)
      pruning_ratio: fraction of columns to prune (e.g., 0.5 means prune bottom 50% columns)
                     
    Returns:
      pruned_weight: a 1D numpy array containing the pruned weights (row-major order)
      indices: a 1D numpy array (dtype uint16) with the recorded column indices
    """
    output_channel, input_channel = weight.shape
    pruned_weight = []   # List to store selected weight elements.
    indices = []         # List to store selected column indices (for the first row in each block).

    for i in range(0, output_channel, mr):
        end_offset = min(mr, output_channel - i)
        block = weight[i:i+end_offset, :]
        accumulator = np.sum(np.abs(block), axis=0)
        keep_count = int(np.ceil((1 - pruning_ratio) * input_channel))
        if np.all(accumulator == accumulator[0]):
            for j in range(end_offset):
                for k in range(keep_count):
                    pruned_weight.append(block[j, k])
                    if j == 0:
                        indices.append(k)
        else:
            threshold = np.percentile(accumulator, pruning_ratio * 100)
            for j in range(end_offset):
                for k in range(input_channel):
                    if input_channel % 2 == 0:
                        select = accumulator[k] >= threshold
                    else:
                        select = accumulator[k] > threshold
                    if select:
                        pruned_weight.append(block[j, k])
                        if j == 0:
                            indices.append(k)
    
    pruned_weight = np.array(pruned_weight, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint16)
    return pruned_weight, indices

def prune_model_weights(np_weights, pruning_ratio, model_name):
    """
    Processes a dictionary of model weights (including both kernels and biases). For each weight
    kernel (key containing 'weight' and 'conv'), it prunes the weight column-wise according to the given ratio.
    For weights with dimension 4 (assumed to be of shape
    (output_channel, kernel_height, kernel_width, input_channel)),
    it reshapes them to 2D with shape (output_channel, kernel_height * kernel_width * input_channel).
    The corresponding bias is retained.
    
    Parameters:
      np_weights: dict, keys are layer names and values are numpy arrays (weights or biases)
      mr: block size (number of rows per block) for the pruning routine.
      pruning_ratio: fraction of columns to prune (e.g., 0.5 means prune bottom 50% columns).
      
    Returns:
      new_model: dict, containing:
          - For each weight key: new entries for "layer_weight_pruned" and "layer_weight_indice"
          - For each bias key: the bias is retained unmodified.
    """
    weight_to_lmul, weight_to_tile_size = fetch_lmul_and_tile(f"profile_summary_{model_name}")
    new_model = {}
    for key, value in np_weights.items():
        if key not in weight_to_lmul and key not in weight_to_tile_size:
            print(f'key = {key}, value.ndim = {value.ndim} not being pruned')
            new_model[key] = value
        elif "weight" in key:
            if value.ndim != 2:
                if value.ndim == 4:
                    output_channel, kernel_height, kernel_width, input_channel = value.shape
                    weight_2d = value.reshape(output_channel, kernel_height * kernel_width * input_channel)
                else:
                    raise ValueError(f"Unsupported weight dimension {value.ndim} for key {key}")
            else:
                weight_2d = value
            lmul = int(weight_to_lmul[key])
            nr = lmul * (vlen / 32)  # 32 for float32
            mr = int(weight_to_tile_size[key])
            pruned_weight, indices = f32_data_pruning_column_wise_with_ratio(weight_2d, nr, mr, pruning_ratio)
            new_model[key] = pruned_weight
            new_model[key + "_indice"] = indices

            bias_key = key.replace("weight", "bias")
            if bias_key in np_weights:
                new_model[bias_key] = np_weights[bias_key]
    return new_model

if __name__ == "__main__":
    np_weights = {
        "layer2_0_conv3_weight": np.random.randn(8, 3, 3, 6).astype(np.float32),  # Shape: (8, 3, 3, 6)
        "layer2_0_conv3_bias": np.random.randn(8).astype(np.float32),
        # You can also include other layers...
    }
    pruning_ratio = 0.75  # For example, prune 75% of the columns.
    new_model = prune_model_weights(np_weights, pruning_ratio, "cnhw_pruned_50_densenet121_1")
