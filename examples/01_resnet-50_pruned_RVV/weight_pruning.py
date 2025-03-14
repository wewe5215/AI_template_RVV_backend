import numpy as np
from group_op_and_lmul import fetch_lmul_for_op

# Fetch LMUL values for each op (used later for index scaling)
weight_to_lmul = fetch_lmul_for_op()
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
    # print(f'weight_shape = {weight.shape}')
    pruned_weight = []   # List to store selected weight elements.
    indices = []         # List to store selected column indices (for the first row in each block).
    
    # Process the weight array in blocks of mr rows.
    print(f'output_channel = {output_channel}')
    for i in range(0, output_channel, mr):
        end_offset = min(mr, output_channel - i)
        block = weight[i:i+end_offset, :]  # Block shape: (end_offset, input_channel)
        
        # Compute accumulator: sum of absolute values for each column in the block.
        accumulator = np.sum(np.abs(block), axis=0)
        
        # Determine threshold so that the bottom pruning_ratio fraction of columns are pruned.
        # (That is, keep columns with accumulator values in the top (1-pruning_ratio)*100 percentile.)
        threshold = np.percentile(accumulator, pruning_ratio * 100)
        
        # For each element in the block, select the element if its column's accumulator passes the threshold.
        for j in range(end_offset):
            for k in range(input_channel):
                # Use '>=' for even number of columns, '>' for odd (following the C-code logic).
                if input_channel % 2 == 0:
                    select = accumulator[k] >= threshold
                else:
                    select = accumulator[k] > threshold
                
                if select:
                    pruned_weight.append(block[j, k])
                    # For the first row in the block, record the column index (multiplied by nr).
                    if j == 0:
                        indices.append(k * nr)
                        
    pruned_weight = np.array(pruned_weight, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint16)
    return pruned_weight, indices

def prune_model_weights(np_weights, pruning_ratio):
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
    new_model = {}
    for key, value in np_weights.items():
        # Process only keys that are weight kernels (not biases) and include "conv"
        if "weight" in key and "stem" not in key and "fc" not in key:
            # print(f'value.ndim = {value.ndim}, key = {key}')
            # If the weight is not 2D, reshape it appropriately.
            if value.ndim != 2:
                if value.ndim == 4:
                    # Assume weight has shape (output_channel, kernel_height, kernel_width, input_channel)
                    output_channel, kernel_height, kernel_width, input_channel = value.shape
                    # Reshape to 2D: each row corresponds to an output channel,
                    # and each column is a flattened kernel.
                    weight_2d = value.reshape(output_channel, kernel_height * kernel_width * input_channel)
                    # print(f"Reshaped from {value.shape} to {weight_2d.shape}")
                else:
                    raise ValueError(f"Unsupported weight dimension {value.ndim} for key {key}")
            else:
                weight_2d = value

            # Calculate nr based on your mapping (using weight_to_lmul) and vlen.
            nr = weight_to_lmul[key] * (vlen / 32)  # 32 for float32
            print(f'key = {key} being pruned with lmul = {weight_to_lmul[key]}, value.ndim = {value.ndim}')
            if weight_to_lmul[key] == 1 or weight_to_lmul[key] == 4:
                mr = 7
            elif weight_to_lmul[key] == 2:
                mr = 8
            else:
                mr = 3
            # Apply column-wise pruning using the provided ratio.
            pruned_weight, indices = f32_data_pruning_column_wise_with_ratio(weight_2d, nr, mr, pruning_ratio)
            new_model[key] = pruned_weight
            new_model[key + "_indice"] = indices
            part1 = (output_channel + mr - 1) / mr
            part2 = (kernel_height * kernel_width * input_channel) // 2
            print(f'indice name = {key}_indice, {{{part1}, {part2}}}, indice shape = {indices.shape}')
            # print(indices)
            # Also retain the corresponding bias if present.
            bias_key = key.replace("weight", "bias")
            if bias_key in np_weights:
                new_model[bias_key] = np_weights[bias_key]
        elif "bias" in key:
            # For biases that do not have a corresponding weight (or haven't been processed yet), simply copy them.
            bias_corresponding_weight = key.replace("bias", "weight")
            if bias_corresponding_weight not in np_weights:
                print(f'key = {key}, value.ndim = {value.ndim} not being pruned(elif bias not in key)')
                new_model[key] = value
        else:
            print(f'key = {key}, value.ndim = {value.ndim} not being pruned')
            new_model[key] = value
    return new_model

# Example usage:
if __name__ == "__main__":
    # Example np_weights dictionary.
    # For demonstration, let's create a 4D weight (conv) and a bias.
    np_weights = {
        "stem_conv1_weight": np.random.randn(8, 3, 3, 6).astype(np.float32),  # Shape: (8, 3, 3, 6)
        "stem_conv1_bias": np.random.randn(8).astype(np.float32),
        # You can also include other layers...
    }
    
    mr = 3            # Block size for pruning.
    pruning_ratio = 0.75  # For example, prune 75% of the columns.
    
    new_model = prune_model_weights(np_weights, pruning_ratio)
    
    # Display the new dictionary entries.
    for k, v in new_model.items():
        print(f"{k}:\n {v}\n")
