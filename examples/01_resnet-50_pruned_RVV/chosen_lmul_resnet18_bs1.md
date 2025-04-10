| Parameter                    | operator                | Dimension      | lmul |
| ---------------------------- | ----------------------- | -------------- | ---- |
| stem_conv1_weight            | conv2d_bias_relu_0      | 64, 7, 7, 3    | 2    |
| layer1_0_conv1_weight        | conv2d_bias_relu_2      | 64, 3, 3, 64   | 2    |
| layer1_0_conv2_weight        | conv2d_bias_add_relu_3  | 64, 3, 3, 64   | 2    |
| layer1_1_conv1_weight        | conv2d_bias_relu_2      | 64, 3, 3, 64   | 2    |
| layer1_1_conv2_weight        | conv2d_bias_add_relu_3  | 64, 3, 3, 64   | 2    |
| layer2_0_conv1_weight        | conv2d_bias_relu_6      | 128, 3, 3, 64  | 4    |
| layer2_0_conv2_weight        | conv2d_bias_add_relu_8  | 128, 3, 3, 128 | 2    |
| layer2_0_downsample_0_weight | conv2d_bias_7           | 128, 1, 1, 64  | 8    |
| layer2_1_conv1_weight        | conv2d_bias_relu_9      | 128, 3, 3, 128 | 2    |
| layer2_1_conv2_weight        | conv2d_bias_add_relu_8  | 128, 3, 3, 128 | 2    |
| layer3_0_conv1_weight        | conv2d_bias_relu_11     | 256, 3, 3, 128 | 4    |
| layer3_0_conv2_weight        | conv2d_bias_add_relu_13 | 256, 3, 3, 256 | 2    |
| layer3_0_downsample_0_weight | conv2d_bias_12          | 256, 1, 1, 128 | 2    |
| layer3_1_conv1_weight        | conv2d_bias_relu_14     | 256, 3, 3, 256 | 2    |
| layer3_1_conv2_weight        | conv2d_bias_add_relu_13 | 256, 3, 3, 256 | 2    |
| layer4_0_conv1_weight        | conv2d_bias_relu_16     | 512, 3, 3, 256 | 2    |
| layer4_0_conv2_weight        | conv2d_bias_add_relu_18 | 512, 3, 3, 512 | 2    |
| layer4_0_downsample_0_weight | conv2d_bias_17          | 512, 1, 1, 256 | 2    |
| layer4_1_conv1_weight        | conv2d_bias_relu_19     | 512, 3, 3, 512 | 2    |
| layer4_1_conv2_weight        | conv2d_bias_add_relu_18 | 512, 3, 3, 512 | 2    |