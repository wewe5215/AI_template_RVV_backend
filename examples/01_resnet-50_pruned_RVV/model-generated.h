
#pragma once

#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "model.h"
#include "macros.h"
#include "jagged.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <math.h>
#include <iomanip>
#include <pthreadpool.h>
#include <thread>


void conv2d_bias_relu_0(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void max_pool2d_1(
  void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  pthreadpool*
);

void conv2d_bias_2(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_3(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_4(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_add_relu_5(
  void*,
  void*,
  void*,

  void*,
  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_6(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_12(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_13(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_14(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_add_relu_15(
  void*,
  void*,
  void*,

  void*,
  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_16(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_17(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_25(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_26(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_27(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_add_relu_28(
  void*,
  void*,
  void*,

  void*,
  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_29(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_30(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_44(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_45(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_46(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_add_relu_47(
  void*,
  void*,
  void*,

  void*,
  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_48(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void conv2d_bias_relu_49(
  void*,
  void*,
  void*,

  void*,

  uint8_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int,
  int,
  int,
  int,
  int,
  int,
  pthreadpool*
);

void avg_pool2d_54(
  void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  pthreadpool*
);

void gemm_rcr_bias_56(
  void*,
  void*,
  void*,
  void*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


    int64_t*,

    int64_t*,

    pthreadpool*
);

namespace ait {

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class Model : public ModelBase<Model> {


  public:
    Model(
        size_t blob_size,
        size_t workspace_size,
        size_t unique_workspace_size,
        size_t num_inputs,
        size_t num_outputs,
        size_t num_unbound_constants,
        uint8_t* constants,
        AITemplateAllocator& allocator)
        : ModelBase(
            blob_size,
            workspace_size,
            unique_workspace_size,
            num_inputs,
            num_outputs,
            num_unbound_constants,
            constants),
          threadpool_(pthreadpool_create(std::thread::hardware_concurrency()), pthreadpool_destroy) {
         constant_name_to_ptr_["stem_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&stem_conv1_weight));
     constant_name_to_ptr_["stem_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&stem_conv1_bias));
     constant_name_to_ptr_["layer1_0_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv1_weight));
     constant_name_to_ptr_["layer1_0_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv1_bias));
     constant_name_to_ptr_["layer1_0_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv2_weight));
     constant_name_to_ptr_["layer1_0_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv2_bias));
     constant_name_to_ptr_["layer1_0_downsample_0_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_downsample_0_weight));
     constant_name_to_ptr_["layer1_0_downsample_0_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_downsample_0_bias));
     constant_name_to_ptr_["layer1_0_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv3_weight));
     constant_name_to_ptr_["layer1_0_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_0_conv3_bias));
     constant_name_to_ptr_["layer1_1_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv1_weight));
     constant_name_to_ptr_["layer1_1_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv1_bias));
     constant_name_to_ptr_["layer1_1_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv2_weight));
     constant_name_to_ptr_["layer1_1_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv2_bias));
     constant_name_to_ptr_["layer1_1_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv3_weight));
     constant_name_to_ptr_["layer1_1_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_1_conv3_bias));
     constant_name_to_ptr_["layer1_2_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv1_weight));
     constant_name_to_ptr_["layer1_2_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv1_bias));
     constant_name_to_ptr_["layer1_2_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv2_weight));
     constant_name_to_ptr_["layer1_2_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv2_bias));
     constant_name_to_ptr_["layer1_2_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv3_weight));
     constant_name_to_ptr_["layer1_2_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer1_2_conv3_bias));
     constant_name_to_ptr_["layer2_0_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv1_weight));
     constant_name_to_ptr_["layer2_0_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv1_bias));
     constant_name_to_ptr_["layer2_0_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv2_weight));
     constant_name_to_ptr_["layer2_0_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv2_bias));
     constant_name_to_ptr_["layer2_0_downsample_0_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_downsample_0_weight));
     constant_name_to_ptr_["layer2_0_downsample_0_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_downsample_0_bias));
     constant_name_to_ptr_["layer2_0_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv3_weight));
     constant_name_to_ptr_["layer2_0_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_0_conv3_bias));
     constant_name_to_ptr_["layer2_1_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv1_weight));
     constant_name_to_ptr_["layer2_1_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv1_bias));
     constant_name_to_ptr_["layer2_1_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv2_weight));
     constant_name_to_ptr_["layer2_1_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv2_bias));
     constant_name_to_ptr_["layer2_1_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv3_weight));
     constant_name_to_ptr_["layer2_1_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_1_conv3_bias));
     constant_name_to_ptr_["layer2_2_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv1_weight));
     constant_name_to_ptr_["layer2_2_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv1_bias));
     constant_name_to_ptr_["layer2_2_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv2_weight));
     constant_name_to_ptr_["layer2_2_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv2_bias));
     constant_name_to_ptr_["layer2_2_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv3_weight));
     constant_name_to_ptr_["layer2_2_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_2_conv3_bias));
     constant_name_to_ptr_["layer2_3_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv1_weight));
     constant_name_to_ptr_["layer2_3_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv1_bias));
     constant_name_to_ptr_["layer2_3_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv2_weight));
     constant_name_to_ptr_["layer2_3_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv2_bias));
     constant_name_to_ptr_["layer2_3_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv3_weight));
     constant_name_to_ptr_["layer2_3_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer2_3_conv3_bias));
     constant_name_to_ptr_["layer3_0_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv1_weight));
     constant_name_to_ptr_["layer3_0_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv1_bias));
     constant_name_to_ptr_["layer3_0_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv2_weight));
     constant_name_to_ptr_["layer3_0_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv2_bias));
     constant_name_to_ptr_["layer3_0_downsample_0_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_downsample_0_weight));
     constant_name_to_ptr_["layer3_0_downsample_0_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_downsample_0_bias));
     constant_name_to_ptr_["layer3_0_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv3_weight));
     constant_name_to_ptr_["layer3_0_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_0_conv3_bias));
     constant_name_to_ptr_["layer3_1_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv1_weight));
     constant_name_to_ptr_["layer3_1_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv1_bias));
     constant_name_to_ptr_["layer3_1_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv2_weight));
     constant_name_to_ptr_["layer3_1_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv2_bias));
     constant_name_to_ptr_["layer3_1_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv3_weight));
     constant_name_to_ptr_["layer3_1_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_1_conv3_bias));
     constant_name_to_ptr_["layer3_2_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv1_weight));
     constant_name_to_ptr_["layer3_2_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv1_bias));
     constant_name_to_ptr_["layer3_2_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv2_weight));
     constant_name_to_ptr_["layer3_2_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv2_bias));
     constant_name_to_ptr_["layer3_2_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv3_weight));
     constant_name_to_ptr_["layer3_2_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_2_conv3_bias));
     constant_name_to_ptr_["layer3_3_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv1_weight));
     constant_name_to_ptr_["layer3_3_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv1_bias));
     constant_name_to_ptr_["layer3_3_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv2_weight));
     constant_name_to_ptr_["layer3_3_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv2_bias));
     constant_name_to_ptr_["layer3_3_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv3_weight));
     constant_name_to_ptr_["layer3_3_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_3_conv3_bias));
     constant_name_to_ptr_["layer3_4_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv1_weight));
     constant_name_to_ptr_["layer3_4_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv1_bias));
     constant_name_to_ptr_["layer3_4_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv2_weight));
     constant_name_to_ptr_["layer3_4_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv2_bias));
     constant_name_to_ptr_["layer3_4_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv3_weight));
     constant_name_to_ptr_["layer3_4_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_4_conv3_bias));
     constant_name_to_ptr_["layer3_5_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv1_weight));
     constant_name_to_ptr_["layer3_5_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv1_bias));
     constant_name_to_ptr_["layer3_5_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv2_weight));
     constant_name_to_ptr_["layer3_5_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv2_bias));
     constant_name_to_ptr_["layer3_5_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv3_weight));
     constant_name_to_ptr_["layer3_5_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer3_5_conv3_bias));
     constant_name_to_ptr_["layer4_0_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv1_weight));
     constant_name_to_ptr_["layer4_0_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv1_bias));
     constant_name_to_ptr_["layer4_0_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv2_weight));
     constant_name_to_ptr_["layer4_0_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv2_bias));
     constant_name_to_ptr_["layer4_0_downsample_0_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_downsample_0_weight));
     constant_name_to_ptr_["layer4_0_downsample_0_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_downsample_0_bias));
     constant_name_to_ptr_["layer4_0_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv3_weight));
     constant_name_to_ptr_["layer4_0_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_0_conv3_bias));
     constant_name_to_ptr_["layer4_1_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv1_weight));
     constant_name_to_ptr_["layer4_1_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv1_bias));
     constant_name_to_ptr_["layer4_1_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv2_weight));
     constant_name_to_ptr_["layer4_1_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv2_bias));
     constant_name_to_ptr_["layer4_1_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv3_weight));
     constant_name_to_ptr_["layer4_1_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_1_conv3_bias));
     constant_name_to_ptr_["layer4_2_conv1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv1_weight));
     constant_name_to_ptr_["layer4_2_conv1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv1_bias));
     constant_name_to_ptr_["layer4_2_conv2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv2_weight));
     constant_name_to_ptr_["layer4_2_conv2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv2_bias));
     constant_name_to_ptr_["layer4_2_conv3_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv3_weight));
     constant_name_to_ptr_["layer4_2_conv3_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&layer4_2_conv3_bias));
     constant_name_to_ptr_["fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&fc_weight));
     constant_name_to_ptr_["fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&fc_bias));
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
        conv2d_bias_relu_0_0 = reinterpret_cast<decltype(conv2d_bias_relu_0_0)>(blob_ptr + 0);
    max_pool2d_1_0 = reinterpret_cast<decltype(max_pool2d_1_0)>(blob_ptr + 3211264);
    conv2d_bias_2_0 = reinterpret_cast<decltype(conv2d_bias_2_0)>(blob_ptr + 0);
    conv2d_bias_relu_3_0 = reinterpret_cast<decltype(conv2d_bias_relu_3_0)>(blob_ptr + 4014080);
    conv2d_bias_relu_4_0 = reinterpret_cast<decltype(conv2d_bias_relu_4_0)>(blob_ptr + 6422528);
    conv2d_bias_add_relu_5_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_5_0)>(blob_ptr + 3211264);
    conv2d_bias_relu_6_0 = reinterpret_cast<decltype(conv2d_bias_relu_6_0)>(blob_ptr + 0);
    conv2d_bias_relu_7_0 = reinterpret_cast<decltype(conv2d_bias_relu_7_0)>(blob_ptr + 6422528);
    conv2d_bias_add_relu_8_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_8_0)>(blob_ptr + 0);
    conv2d_bias_relu_9_0 = reinterpret_cast<decltype(conv2d_bias_relu_9_0)>(blob_ptr + 3211264);
    conv2d_bias_relu_10_0 = reinterpret_cast<decltype(conv2d_bias_relu_10_0)>(blob_ptr + 6422528);
    conv2d_bias_add_relu_11_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_11_0)>(blob_ptr + 3211264);
    conv2d_bias_relu_12_0 = reinterpret_cast<decltype(conv2d_bias_relu_12_0)>(blob_ptr + 0);
    conv2d_bias_13_0 = reinterpret_cast<decltype(conv2d_bias_13_0)>(blob_ptr + 1605632);
    conv2d_bias_relu_14_0 = reinterpret_cast<decltype(conv2d_bias_relu_14_0)>(blob_ptr + 3211264);
    conv2d_bias_add_relu_15_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_15_0)>(blob_ptr + 0);
    conv2d_bias_relu_16_0 = reinterpret_cast<decltype(conv2d_bias_relu_16_0)>(blob_ptr + 1605632);
    conv2d_bias_relu_17_0 = reinterpret_cast<decltype(conv2d_bias_relu_17_0)>(blob_ptr + 3211264);
    conv2d_bias_add_relu_18_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_18_0)>(blob_ptr + 1605632);
    conv2d_bias_relu_19_0 = reinterpret_cast<decltype(conv2d_bias_relu_19_0)>(blob_ptr + 0);
    conv2d_bias_relu_20_0 = reinterpret_cast<decltype(conv2d_bias_relu_20_0)>(blob_ptr + 3211264);
    conv2d_bias_add_relu_21_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_21_0)>(blob_ptr + 0);
    conv2d_bias_relu_22_0 = reinterpret_cast<decltype(conv2d_bias_relu_22_0)>(blob_ptr + 1605632);
    conv2d_bias_relu_23_0 = reinterpret_cast<decltype(conv2d_bias_relu_23_0)>(blob_ptr + 3211264);
    conv2d_bias_add_relu_24_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_24_0)>(blob_ptr + 1605632);
    conv2d_bias_relu_25_0 = reinterpret_cast<decltype(conv2d_bias_relu_25_0)>(blob_ptr + 0);
    conv2d_bias_26_0 = reinterpret_cast<decltype(conv2d_bias_26_0)>(blob_ptr + 802816);
    conv2d_bias_relu_27_0 = reinterpret_cast<decltype(conv2d_bias_relu_27_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_28_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_28_0)>(blob_ptr + 0);
    conv2d_bias_relu_29_0 = reinterpret_cast<decltype(conv2d_bias_relu_29_0)>(blob_ptr + 802816);
    conv2d_bias_relu_30_0 = reinterpret_cast<decltype(conv2d_bias_relu_30_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_31_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_31_0)>(blob_ptr + 802816);
    conv2d_bias_relu_32_0 = reinterpret_cast<decltype(conv2d_bias_relu_32_0)>(blob_ptr + 0);
    conv2d_bias_relu_33_0 = reinterpret_cast<decltype(conv2d_bias_relu_33_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_34_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_34_0)>(blob_ptr + 0);
    conv2d_bias_relu_35_0 = reinterpret_cast<decltype(conv2d_bias_relu_35_0)>(blob_ptr + 802816);
    conv2d_bias_relu_36_0 = reinterpret_cast<decltype(conv2d_bias_relu_36_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_37_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_37_0)>(blob_ptr + 802816);
    conv2d_bias_relu_38_0 = reinterpret_cast<decltype(conv2d_bias_relu_38_0)>(blob_ptr + 0);
    conv2d_bias_relu_39_0 = reinterpret_cast<decltype(conv2d_bias_relu_39_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_40_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_40_0)>(blob_ptr + 0);
    conv2d_bias_relu_41_0 = reinterpret_cast<decltype(conv2d_bias_relu_41_0)>(blob_ptr + 802816);
    conv2d_bias_relu_42_0 = reinterpret_cast<decltype(conv2d_bias_relu_42_0)>(blob_ptr + 1605632);
    conv2d_bias_add_relu_43_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_43_0)>(blob_ptr + 802816);
    conv2d_bias_relu_44_0 = reinterpret_cast<decltype(conv2d_bias_relu_44_0)>(blob_ptr + 0);
    conv2d_bias_45_0 = reinterpret_cast<decltype(conv2d_bias_45_0)>(blob_ptr + 401408);
    conv2d_bias_relu_46_0 = reinterpret_cast<decltype(conv2d_bias_relu_46_0)>(blob_ptr + 802816);
    conv2d_bias_add_relu_47_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_47_0)>(blob_ptr + 0);
    conv2d_bias_relu_48_0 = reinterpret_cast<decltype(conv2d_bias_relu_48_0)>(blob_ptr + 401408);
    conv2d_bias_relu_49_0 = reinterpret_cast<decltype(conv2d_bias_relu_49_0)>(blob_ptr + 802816);
    conv2d_bias_add_relu_50_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_50_0)>(blob_ptr + 401408);
    conv2d_bias_relu_51_0 = reinterpret_cast<decltype(conv2d_bias_relu_51_0)>(blob_ptr + 0);
    conv2d_bias_relu_52_0 = reinterpret_cast<decltype(conv2d_bias_relu_52_0)>(blob_ptr + 802816);
    conv2d_bias_add_relu_53_0 = reinterpret_cast<decltype(conv2d_bias_add_relu_53_0)>(blob_ptr + 0);
    avg_pool2d_54_0 = reinterpret_cast<decltype(avg_pool2d_54_0)>(blob_ptr + 401408);
    
         params_[0].shape_ptrs = {ParamDim(1, 1, &input0_dim_0), ParamDim(224, 224, &input0_dim_1), ParamDim(224, 224, &input0_dim_2), ParamDim(3, 3, &input0_dim_3)};
     params_[2].shape_ptrs = {ParamDim(64, 64, &stem_conv1_weight_dim_0), ParamDim(7, 7, &stem_conv1_weight_dim_1), ParamDim(7, 7, &stem_conv1_weight_dim_2), ParamDim(3, 3, &stem_conv1_weight_dim_3)};
     params_[3].shape_ptrs = {ParamDim(64, 64, &stem_conv1_bias_dim_0)};
     params_[4].shape_ptrs = {ParamDim(64, 64, &layer1_0_conv1_weight_dim_0), ParamDim(1, 1, &layer1_0_conv1_weight_dim_1), ParamDim(1, 1, &layer1_0_conv1_weight_dim_2), ParamDim(64, 64, &layer1_0_conv1_weight_dim_3)};
     params_[5].shape_ptrs = {ParamDim(64, 64, &layer1_0_conv1_bias_dim_0)};
     params_[6].shape_ptrs = {ParamDim(64, 64, &layer1_0_conv2_weight_dim_0), ParamDim(3, 3, &layer1_0_conv2_weight_dim_1), ParamDim(3, 3, &layer1_0_conv2_weight_dim_2), ParamDim(64, 64, &layer1_0_conv2_weight_dim_3)};
     params_[7].shape_ptrs = {ParamDim(64, 64, &layer1_0_conv2_bias_dim_0)};
     params_[8].shape_ptrs = {ParamDim(256, 256, &layer1_0_downsample_0_weight_dim_0), ParamDim(1, 1, &layer1_0_downsample_0_weight_dim_1), ParamDim(1, 1, &layer1_0_downsample_0_weight_dim_2), ParamDim(64, 64, &layer1_0_downsample_0_weight_dim_3)};
     params_[9].shape_ptrs = {ParamDim(256, 256, &layer1_0_downsample_0_bias_dim_0)};
     params_[10].shape_ptrs = {ParamDim(256, 256, &layer1_0_conv3_weight_dim_0), ParamDim(1, 1, &layer1_0_conv3_weight_dim_1), ParamDim(1, 1, &layer1_0_conv3_weight_dim_2), ParamDim(64, 64, &layer1_0_conv3_weight_dim_3)};
     params_[11].shape_ptrs = {ParamDim(256, 256, &layer1_0_conv3_bias_dim_0)};
     params_[12].shape_ptrs = {ParamDim(64, 64, &layer1_1_conv1_weight_dim_0), ParamDim(1, 1, &layer1_1_conv1_weight_dim_1), ParamDim(1, 1, &layer1_1_conv1_weight_dim_2), ParamDim(256, 256, &layer1_1_conv1_weight_dim_3)};
     params_[13].shape_ptrs = {ParamDim(64, 64, &layer1_1_conv1_bias_dim_0)};
     params_[14].shape_ptrs = {ParamDim(64, 64, &layer1_1_conv2_weight_dim_0), ParamDim(3, 3, &layer1_1_conv2_weight_dim_1), ParamDim(3, 3, &layer1_1_conv2_weight_dim_2), ParamDim(64, 64, &layer1_1_conv2_weight_dim_3)};
     params_[15].shape_ptrs = {ParamDim(64, 64, &layer1_1_conv2_bias_dim_0)};
     params_[16].shape_ptrs = {ParamDim(256, 256, &layer1_1_conv3_weight_dim_0), ParamDim(1, 1, &layer1_1_conv3_weight_dim_1), ParamDim(1, 1, &layer1_1_conv3_weight_dim_2), ParamDim(64, 64, &layer1_1_conv3_weight_dim_3)};
     params_[17].shape_ptrs = {ParamDim(256, 256, &layer1_1_conv3_bias_dim_0)};
     params_[18].shape_ptrs = {ParamDim(64, 64, &layer1_2_conv1_weight_dim_0), ParamDim(1, 1, &layer1_2_conv1_weight_dim_1), ParamDim(1, 1, &layer1_2_conv1_weight_dim_2), ParamDim(256, 256, &layer1_2_conv1_weight_dim_3)};
     params_[19].shape_ptrs = {ParamDim(64, 64, &layer1_2_conv1_bias_dim_0)};
     params_[20].shape_ptrs = {ParamDim(64, 64, &layer1_2_conv2_weight_dim_0), ParamDim(3, 3, &layer1_2_conv2_weight_dim_1), ParamDim(3, 3, &layer1_2_conv2_weight_dim_2), ParamDim(64, 64, &layer1_2_conv2_weight_dim_3)};
     params_[21].shape_ptrs = {ParamDim(64, 64, &layer1_2_conv2_bias_dim_0)};
     params_[22].shape_ptrs = {ParamDim(256, 256, &layer1_2_conv3_weight_dim_0), ParamDim(1, 1, &layer1_2_conv3_weight_dim_1), ParamDim(1, 1, &layer1_2_conv3_weight_dim_2), ParamDim(64, 64, &layer1_2_conv3_weight_dim_3)};
     params_[23].shape_ptrs = {ParamDim(256, 256, &layer1_2_conv3_bias_dim_0)};
     params_[24].shape_ptrs = {ParamDim(128, 128, &layer2_0_conv1_weight_dim_0), ParamDim(1, 1, &layer2_0_conv1_weight_dim_1), ParamDim(1, 1, &layer2_0_conv1_weight_dim_2), ParamDim(256, 256, &layer2_0_conv1_weight_dim_3)};
     params_[25].shape_ptrs = {ParamDim(128, 128, &layer2_0_conv1_bias_dim_0)};
     params_[26].shape_ptrs = {ParamDim(128, 128, &layer2_0_conv2_weight_dim_0), ParamDim(3, 3, &layer2_0_conv2_weight_dim_1), ParamDim(3, 3, &layer2_0_conv2_weight_dim_2), ParamDim(128, 128, &layer2_0_conv2_weight_dim_3)};
     params_[27].shape_ptrs = {ParamDim(128, 128, &layer2_0_conv2_bias_dim_0)};
     params_[28].shape_ptrs = {ParamDim(512, 512, &layer2_0_downsample_0_weight_dim_0), ParamDim(1, 1, &layer2_0_downsample_0_weight_dim_1), ParamDim(1, 1, &layer2_0_downsample_0_weight_dim_2), ParamDim(256, 256, &layer2_0_downsample_0_weight_dim_3)};
     params_[29].shape_ptrs = {ParamDim(512, 512, &layer2_0_downsample_0_bias_dim_0)};
     params_[30].shape_ptrs = {ParamDim(512, 512, &layer2_0_conv3_weight_dim_0), ParamDim(1, 1, &layer2_0_conv3_weight_dim_1), ParamDim(1, 1, &layer2_0_conv3_weight_dim_2), ParamDim(128, 128, &layer2_0_conv3_weight_dim_3)};
     params_[31].shape_ptrs = {ParamDim(512, 512, &layer2_0_conv3_bias_dim_0)};
     params_[32].shape_ptrs = {ParamDim(128, 128, &layer2_1_conv1_weight_dim_0), ParamDim(1, 1, &layer2_1_conv1_weight_dim_1), ParamDim(1, 1, &layer2_1_conv1_weight_dim_2), ParamDim(512, 512, &layer2_1_conv1_weight_dim_3)};
     params_[33].shape_ptrs = {ParamDim(128, 128, &layer2_1_conv1_bias_dim_0)};
     params_[34].shape_ptrs = {ParamDim(128, 128, &layer2_1_conv2_weight_dim_0), ParamDim(3, 3, &layer2_1_conv2_weight_dim_1), ParamDim(3, 3, &layer2_1_conv2_weight_dim_2), ParamDim(128, 128, &layer2_1_conv2_weight_dim_3)};
     params_[35].shape_ptrs = {ParamDim(128, 128, &layer2_1_conv2_bias_dim_0)};
     params_[36].shape_ptrs = {ParamDim(512, 512, &layer2_1_conv3_weight_dim_0), ParamDim(1, 1, &layer2_1_conv3_weight_dim_1), ParamDim(1, 1, &layer2_1_conv3_weight_dim_2), ParamDim(128, 128, &layer2_1_conv3_weight_dim_3)};
     params_[37].shape_ptrs = {ParamDim(512, 512, &layer2_1_conv3_bias_dim_0)};
     params_[38].shape_ptrs = {ParamDim(128, 128, &layer2_2_conv1_weight_dim_0), ParamDim(1, 1, &layer2_2_conv1_weight_dim_1), ParamDim(1, 1, &layer2_2_conv1_weight_dim_2), ParamDim(512, 512, &layer2_2_conv1_weight_dim_3)};
     params_[39].shape_ptrs = {ParamDim(128, 128, &layer2_2_conv1_bias_dim_0)};
     params_[40].shape_ptrs = {ParamDim(128, 128, &layer2_2_conv2_weight_dim_0), ParamDim(3, 3, &layer2_2_conv2_weight_dim_1), ParamDim(3, 3, &layer2_2_conv2_weight_dim_2), ParamDim(128, 128, &layer2_2_conv2_weight_dim_3)};
     params_[41].shape_ptrs = {ParamDim(128, 128, &layer2_2_conv2_bias_dim_0)};
     params_[42].shape_ptrs = {ParamDim(512, 512, &layer2_2_conv3_weight_dim_0), ParamDim(1, 1, &layer2_2_conv3_weight_dim_1), ParamDim(1, 1, &layer2_2_conv3_weight_dim_2), ParamDim(128, 128, &layer2_2_conv3_weight_dim_3)};
     params_[43].shape_ptrs = {ParamDim(512, 512, &layer2_2_conv3_bias_dim_0)};
     params_[44].shape_ptrs = {ParamDim(128, 128, &layer2_3_conv1_weight_dim_0), ParamDim(1, 1, &layer2_3_conv1_weight_dim_1), ParamDim(1, 1, &layer2_3_conv1_weight_dim_2), ParamDim(512, 512, &layer2_3_conv1_weight_dim_3)};
     params_[45].shape_ptrs = {ParamDim(128, 128, &layer2_3_conv1_bias_dim_0)};
     params_[46].shape_ptrs = {ParamDim(128, 128, &layer2_3_conv2_weight_dim_0), ParamDim(3, 3, &layer2_3_conv2_weight_dim_1), ParamDim(3, 3, &layer2_3_conv2_weight_dim_2), ParamDim(128, 128, &layer2_3_conv2_weight_dim_3)};
     params_[47].shape_ptrs = {ParamDim(128, 128, &layer2_3_conv2_bias_dim_0)};
     params_[48].shape_ptrs = {ParamDim(512, 512, &layer2_3_conv3_weight_dim_0), ParamDim(1, 1, &layer2_3_conv3_weight_dim_1), ParamDim(1, 1, &layer2_3_conv3_weight_dim_2), ParamDim(128, 128, &layer2_3_conv3_weight_dim_3)};
     params_[49].shape_ptrs = {ParamDim(512, 512, &layer2_3_conv3_bias_dim_0)};
     params_[50].shape_ptrs = {ParamDim(256, 256, &layer3_0_conv1_weight_dim_0), ParamDim(1, 1, &layer3_0_conv1_weight_dim_1), ParamDim(1, 1, &layer3_0_conv1_weight_dim_2), ParamDim(512, 512, &layer3_0_conv1_weight_dim_3)};
     params_[51].shape_ptrs = {ParamDim(256, 256, &layer3_0_conv1_bias_dim_0)};
     params_[52].shape_ptrs = {ParamDim(256, 256, &layer3_0_conv2_weight_dim_0), ParamDim(3, 3, &layer3_0_conv2_weight_dim_1), ParamDim(3, 3, &layer3_0_conv2_weight_dim_2), ParamDim(256, 256, &layer3_0_conv2_weight_dim_3)};
     params_[53].shape_ptrs = {ParamDim(256, 256, &layer3_0_conv2_bias_dim_0)};
     params_[54].shape_ptrs = {ParamDim(1024, 1024, &layer3_0_downsample_0_weight_dim_0), ParamDim(1, 1, &layer3_0_downsample_0_weight_dim_1), ParamDim(1, 1, &layer3_0_downsample_0_weight_dim_2), ParamDim(512, 512, &layer3_0_downsample_0_weight_dim_3)};
     params_[55].shape_ptrs = {ParamDim(1024, 1024, &layer3_0_downsample_0_bias_dim_0)};
     params_[56].shape_ptrs = {ParamDim(1024, 1024, &layer3_0_conv3_weight_dim_0), ParamDim(1, 1, &layer3_0_conv3_weight_dim_1), ParamDim(1, 1, &layer3_0_conv3_weight_dim_2), ParamDim(256, 256, &layer3_0_conv3_weight_dim_3)};
     params_[57].shape_ptrs = {ParamDim(1024, 1024, &layer3_0_conv3_bias_dim_0)};
     params_[58].shape_ptrs = {ParamDim(256, 256, &layer3_1_conv1_weight_dim_0), ParamDim(1, 1, &layer3_1_conv1_weight_dim_1), ParamDim(1, 1, &layer3_1_conv1_weight_dim_2), ParamDim(1024, 1024, &layer3_1_conv1_weight_dim_3)};
     params_[59].shape_ptrs = {ParamDim(256, 256, &layer3_1_conv1_bias_dim_0)};
     params_[60].shape_ptrs = {ParamDim(256, 256, &layer3_1_conv2_weight_dim_0), ParamDim(3, 3, &layer3_1_conv2_weight_dim_1), ParamDim(3, 3, &layer3_1_conv2_weight_dim_2), ParamDim(256, 256, &layer3_1_conv2_weight_dim_3)};
     params_[61].shape_ptrs = {ParamDim(256, 256, &layer3_1_conv2_bias_dim_0)};
     params_[62].shape_ptrs = {ParamDim(1024, 1024, &layer3_1_conv3_weight_dim_0), ParamDim(1, 1, &layer3_1_conv3_weight_dim_1), ParamDim(1, 1, &layer3_1_conv3_weight_dim_2), ParamDim(256, 256, &layer3_1_conv3_weight_dim_3)};
     params_[63].shape_ptrs = {ParamDim(1024, 1024, &layer3_1_conv3_bias_dim_0)};
     params_[64].shape_ptrs = {ParamDim(256, 256, &layer3_2_conv1_weight_dim_0), ParamDim(1, 1, &layer3_2_conv1_weight_dim_1), ParamDim(1, 1, &layer3_2_conv1_weight_dim_2), ParamDim(1024, 1024, &layer3_2_conv1_weight_dim_3)};
     params_[65].shape_ptrs = {ParamDim(256, 256, &layer3_2_conv1_bias_dim_0)};
     params_[66].shape_ptrs = {ParamDim(256, 256, &layer3_2_conv2_weight_dim_0), ParamDim(3, 3, &layer3_2_conv2_weight_dim_1), ParamDim(3, 3, &layer3_2_conv2_weight_dim_2), ParamDim(256, 256, &layer3_2_conv2_weight_dim_3)};
     params_[67].shape_ptrs = {ParamDim(256, 256, &layer3_2_conv2_bias_dim_0)};
     params_[68].shape_ptrs = {ParamDim(1024, 1024, &layer3_2_conv3_weight_dim_0), ParamDim(1, 1, &layer3_2_conv3_weight_dim_1), ParamDim(1, 1, &layer3_2_conv3_weight_dim_2), ParamDim(256, 256, &layer3_2_conv3_weight_dim_3)};
     params_[69].shape_ptrs = {ParamDim(1024, 1024, &layer3_2_conv3_bias_dim_0)};
     params_[70].shape_ptrs = {ParamDim(256, 256, &layer3_3_conv1_weight_dim_0), ParamDim(1, 1, &layer3_3_conv1_weight_dim_1), ParamDim(1, 1, &layer3_3_conv1_weight_dim_2), ParamDim(1024, 1024, &layer3_3_conv1_weight_dim_3)};
     params_[71].shape_ptrs = {ParamDim(256, 256, &layer3_3_conv1_bias_dim_0)};
     params_[72].shape_ptrs = {ParamDim(256, 256, &layer3_3_conv2_weight_dim_0), ParamDim(3, 3, &layer3_3_conv2_weight_dim_1), ParamDim(3, 3, &layer3_3_conv2_weight_dim_2), ParamDim(256, 256, &layer3_3_conv2_weight_dim_3)};
     params_[73].shape_ptrs = {ParamDim(256, 256, &layer3_3_conv2_bias_dim_0)};
     params_[74].shape_ptrs = {ParamDim(1024, 1024, &layer3_3_conv3_weight_dim_0), ParamDim(1, 1, &layer3_3_conv3_weight_dim_1), ParamDim(1, 1, &layer3_3_conv3_weight_dim_2), ParamDim(256, 256, &layer3_3_conv3_weight_dim_3)};
     params_[75].shape_ptrs = {ParamDim(1024, 1024, &layer3_3_conv3_bias_dim_0)};
     params_[76].shape_ptrs = {ParamDim(256, 256, &layer3_4_conv1_weight_dim_0), ParamDim(1, 1, &layer3_4_conv1_weight_dim_1), ParamDim(1, 1, &layer3_4_conv1_weight_dim_2), ParamDim(1024, 1024, &layer3_4_conv1_weight_dim_3)};
     params_[77].shape_ptrs = {ParamDim(256, 256, &layer3_4_conv1_bias_dim_0)};
     params_[78].shape_ptrs = {ParamDim(256, 256, &layer3_4_conv2_weight_dim_0), ParamDim(3, 3, &layer3_4_conv2_weight_dim_1), ParamDim(3, 3, &layer3_4_conv2_weight_dim_2), ParamDim(256, 256, &layer3_4_conv2_weight_dim_3)};
     params_[79].shape_ptrs = {ParamDim(256, 256, &layer3_4_conv2_bias_dim_0)};
     params_[80].shape_ptrs = {ParamDim(1024, 1024, &layer3_4_conv3_weight_dim_0), ParamDim(1, 1, &layer3_4_conv3_weight_dim_1), ParamDim(1, 1, &layer3_4_conv3_weight_dim_2), ParamDim(256, 256, &layer3_4_conv3_weight_dim_3)};
     params_[81].shape_ptrs = {ParamDim(1024, 1024, &layer3_4_conv3_bias_dim_0)};
     params_[82].shape_ptrs = {ParamDim(256, 256, &layer3_5_conv1_weight_dim_0), ParamDim(1, 1, &layer3_5_conv1_weight_dim_1), ParamDim(1, 1, &layer3_5_conv1_weight_dim_2), ParamDim(1024, 1024, &layer3_5_conv1_weight_dim_3)};
     params_[83].shape_ptrs = {ParamDim(256, 256, &layer3_5_conv1_bias_dim_0)};
     params_[84].shape_ptrs = {ParamDim(256, 256, &layer3_5_conv2_weight_dim_0), ParamDim(3, 3, &layer3_5_conv2_weight_dim_1), ParamDim(3, 3, &layer3_5_conv2_weight_dim_2), ParamDim(256, 256, &layer3_5_conv2_weight_dim_3)};
     params_[85].shape_ptrs = {ParamDim(256, 256, &layer3_5_conv2_bias_dim_0)};
     params_[86].shape_ptrs = {ParamDim(1024, 1024, &layer3_5_conv3_weight_dim_0), ParamDim(1, 1, &layer3_5_conv3_weight_dim_1), ParamDim(1, 1, &layer3_5_conv3_weight_dim_2), ParamDim(256, 256, &layer3_5_conv3_weight_dim_3)};
     params_[87].shape_ptrs = {ParamDim(1024, 1024, &layer3_5_conv3_bias_dim_0)};
     params_[88].shape_ptrs = {ParamDim(512, 512, &layer4_0_conv1_weight_dim_0), ParamDim(1, 1, &layer4_0_conv1_weight_dim_1), ParamDim(1, 1, &layer4_0_conv1_weight_dim_2), ParamDim(1024, 1024, &layer4_0_conv1_weight_dim_3)};
     params_[89].shape_ptrs = {ParamDim(512, 512, &layer4_0_conv1_bias_dim_0)};
     params_[90].shape_ptrs = {ParamDim(512, 512, &layer4_0_conv2_weight_dim_0), ParamDim(3, 3, &layer4_0_conv2_weight_dim_1), ParamDim(3, 3, &layer4_0_conv2_weight_dim_2), ParamDim(512, 512, &layer4_0_conv2_weight_dim_3)};
     params_[91].shape_ptrs = {ParamDim(512, 512, &layer4_0_conv2_bias_dim_0)};
     params_[92].shape_ptrs = {ParamDim(2048, 2048, &layer4_0_downsample_0_weight_dim_0), ParamDim(1, 1, &layer4_0_downsample_0_weight_dim_1), ParamDim(1, 1, &layer4_0_downsample_0_weight_dim_2), ParamDim(1024, 1024, &layer4_0_downsample_0_weight_dim_3)};
     params_[93].shape_ptrs = {ParamDim(2048, 2048, &layer4_0_downsample_0_bias_dim_0)};
     params_[94].shape_ptrs = {ParamDim(2048, 2048, &layer4_0_conv3_weight_dim_0), ParamDim(1, 1, &layer4_0_conv3_weight_dim_1), ParamDim(1, 1, &layer4_0_conv3_weight_dim_2), ParamDim(512, 512, &layer4_0_conv3_weight_dim_3)};
     params_[95].shape_ptrs = {ParamDim(2048, 2048, &layer4_0_conv3_bias_dim_0)};
     params_[96].shape_ptrs = {ParamDim(512, 512, &layer4_1_conv1_weight_dim_0), ParamDim(1, 1, &layer4_1_conv1_weight_dim_1), ParamDim(1, 1, &layer4_1_conv1_weight_dim_2), ParamDim(2048, 2048, &layer4_1_conv1_weight_dim_3)};
     params_[97].shape_ptrs = {ParamDim(512, 512, &layer4_1_conv1_bias_dim_0)};
     params_[98].shape_ptrs = {ParamDim(512, 512, &layer4_1_conv2_weight_dim_0), ParamDim(3, 3, &layer4_1_conv2_weight_dim_1), ParamDim(3, 3, &layer4_1_conv2_weight_dim_2), ParamDim(512, 512, &layer4_1_conv2_weight_dim_3)};
     params_[99].shape_ptrs = {ParamDim(512, 512, &layer4_1_conv2_bias_dim_0)};
     params_[100].shape_ptrs = {ParamDim(2048, 2048, &layer4_1_conv3_weight_dim_0), ParamDim(1, 1, &layer4_1_conv3_weight_dim_1), ParamDim(1, 1, &layer4_1_conv3_weight_dim_2), ParamDim(512, 512, &layer4_1_conv3_weight_dim_3)};
     params_[101].shape_ptrs = {ParamDim(2048, 2048, &layer4_1_conv3_bias_dim_0)};
     params_[102].shape_ptrs = {ParamDim(512, 512, &layer4_2_conv1_weight_dim_0), ParamDim(1, 1, &layer4_2_conv1_weight_dim_1), ParamDim(1, 1, &layer4_2_conv1_weight_dim_2), ParamDim(2048, 2048, &layer4_2_conv1_weight_dim_3)};
     params_[103].shape_ptrs = {ParamDim(512, 512, &layer4_2_conv1_bias_dim_0)};
     params_[104].shape_ptrs = {ParamDim(512, 512, &layer4_2_conv2_weight_dim_0), ParamDim(3, 3, &layer4_2_conv2_weight_dim_1), ParamDim(3, 3, &layer4_2_conv2_weight_dim_2), ParamDim(512, 512, &layer4_2_conv2_weight_dim_3)};
     params_[105].shape_ptrs = {ParamDim(512, 512, &layer4_2_conv2_bias_dim_0)};
     params_[106].shape_ptrs = {ParamDim(2048, 2048, &layer4_2_conv3_weight_dim_0), ParamDim(1, 1, &layer4_2_conv3_weight_dim_1), ParamDim(1, 1, &layer4_2_conv3_weight_dim_2), ParamDim(512, 512, &layer4_2_conv3_weight_dim_3)};
     params_[107].shape_ptrs = {ParamDim(2048, 2048, &layer4_2_conv3_bias_dim_0)};
     params_[108].shape_ptrs = {ParamDim(1000, 1000, &fc_weight_dim_0), ParamDim(2048, 2048, &fc_weight_dim_1)};
     params_[109].shape_ptrs = {ParamDim(1000, 1000, &fc_bias_dim_0)};
     params_[1].shape_ptrs = {ParamDim(1, 1, &reshape_55_0_dim_0), ParamDim(1, 1, &output_0_dim_1), ParamDim(1, 1, &output_0_dim_2), ParamDim(1000, 1000, &fc_weight_dim_0)};

    }

    ~Model() {

    }

    void SetUpInputsOutputs() {
             input0 = static_cast<decltype(input0)>(params_[0].ptr);

if (input0 == nullptr) {
    throw std::runtime_error("Constant input0 was not set! Set the value with set_constant.");
}
    

if (stem_conv1_weight == nullptr) {
    throw std::runtime_error("Constant stem_conv1_weight was not set! Set the value with set_constant.");
}
    

if (stem_conv1_bias == nullptr) {
    throw std::runtime_error("Constant stem_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer1_0_downsample_0_weight == nullptr) {
    throw std::runtime_error("Constant layer1_0_downsample_0_weight was not set! Set the value with set_constant.");
}
    

if (layer1_0_downsample_0_bias == nullptr) {
    throw std::runtime_error("Constant layer1_0_downsample_0_bias was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer1_0_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer1_0_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer1_1_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer1_1_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer1_2_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer1_2_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer2_0_downsample_0_weight == nullptr) {
    throw std::runtime_error("Constant layer2_0_downsample_0_weight was not set! Set the value with set_constant.");
}
    

if (layer2_0_downsample_0_bias == nullptr) {
    throw std::runtime_error("Constant layer2_0_downsample_0_bias was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer2_0_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer2_0_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer2_1_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer2_1_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer2_2_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer2_2_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer2_3_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer2_3_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_0_downsample_0_weight == nullptr) {
    throw std::runtime_error("Constant layer3_0_downsample_0_weight was not set! Set the value with set_constant.");
}
    

if (layer3_0_downsample_0_bias == nullptr) {
    throw std::runtime_error("Constant layer3_0_downsample_0_bias was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_0_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_0_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_1_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_1_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_2_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_2_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_3_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_3_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_4_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_4_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer3_5_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer3_5_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer4_0_downsample_0_weight == nullptr) {
    throw std::runtime_error("Constant layer4_0_downsample_0_weight was not set! Set the value with set_constant.");
}
    

if (layer4_0_downsample_0_bias == nullptr) {
    throw std::runtime_error("Constant layer4_0_downsample_0_bias was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer4_0_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer4_0_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer4_1_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer4_1_conv3_bias was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv1_weight == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv1_weight was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv1_bias == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv1_bias was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv2_weight == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv2_weight was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv2_bias == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv2_bias was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv3_weight == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv3_weight was not set! Set the value with set_constant.");
}
    

if (layer4_2_conv3_bias == nullptr) {
    throw std::runtime_error("Constant layer4_2_conv3_bias was not set! Set the value with set_constant.");
}
    

if (fc_weight == nullptr) {
    throw std::runtime_error("Constant fc_weight was not set! Set the value with set_constant.");
}
    

if (fc_bias == nullptr) {
    throw std::runtime_error("Constant fc_bias was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params_[1].ptr);

if (output_0 == nullptr) {
    throw std::runtime_error("Constant output_0 was not set! Set the value with set_constant.");
}
    
    }

    void ResetConstants(uint8_t* constants) {
        /*
         * This can be called if we want to use a different piece of memory
         * for the constants to be consumed.
         */
        
    }

    void DeviceToDeviceCopies(StreamType stream) {
  return;
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
  
  
    conv2d_bias_relu_0(
        input0,
        stem_conv1_weight,
        conv2d_bias_relu_0_0,

        stem_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &stem_conv1_weight_dim_0,
        &input0_dim_3,
        &stem_conv1_weight_dim_1,
        &stem_conv1_weight_dim_2,
        &input0_dim_1,
        &input0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_0_0_dim_1,
        &conv2d_bias_relu_0_0_dim_2,
        2,
        1,
        3,
        2,
        1,
        3,
        threadpool_.get()
    );
  
  
    max_pool2d_1(
        conv2d_bias_relu_0_0,
        max_pool2d_1_0,
        &input0_dim_0,
        &conv2d_bias_relu_0_0_dim_3,
        &conv2d_bias_relu_0_0_dim_1,
        &conv2d_bias_relu_0_0_dim_2,
        &input0_dim_0,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        3,
        3,
        2,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_2(
        max_pool2d_1_0,
        layer1_0_downsample_0_weight,
        conv2d_bias_2_0,

        layer1_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_downsample_0_weight_dim_0,
        &max_pool2d_1_0_dim_3,
        &layer1_0_downsample_0_weight_dim_1,
        &layer1_0_downsample_0_weight_dim_2,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_3(
        max_pool2d_1_0,
        layer1_0_conv1_weight,
        conv2d_bias_relu_3_0,

        layer1_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv1_weight_dim_0,
        &max_pool2d_1_0_dim_3,
        &layer1_0_conv1_weight_dim_1,
        &layer1_0_conv1_weight_dim_2,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_3_0_dim_1,
        &conv2d_bias_relu_3_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_4(
        conv2d_bias_relu_3_0,
        layer1_0_conv2_weight,
        conv2d_bias_relu_4_0,

        layer1_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv2_weight_dim_0,
        &conv2d_bias_relu_3_0_dim_3,
        &layer1_0_conv2_weight_dim_1,
        &layer1_0_conv2_weight_dim_2,
        &conv2d_bias_relu_3_0_dim_1,
        &conv2d_bias_relu_3_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_4_0_dim_1,
        &conv2d_bias_relu_4_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_4_0,
        layer1_0_conv3_weight,
        conv2d_bias_add_relu_5_0,

        layer1_0_conv3_bias,
        conv2d_bias_2_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv3_weight_dim_0,
        &conv2d_bias_relu_4_0_dim_3,
        &layer1_0_conv3_weight_dim_1,
        &layer1_0_conv3_weight_dim_2,
        &conv2d_bias_relu_4_0_dim_1,
        &conv2d_bias_relu_4_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_5_0_dim_1,
        &conv2d_bias_add_relu_5_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_6(
        conv2d_bias_add_relu_5_0,
        layer1_1_conv1_weight,
        conv2d_bias_relu_6_0,

        layer1_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_5_0_dim_3,
        &layer1_1_conv1_weight_dim_1,
        &layer1_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_5_0_dim_1,
        &conv2d_bias_add_relu_5_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_6_0_dim_1,
        &conv2d_bias_relu_6_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_4(
        conv2d_bias_relu_6_0,
        layer1_1_conv2_weight,
        conv2d_bias_relu_7_0,

        layer1_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv2_weight_dim_0,
        &conv2d_bias_relu_6_0_dim_3,
        &layer1_1_conv2_weight_dim_1,
        &layer1_1_conv2_weight_dim_2,
        &conv2d_bias_relu_6_0_dim_1,
        &conv2d_bias_relu_6_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_7_0_dim_1,
        &conv2d_bias_relu_7_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_7_0,
        layer1_1_conv3_weight,
        conv2d_bias_add_relu_8_0,

        layer1_1_conv3_bias,
        conv2d_bias_add_relu_5_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv3_weight_dim_0,
        &conv2d_bias_relu_7_0_dim_3,
        &layer1_1_conv3_weight_dim_1,
        &layer1_1_conv3_weight_dim_2,
        &conv2d_bias_relu_7_0_dim_1,
        &conv2d_bias_relu_7_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_8_0_dim_1,
        &conv2d_bias_add_relu_8_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_6(
        conv2d_bias_add_relu_8_0,
        layer1_2_conv1_weight,
        conv2d_bias_relu_9_0,

        layer1_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_8_0_dim_3,
        &layer1_2_conv1_weight_dim_1,
        &layer1_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_8_0_dim_1,
        &conv2d_bias_add_relu_8_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_9_0_dim_1,
        &conv2d_bias_relu_9_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_4(
        conv2d_bias_relu_9_0,
        layer1_2_conv2_weight,
        conv2d_bias_relu_10_0,

        layer1_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv2_weight_dim_0,
        &conv2d_bias_relu_9_0_dim_3,
        &layer1_2_conv2_weight_dim_1,
        &layer1_2_conv2_weight_dim_2,
        &conv2d_bias_relu_9_0_dim_1,
        &conv2d_bias_relu_9_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_10_0_dim_1,
        &conv2d_bias_relu_10_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_10_0,
        layer1_2_conv3_weight,
        conv2d_bias_add_relu_11_0,

        layer1_2_conv3_bias,
        conv2d_bias_add_relu_8_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv3_weight_dim_0,
        &conv2d_bias_relu_10_0_dim_3,
        &layer1_2_conv3_weight_dim_1,
        &layer1_2_conv3_weight_dim_2,
        &conv2d_bias_relu_10_0_dim_1,
        &conv2d_bias_relu_10_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_12(
        conv2d_bias_add_relu_11_0,
        layer2_0_conv1_weight,
        conv2d_bias_relu_12_0,

        layer2_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_11_0_dim_3,
        &layer2_0_conv1_weight_dim_1,
        &layer2_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_12_0_dim_1,
        &conv2d_bias_relu_12_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_13(
        conv2d_bias_add_relu_11_0,
        layer2_0_downsample_0_weight,
        conv2d_bias_13_0,

        layer2_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_11_0_dim_3,
        &layer2_0_downsample_0_weight_dim_1,
        &layer2_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_13_0_dim_1,
        &conv2d_bias_13_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_14(
        conv2d_bias_relu_12_0,
        layer2_0_conv2_weight,
        conv2d_bias_relu_14_0,

        layer2_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv2_weight_dim_0,
        &conv2d_bias_relu_12_0_dim_3,
        &layer2_0_conv2_weight_dim_1,
        &layer2_0_conv2_weight_dim_2,
        &conv2d_bias_relu_12_0_dim_1,
        &conv2d_bias_relu_12_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_14_0_dim_1,
        &conv2d_bias_relu_14_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_14_0,
        layer2_0_conv3_weight,
        conv2d_bias_add_relu_15_0,

        layer2_0_conv3_bias,
        conv2d_bias_13_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv3_weight_dim_0,
        &conv2d_bias_relu_14_0_dim_3,
        &layer2_0_conv3_weight_dim_1,
        &layer2_0_conv3_weight_dim_2,
        &conv2d_bias_relu_14_0_dim_1,
        &conv2d_bias_relu_14_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_15_0_dim_1,
        &conv2d_bias_add_relu_15_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_15_0,
        layer2_1_conv1_weight,
        conv2d_bias_relu_16_0,

        layer2_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_15_0_dim_3,
        &layer2_1_conv1_weight_dim_1,
        &layer2_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_15_0_dim_1,
        &conv2d_bias_add_relu_15_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_16_0_dim_1,
        &conv2d_bias_relu_16_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_17(
        conv2d_bias_relu_16_0,
        layer2_1_conv2_weight,
        conv2d_bias_relu_17_0,

        layer2_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv2_weight_dim_0,
        &conv2d_bias_relu_16_0_dim_3,
        &layer2_1_conv2_weight_dim_1,
        &layer2_1_conv2_weight_dim_2,
        &conv2d_bias_relu_16_0_dim_1,
        &conv2d_bias_relu_16_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_17_0_dim_1,
        &conv2d_bias_relu_17_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_17_0,
        layer2_1_conv3_weight,
        conv2d_bias_add_relu_18_0,

        layer2_1_conv3_bias,
        conv2d_bias_add_relu_15_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv3_weight_dim_0,
        &conv2d_bias_relu_17_0_dim_3,
        &layer2_1_conv3_weight_dim_1,
        &layer2_1_conv3_weight_dim_2,
        &conv2d_bias_relu_17_0_dim_1,
        &conv2d_bias_relu_17_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_18_0_dim_1,
        &conv2d_bias_add_relu_18_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_18_0,
        layer2_2_conv1_weight,
        conv2d_bias_relu_19_0,

        layer2_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_18_0_dim_3,
        &layer2_2_conv1_weight_dim_1,
        &layer2_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_18_0_dim_1,
        &conv2d_bias_add_relu_18_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_19_0_dim_1,
        &conv2d_bias_relu_19_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_17(
        conv2d_bias_relu_19_0,
        layer2_2_conv2_weight,
        conv2d_bias_relu_20_0,

        layer2_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv2_weight_dim_0,
        &conv2d_bias_relu_19_0_dim_3,
        &layer2_2_conv2_weight_dim_1,
        &layer2_2_conv2_weight_dim_2,
        &conv2d_bias_relu_19_0_dim_1,
        &conv2d_bias_relu_19_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_20_0_dim_1,
        &conv2d_bias_relu_20_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_20_0,
        layer2_2_conv3_weight,
        conv2d_bias_add_relu_21_0,

        layer2_2_conv3_bias,
        conv2d_bias_add_relu_18_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv3_weight_dim_0,
        &conv2d_bias_relu_20_0_dim_3,
        &layer2_2_conv3_weight_dim_1,
        &layer2_2_conv3_weight_dim_2,
        &conv2d_bias_relu_20_0_dim_1,
        &conv2d_bias_relu_20_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_21_0_dim_1,
        &conv2d_bias_add_relu_21_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_21_0,
        layer2_3_conv1_weight,
        conv2d_bias_relu_22_0,

        layer2_3_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv1_weight_dim_0,
        &conv2d_bias_add_relu_21_0_dim_3,
        &layer2_3_conv1_weight_dim_1,
        &layer2_3_conv1_weight_dim_2,
        &conv2d_bias_add_relu_21_0_dim_1,
        &conv2d_bias_add_relu_21_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_22_0_dim_1,
        &conv2d_bias_relu_22_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_17(
        conv2d_bias_relu_22_0,
        layer2_3_conv2_weight,
        conv2d_bias_relu_23_0,

        layer2_3_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv2_weight_dim_0,
        &conv2d_bias_relu_22_0_dim_3,
        &layer2_3_conv2_weight_dim_1,
        &layer2_3_conv2_weight_dim_2,
        &conv2d_bias_relu_22_0_dim_1,
        &conv2d_bias_relu_22_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_23_0_dim_1,
        &conv2d_bias_relu_23_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_23_0,
        layer2_3_conv3_weight,
        conv2d_bias_add_relu_24_0,

        layer2_3_conv3_bias,
        conv2d_bias_add_relu_21_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv3_weight_dim_0,
        &conv2d_bias_relu_23_0_dim_3,
        &layer2_3_conv3_weight_dim_1,
        &layer2_3_conv3_weight_dim_2,
        &conv2d_bias_relu_23_0_dim_1,
        &conv2d_bias_relu_23_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_25(
        conv2d_bias_add_relu_24_0,
        layer3_0_conv1_weight,
        conv2d_bias_relu_25_0,

        layer3_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_24_0_dim_3,
        &layer3_0_conv1_weight_dim_1,
        &layer3_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_25_0_dim_1,
        &conv2d_bias_relu_25_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_26(
        conv2d_bias_add_relu_24_0,
        layer3_0_downsample_0_weight,
        conv2d_bias_26_0,

        layer3_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_24_0_dim_3,
        &layer3_0_downsample_0_weight_dim_1,
        &layer3_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_26_0_dim_1,
        &conv2d_bias_26_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_27(
        conv2d_bias_relu_25_0,
        layer3_0_conv2_weight,
        conv2d_bias_relu_27_0,

        layer3_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv2_weight_dim_0,
        &conv2d_bias_relu_25_0_dim_3,
        &layer3_0_conv2_weight_dim_1,
        &layer3_0_conv2_weight_dim_2,
        &conv2d_bias_relu_25_0_dim_1,
        &conv2d_bias_relu_25_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_27_0_dim_1,
        &conv2d_bias_relu_27_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_27_0,
        layer3_0_conv3_weight,
        conv2d_bias_add_relu_28_0,

        layer3_0_conv3_bias,
        conv2d_bias_26_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv3_weight_dim_0,
        &conv2d_bias_relu_27_0_dim_3,
        &layer3_0_conv3_weight_dim_1,
        &layer3_0_conv3_weight_dim_2,
        &conv2d_bias_relu_27_0_dim_1,
        &conv2d_bias_relu_27_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_28_0_dim_1,
        &conv2d_bias_add_relu_28_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_28_0,
        layer3_1_conv1_weight,
        conv2d_bias_relu_29_0,

        layer3_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_28_0_dim_3,
        &layer3_1_conv1_weight_dim_1,
        &layer3_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_28_0_dim_1,
        &conv2d_bias_add_relu_28_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_29_0_dim_1,
        &conv2d_bias_relu_29_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_30(
        conv2d_bias_relu_29_0,
        layer3_1_conv2_weight,
        conv2d_bias_relu_30_0,

        layer3_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv2_weight_dim_0,
        &conv2d_bias_relu_29_0_dim_3,
        &layer3_1_conv2_weight_dim_1,
        &layer3_1_conv2_weight_dim_2,
        &conv2d_bias_relu_29_0_dim_1,
        &conv2d_bias_relu_29_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_30_0_dim_1,
        &conv2d_bias_relu_30_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_30_0,
        layer3_1_conv3_weight,
        conv2d_bias_add_relu_31_0,

        layer3_1_conv3_bias,
        conv2d_bias_add_relu_28_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv3_weight_dim_0,
        &conv2d_bias_relu_30_0_dim_3,
        &layer3_1_conv3_weight_dim_1,
        &layer3_1_conv3_weight_dim_2,
        &conv2d_bias_relu_30_0_dim_1,
        &conv2d_bias_relu_30_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_31_0_dim_1,
        &conv2d_bias_add_relu_31_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_31_0,
        layer3_2_conv1_weight,
        conv2d_bias_relu_32_0,

        layer3_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_31_0_dim_3,
        &layer3_2_conv1_weight_dim_1,
        &layer3_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_31_0_dim_1,
        &conv2d_bias_add_relu_31_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_32_0_dim_1,
        &conv2d_bias_relu_32_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_30(
        conv2d_bias_relu_32_0,
        layer3_2_conv2_weight,
        conv2d_bias_relu_33_0,

        layer3_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv2_weight_dim_0,
        &conv2d_bias_relu_32_0_dim_3,
        &layer3_2_conv2_weight_dim_1,
        &layer3_2_conv2_weight_dim_2,
        &conv2d_bias_relu_32_0_dim_1,
        &conv2d_bias_relu_32_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_33_0_dim_1,
        &conv2d_bias_relu_33_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_33_0,
        layer3_2_conv3_weight,
        conv2d_bias_add_relu_34_0,

        layer3_2_conv3_bias,
        conv2d_bias_add_relu_31_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv3_weight_dim_0,
        &conv2d_bias_relu_33_0_dim_3,
        &layer3_2_conv3_weight_dim_1,
        &layer3_2_conv3_weight_dim_2,
        &conv2d_bias_relu_33_0_dim_1,
        &conv2d_bias_relu_33_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_34_0_dim_1,
        &conv2d_bias_add_relu_34_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_34_0,
        layer3_3_conv1_weight,
        conv2d_bias_relu_35_0,

        layer3_3_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv1_weight_dim_0,
        &conv2d_bias_add_relu_34_0_dim_3,
        &layer3_3_conv1_weight_dim_1,
        &layer3_3_conv1_weight_dim_2,
        &conv2d_bias_add_relu_34_0_dim_1,
        &conv2d_bias_add_relu_34_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_35_0_dim_1,
        &conv2d_bias_relu_35_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_30(
        conv2d_bias_relu_35_0,
        layer3_3_conv2_weight,
        conv2d_bias_relu_36_0,

        layer3_3_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv2_weight_dim_0,
        &conv2d_bias_relu_35_0_dim_3,
        &layer3_3_conv2_weight_dim_1,
        &layer3_3_conv2_weight_dim_2,
        &conv2d_bias_relu_35_0_dim_1,
        &conv2d_bias_relu_35_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_36_0_dim_1,
        &conv2d_bias_relu_36_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_36_0,
        layer3_3_conv3_weight,
        conv2d_bias_add_relu_37_0,

        layer3_3_conv3_bias,
        conv2d_bias_add_relu_34_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv3_weight_dim_0,
        &conv2d_bias_relu_36_0_dim_3,
        &layer3_3_conv3_weight_dim_1,
        &layer3_3_conv3_weight_dim_2,
        &conv2d_bias_relu_36_0_dim_1,
        &conv2d_bias_relu_36_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_37_0_dim_1,
        &conv2d_bias_add_relu_37_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_37_0,
        layer3_4_conv1_weight,
        conv2d_bias_relu_38_0,

        layer3_4_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv1_weight_dim_0,
        &conv2d_bias_add_relu_37_0_dim_3,
        &layer3_4_conv1_weight_dim_1,
        &layer3_4_conv1_weight_dim_2,
        &conv2d_bias_add_relu_37_0_dim_1,
        &conv2d_bias_add_relu_37_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_38_0_dim_1,
        &conv2d_bias_relu_38_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_30(
        conv2d_bias_relu_38_0,
        layer3_4_conv2_weight,
        conv2d_bias_relu_39_0,

        layer3_4_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv2_weight_dim_0,
        &conv2d_bias_relu_38_0_dim_3,
        &layer3_4_conv2_weight_dim_1,
        &layer3_4_conv2_weight_dim_2,
        &conv2d_bias_relu_38_0_dim_1,
        &conv2d_bias_relu_38_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_39_0_dim_1,
        &conv2d_bias_relu_39_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_39_0,
        layer3_4_conv3_weight,
        conv2d_bias_add_relu_40_0,

        layer3_4_conv3_bias,
        conv2d_bias_add_relu_37_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv3_weight_dim_0,
        &conv2d_bias_relu_39_0_dim_3,
        &layer3_4_conv3_weight_dim_1,
        &layer3_4_conv3_weight_dim_2,
        &conv2d_bias_relu_39_0_dim_1,
        &conv2d_bias_relu_39_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_40_0_dim_1,
        &conv2d_bias_add_relu_40_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_40_0,
        layer3_5_conv1_weight,
        conv2d_bias_relu_41_0,

        layer3_5_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv1_weight_dim_0,
        &conv2d_bias_add_relu_40_0_dim_3,
        &layer3_5_conv1_weight_dim_1,
        &layer3_5_conv1_weight_dim_2,
        &conv2d_bias_add_relu_40_0_dim_1,
        &conv2d_bias_add_relu_40_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_41_0_dim_1,
        &conv2d_bias_relu_41_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_30(
        conv2d_bias_relu_41_0,
        layer3_5_conv2_weight,
        conv2d_bias_relu_42_0,

        layer3_5_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv2_weight_dim_0,
        &conv2d_bias_relu_41_0_dim_3,
        &layer3_5_conv2_weight_dim_1,
        &layer3_5_conv2_weight_dim_2,
        &conv2d_bias_relu_41_0_dim_1,
        &conv2d_bias_relu_41_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_42_0_dim_1,
        &conv2d_bias_relu_42_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_42_0,
        layer3_5_conv3_weight,
        conv2d_bias_add_relu_43_0,

        layer3_5_conv3_bias,
        conv2d_bias_add_relu_40_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv3_weight_dim_0,
        &conv2d_bias_relu_42_0_dim_3,
        &layer3_5_conv3_weight_dim_1,
        &layer3_5_conv3_weight_dim_2,
        &conv2d_bias_relu_42_0_dim_1,
        &conv2d_bias_relu_42_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_44(
        conv2d_bias_add_relu_43_0,
        layer4_0_conv1_weight,
        conv2d_bias_relu_44_0,

        layer4_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_43_0_dim_3,
        &layer4_0_conv1_weight_dim_1,
        &layer4_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_44_0_dim_1,
        &conv2d_bias_relu_44_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_45(
        conv2d_bias_add_relu_43_0,
        layer4_0_downsample_0_weight,
        conv2d_bias_45_0,

        layer4_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_43_0_dim_3,
        &layer4_0_downsample_0_weight_dim_1,
        &layer4_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_45_0_dim_1,
        &conv2d_bias_45_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_46(
        conv2d_bias_relu_44_0,
        layer4_0_conv2_weight,
        conv2d_bias_relu_46_0,

        layer4_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv2_weight_dim_0,
        &conv2d_bias_relu_44_0_dim_3,
        &layer4_0_conv2_weight_dim_1,
        &layer4_0_conv2_weight_dim_2,
        &conv2d_bias_relu_44_0_dim_1,
        &conv2d_bias_relu_44_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_46_0_dim_1,
        &conv2d_bias_relu_46_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_46_0,
        layer4_0_conv3_weight,
        conv2d_bias_add_relu_47_0,

        layer4_0_conv3_bias,
        conv2d_bias_45_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv3_weight_dim_0,
        &conv2d_bias_relu_46_0_dim_3,
        &layer4_0_conv3_weight_dim_1,
        &layer4_0_conv3_weight_dim_2,
        &conv2d_bias_relu_46_0_dim_1,
        &conv2d_bias_relu_46_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_47_0_dim_1,
        &conv2d_bias_add_relu_47_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_48(
        conv2d_bias_add_relu_47_0,
        layer4_1_conv1_weight,
        conv2d_bias_relu_48_0,

        layer4_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_47_0_dim_3,
        &layer4_1_conv1_weight_dim_1,
        &layer4_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_47_0_dim_1,
        &conv2d_bias_add_relu_47_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_48_0_dim_1,
        &conv2d_bias_relu_48_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_49(
        conv2d_bias_relu_48_0,
        layer4_1_conv2_weight,
        conv2d_bias_relu_49_0,

        layer4_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv2_weight_dim_0,
        &conv2d_bias_relu_48_0_dim_3,
        &layer4_1_conv2_weight_dim_1,
        &layer4_1_conv2_weight_dim_2,
        &conv2d_bias_relu_48_0_dim_1,
        &conv2d_bias_relu_48_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_49_0_dim_1,
        &conv2d_bias_relu_49_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_49_0,
        layer4_1_conv3_weight,
        conv2d_bias_add_relu_50_0,

        layer4_1_conv3_bias,
        conv2d_bias_add_relu_47_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv3_weight_dim_0,
        &conv2d_bias_relu_49_0_dim_3,
        &layer4_1_conv3_weight_dim_1,
        &layer4_1_conv3_weight_dim_2,
        &conv2d_bias_relu_49_0_dim_1,
        &conv2d_bias_relu_49_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_50_0_dim_1,
        &conv2d_bias_add_relu_50_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_48(
        conv2d_bias_add_relu_50_0,
        layer4_2_conv1_weight,
        conv2d_bias_relu_51_0,

        layer4_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_50_0_dim_3,
        &layer4_2_conv1_weight_dim_1,
        &layer4_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_50_0_dim_1,
        &conv2d_bias_add_relu_50_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_51_0_dim_1,
        &conv2d_bias_relu_51_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    conv2d_bias_relu_49(
        conv2d_bias_relu_51_0,
        layer4_2_conv2_weight,
        conv2d_bias_relu_52_0,

        layer4_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv2_weight_dim_0,
        &conv2d_bias_relu_51_0_dim_3,
        &layer4_2_conv2_weight_dim_1,
        &layer4_2_conv2_weight_dim_2,
        &conv2d_bias_relu_51_0_dim_1,
        &conv2d_bias_relu_51_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_52_0_dim_1,
        &conv2d_bias_relu_52_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
  
  
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_52_0,
        layer4_2_conv3_weight,
        conv2d_bias_add_relu_53_0,

        layer4_2_conv3_bias,
        conv2d_bias_add_relu_50_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv3_weight_dim_0,
        &conv2d_bias_relu_52_0_dim_3,
        &layer4_2_conv3_weight_dim_1,
        &layer4_2_conv3_weight_dim_2,
        &conv2d_bias_relu_52_0_dim_1,
        &conv2d_bias_relu_52_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_53_0_dim_1,
        &conv2d_bias_add_relu_53_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
  
  
    avg_pool2d_54(
        conv2d_bias_add_relu_53_0,
        avg_pool2d_54_0,
        &input0_dim_0,
        &conv2d_bias_add_relu_53_0_dim_3,
        &conv2d_bias_add_relu_53_0_dim_1,
        &conv2d_bias_add_relu_53_0_dim_2,
        &input0_dim_0,
        &avg_pool2d_54_0_dim_1,
        &avg_pool2d_54_0_dim_2,
        7,
        7,
        1,
        0,
        threadpool_.get()
    );
  
  
    {
    

    gemm_rcr_bias_56(
        avg_pool2d_54_0,
        fc_weight,

        fc_bias,

        output_0,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,


        &fc_weight_dim_0,

        &fc_weight_dim_1,


        &reshape_55_0_dim_0,

        &fc_weight_dim_0,

        threadpool_.get());
    }
  
    }


    void ProfileImpl(StreamType stream, size_t iters, const std::string& filename) {
#ifdef OPTIMIZE_FOR_COMPILATION_TIME
      throw std::runtime_error("Profile is disabled, please recompile without OPTIMIZE_FOR_COMPILE_TIME flag");
#else
      std::ofstream ss(filename);
      if (!ss) {
        throw std::runtime_error(std::string("Could not open file ") + filename);
      }

      ss << "{\n";
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_0" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_0(
        input0,
        stem_conv1_weight,
        conv2d_bias_relu_0_0,

        stem_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &stem_conv1_weight_dim_0,
        &input0_dim_3,
        &stem_conv1_weight_dim_1,
        &stem_conv1_weight_dim_2,
        &input0_dim_1,
        &input0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_0_0_dim_1,
        &conv2d_bias_relu_0_0_dim_2,
        2,
        1,
        3,
        2,
        1,
        3,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_0" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"224\", \"224\", \"3\"], [\"64\", \"7\", \"7\", \"3\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"112\", \"112\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"3\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "max_pool2d_1" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    max_pool2d_1(
        conv2d_bias_relu_0_0,
        max_pool2d_1_0,
        &input0_dim_0,
        &conv2d_bias_relu_0_0_dim_3,
        &conv2d_bias_relu_0_0_dim_1,
        &conv2d_bias_relu_0_0_dim_2,
        &input0_dim_0,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        3,
        3,
        2,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "max_pool2d_1" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"112\", \"112\", \"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"stride\": " << "\"2\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"kernel_size\": " << "\"3\""
        
          << ", \"reduce_func\": " << "\"max\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_2" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_2(
        max_pool2d_1_0,
        layer1_0_downsample_0_weight,
        conv2d_bias_2_0,

        layer1_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_downsample_0_weight_dim_0,
        &max_pool2d_1_0_dim_3,
        &layer1_0_downsample_0_weight_dim_1,
        &layer1_0_downsample_0_weight_dim_2,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_2_0_dim_1,
        &conv2d_bias_2_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_2" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"256\", \"1\", \"1\", \"64\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_3" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_3(
        max_pool2d_1_0,
        layer1_0_conv1_weight,
        conv2d_bias_relu_3_0,

        layer1_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv1_weight_dim_0,
        &max_pool2d_1_0_dim_3,
        &layer1_0_conv1_weight_dim_1,
        &layer1_0_conv1_weight_dim_2,
        &max_pool2d_1_0_dim_1,
        &max_pool2d_1_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_3_0_dim_1,
        &conv2d_bias_relu_3_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"64\", \"1\", \"1\", \"64\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_4" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_4(
        conv2d_bias_relu_3_0,
        layer1_0_conv2_weight,
        conv2d_bias_relu_4_0,

        layer1_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv2_weight_dim_0,
        &conv2d_bias_relu_3_0_dim_3,
        &layer1_0_conv2_weight_dim_1,
        &layer1_0_conv2_weight_dim_2,
        &conv2d_bias_relu_3_0_dim_1,
        &conv2d_bias_relu_3_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_4_0_dim_1,
        &conv2d_bias_relu_4_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_4" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"64\", \"3\", \"3\", \"64\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_5" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_4_0,
        layer1_0_conv3_weight,
        conv2d_bias_add_relu_5_0,

        layer1_0_conv3_bias,
        conv2d_bias_2_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_0_conv3_weight_dim_0,
        &conv2d_bias_relu_4_0_dim_3,
        &layer1_0_conv3_weight_dim_1,
        &layer1_0_conv3_weight_dim_2,
        &conv2d_bias_relu_4_0_dim_1,
        &conv2d_bias_relu_4_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_5_0_dim_1,
        &conv2d_bias_add_relu_5_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_5" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"256\", \"1\", \"1\", \"64\"], [\"256\"], [\"1\", \"56\", \"56\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_6" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_6(
        conv2d_bias_add_relu_5_0,
        layer1_1_conv1_weight,
        conv2d_bias_relu_6_0,

        layer1_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_5_0_dim_3,
        &layer1_1_conv1_weight_dim_1,
        &layer1_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_5_0_dim_1,
        &conv2d_bias_add_relu_5_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_6_0_dim_1,
        &conv2d_bias_relu_6_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_6" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"], [\"64\", \"1\", \"1\", \"256\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_7" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_4(
        conv2d_bias_relu_6_0,
        layer1_1_conv2_weight,
        conv2d_bias_relu_7_0,

        layer1_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv2_weight_dim_0,
        &conv2d_bias_relu_6_0_dim_3,
        &layer1_1_conv2_weight_dim_1,
        &layer1_1_conv2_weight_dim_2,
        &conv2d_bias_relu_6_0_dim_1,
        &conv2d_bias_relu_6_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_7_0_dim_1,
        &conv2d_bias_relu_7_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_7" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"64\", \"3\", \"3\", \"64\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_8" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_7_0,
        layer1_1_conv3_weight,
        conv2d_bias_add_relu_8_0,

        layer1_1_conv3_bias,
        conv2d_bias_add_relu_5_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_1_conv3_weight_dim_0,
        &conv2d_bias_relu_7_0_dim_3,
        &layer1_1_conv3_weight_dim_1,
        &layer1_1_conv3_weight_dim_2,
        &conv2d_bias_relu_7_0_dim_1,
        &conv2d_bias_relu_7_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_8_0_dim_1,
        &conv2d_bias_add_relu_8_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_8" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"256\", \"1\", \"1\", \"64\"], [\"256\"], [\"1\", \"56\", \"56\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_9" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_6(
        conv2d_bias_add_relu_8_0,
        layer1_2_conv1_weight,
        conv2d_bias_relu_9_0,

        layer1_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_8_0_dim_3,
        &layer1_2_conv1_weight_dim_1,
        &layer1_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_8_0_dim_1,
        &conv2d_bias_add_relu_8_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_9_0_dim_1,
        &conv2d_bias_relu_9_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_9" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"], [\"64\", \"1\", \"1\", \"256\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_10" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_4(
        conv2d_bias_relu_9_0,
        layer1_2_conv2_weight,
        conv2d_bias_relu_10_0,

        layer1_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv2_weight_dim_0,
        &conv2d_bias_relu_9_0_dim_3,
        &layer1_2_conv2_weight_dim_1,
        &layer1_2_conv2_weight_dim_2,
        &conv2d_bias_relu_9_0_dim_1,
        &conv2d_bias_relu_9_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_10_0_dim_1,
        &conv2d_bias_relu_10_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_10" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"64\", \"3\", \"3\", \"64\"], [\"64\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_11" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_5(
        conv2d_bias_relu_10_0,
        layer1_2_conv3_weight,
        conv2d_bias_add_relu_11_0,

        layer1_2_conv3_bias,
        conv2d_bias_add_relu_8_0,

        global_workspace_,
        &input0_dim_0,
        &layer1_2_conv3_weight_dim_0,
        &conv2d_bias_relu_10_0_dim_3,
        &layer1_2_conv3_weight_dim_1,
        &layer1_2_conv3_weight_dim_2,
        &conv2d_bias_relu_10_0_dim_1,
        &conv2d_bias_relu_10_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_11" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"64\"], [\"256\", \"1\", \"1\", \"64\"], [\"256\"], [\"1\", \"56\", \"56\", \"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_12" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_12(
        conv2d_bias_add_relu_11_0,
        layer2_0_conv1_weight,
        conv2d_bias_relu_12_0,

        layer2_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_11_0_dim_3,
        &layer2_0_conv1_weight_dim_1,
        &layer2_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_12_0_dim_1,
        &conv2d_bias_relu_12_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_12" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"], [\"128\", \"1\", \"1\", \"256\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"56\", \"56\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_13" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_13(
        conv2d_bias_add_relu_11_0,
        layer2_0_downsample_0_weight,
        conv2d_bias_13_0,

        layer2_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_11_0_dim_3,
        &layer2_0_downsample_0_weight_dim_1,
        &layer2_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_11_0_dim_1,
        &conv2d_bias_add_relu_11_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_13_0_dim_1,
        &conv2d_bias_13_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_13" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"256\"], [\"512\", \"1\", \"1\", \"256\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_14" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_14(
        conv2d_bias_relu_12_0,
        layer2_0_conv2_weight,
        conv2d_bias_relu_14_0,

        layer2_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv2_weight_dim_0,
        &conv2d_bias_relu_12_0_dim_3,
        &layer2_0_conv2_weight_dim_1,
        &layer2_0_conv2_weight_dim_2,
        &conv2d_bias_relu_12_0_dim_1,
        &conv2d_bias_relu_12_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_14_0_dim_1,
        &conv2d_bias_relu_14_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_14" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"56\", \"56\", \"128\"], [\"128\", \"3\", \"3\", \"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_15" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_14_0,
        layer2_0_conv3_weight,
        conv2d_bias_add_relu_15_0,

        layer2_0_conv3_bias,
        conv2d_bias_13_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_0_conv3_weight_dim_0,
        &conv2d_bias_relu_14_0_dim_3,
        &layer2_0_conv3_weight_dim_1,
        &layer2_0_conv3_weight_dim_2,
        &conv2d_bias_relu_14_0_dim_1,
        &conv2d_bias_relu_14_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_15_0_dim_1,
        &conv2d_bias_add_relu_15_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_15" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"512\", \"1\", \"1\", \"128\"], [\"512\"], [\"1\", \"28\", \"28\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_16" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_15_0,
        layer2_1_conv1_weight,
        conv2d_bias_relu_16_0,

        layer2_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_15_0_dim_3,
        &layer2_1_conv1_weight_dim_1,
        &layer2_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_15_0_dim_1,
        &conv2d_bias_add_relu_15_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_16_0_dim_1,
        &conv2d_bias_relu_16_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_16" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"], [\"128\", \"1\", \"1\", \"512\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_17" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_17(
        conv2d_bias_relu_16_0,
        layer2_1_conv2_weight,
        conv2d_bias_relu_17_0,

        layer2_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv2_weight_dim_0,
        &conv2d_bias_relu_16_0_dim_3,
        &layer2_1_conv2_weight_dim_1,
        &layer2_1_conv2_weight_dim_2,
        &conv2d_bias_relu_16_0_dim_1,
        &conv2d_bias_relu_16_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_17_0_dim_1,
        &conv2d_bias_relu_17_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_17" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"128\", \"3\", \"3\", \"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_18" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_17_0,
        layer2_1_conv3_weight,
        conv2d_bias_add_relu_18_0,

        layer2_1_conv3_bias,
        conv2d_bias_add_relu_15_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_1_conv3_weight_dim_0,
        &conv2d_bias_relu_17_0_dim_3,
        &layer2_1_conv3_weight_dim_1,
        &layer2_1_conv3_weight_dim_2,
        &conv2d_bias_relu_17_0_dim_1,
        &conv2d_bias_relu_17_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_18_0_dim_1,
        &conv2d_bias_add_relu_18_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_18" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"512\", \"1\", \"1\", \"128\"], [\"512\"], [\"1\", \"28\", \"28\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_19" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_18_0,
        layer2_2_conv1_weight,
        conv2d_bias_relu_19_0,

        layer2_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_18_0_dim_3,
        &layer2_2_conv1_weight_dim_1,
        &layer2_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_18_0_dim_1,
        &conv2d_bias_add_relu_18_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_19_0_dim_1,
        &conv2d_bias_relu_19_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_19" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"], [\"128\", \"1\", \"1\", \"512\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_20" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_17(
        conv2d_bias_relu_19_0,
        layer2_2_conv2_weight,
        conv2d_bias_relu_20_0,

        layer2_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv2_weight_dim_0,
        &conv2d_bias_relu_19_0_dim_3,
        &layer2_2_conv2_weight_dim_1,
        &layer2_2_conv2_weight_dim_2,
        &conv2d_bias_relu_19_0_dim_1,
        &conv2d_bias_relu_19_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_20_0_dim_1,
        &conv2d_bias_relu_20_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_20" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"128\", \"3\", \"3\", \"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_21" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_20_0,
        layer2_2_conv3_weight,
        conv2d_bias_add_relu_21_0,

        layer2_2_conv3_bias,
        conv2d_bias_add_relu_18_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_2_conv3_weight_dim_0,
        &conv2d_bias_relu_20_0_dim_3,
        &layer2_2_conv3_weight_dim_1,
        &layer2_2_conv3_weight_dim_2,
        &conv2d_bias_relu_20_0_dim_1,
        &conv2d_bias_relu_20_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_21_0_dim_1,
        &conv2d_bias_add_relu_21_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_21" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"512\", \"1\", \"1\", \"128\"], [\"512\"], [\"1\", \"28\", \"28\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_22" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_16(
        conv2d_bias_add_relu_21_0,
        layer2_3_conv1_weight,
        conv2d_bias_relu_22_0,

        layer2_3_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv1_weight_dim_0,
        &conv2d_bias_add_relu_21_0_dim_3,
        &layer2_3_conv1_weight_dim_1,
        &layer2_3_conv1_weight_dim_2,
        &conv2d_bias_add_relu_21_0_dim_1,
        &conv2d_bias_add_relu_21_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_22_0_dim_1,
        &conv2d_bias_relu_22_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_22" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"], [\"128\", \"1\", \"1\", \"512\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_23" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_17(
        conv2d_bias_relu_22_0,
        layer2_3_conv2_weight,
        conv2d_bias_relu_23_0,

        layer2_3_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv2_weight_dim_0,
        &conv2d_bias_relu_22_0_dim_3,
        &layer2_3_conv2_weight_dim_1,
        &layer2_3_conv2_weight_dim_2,
        &conv2d_bias_relu_22_0_dim_1,
        &conv2d_bias_relu_22_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_23_0_dim_1,
        &conv2d_bias_relu_23_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_23" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"128\", \"3\", \"3\", \"128\"], [\"128\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_24" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_15(
        conv2d_bias_relu_23_0,
        layer2_3_conv3_weight,
        conv2d_bias_add_relu_24_0,

        layer2_3_conv3_bias,
        conv2d_bias_add_relu_21_0,

        global_workspace_,
        &input0_dim_0,
        &layer2_3_conv3_weight_dim_0,
        &conv2d_bias_relu_23_0_dim_3,
        &layer2_3_conv3_weight_dim_1,
        &layer2_3_conv3_weight_dim_2,
        &conv2d_bias_relu_23_0_dim_1,
        &conv2d_bias_relu_23_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_24" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"128\"], [\"512\", \"1\", \"1\", \"128\"], [\"512\"], [\"1\", \"28\", \"28\", \"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_25" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_25(
        conv2d_bias_add_relu_24_0,
        layer3_0_conv1_weight,
        conv2d_bias_relu_25_0,

        layer3_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_24_0_dim_3,
        &layer3_0_conv1_weight_dim_1,
        &layer3_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_25_0_dim_1,
        &conv2d_bias_relu_25_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_25" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"], [\"256\", \"1\", \"1\", \"512\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"28\", \"28\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_26" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_26(
        conv2d_bias_add_relu_24_0,
        layer3_0_downsample_0_weight,
        conv2d_bias_26_0,

        layer3_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_24_0_dim_3,
        &layer3_0_downsample_0_weight_dim_1,
        &layer3_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_24_0_dim_1,
        &conv2d_bias_add_relu_24_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_26_0_dim_1,
        &conv2d_bias_26_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_26" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"512\"], [\"1024\", \"1\", \"1\", \"512\"], [\"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_27" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_27(
        conv2d_bias_relu_25_0,
        layer3_0_conv2_weight,
        conv2d_bias_relu_27_0,

        layer3_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv2_weight_dim_0,
        &conv2d_bias_relu_25_0_dim_3,
        &layer3_0_conv2_weight_dim_1,
        &layer3_0_conv2_weight_dim_2,
        &conv2d_bias_relu_25_0_dim_1,
        &conv2d_bias_relu_25_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_27_0_dim_1,
        &conv2d_bias_relu_27_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_27" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"28\", \"28\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_28" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_27_0,
        layer3_0_conv3_weight,
        conv2d_bias_add_relu_28_0,

        layer3_0_conv3_bias,
        conv2d_bias_26_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_0_conv3_weight_dim_0,
        &conv2d_bias_relu_27_0_dim_3,
        &layer3_0_conv3_weight_dim_1,
        &layer3_0_conv3_weight_dim_2,
        &conv2d_bias_relu_27_0_dim_1,
        &conv2d_bias_relu_27_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_28_0_dim_1,
        &conv2d_bias_add_relu_28_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_28" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_29" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_28_0,
        layer3_1_conv1_weight,
        conv2d_bias_relu_29_0,

        layer3_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_28_0_dim_3,
        &layer3_1_conv1_weight_dim_1,
        &layer3_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_28_0_dim_1,
        &conv2d_bias_add_relu_28_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_29_0_dim_1,
        &conv2d_bias_relu_29_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_29" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"256\", \"1\", \"1\", \"1024\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_30" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_30(
        conv2d_bias_relu_29_0,
        layer3_1_conv2_weight,
        conv2d_bias_relu_30_0,

        layer3_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv2_weight_dim_0,
        &conv2d_bias_relu_29_0_dim_3,
        &layer3_1_conv2_weight_dim_1,
        &layer3_1_conv2_weight_dim_2,
        &conv2d_bias_relu_29_0_dim_1,
        &conv2d_bias_relu_29_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_30_0_dim_1,
        &conv2d_bias_relu_30_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_30" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_31" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_30_0,
        layer3_1_conv3_weight,
        conv2d_bias_add_relu_31_0,

        layer3_1_conv3_bias,
        conv2d_bias_add_relu_28_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_1_conv3_weight_dim_0,
        &conv2d_bias_relu_30_0_dim_3,
        &layer3_1_conv3_weight_dim_1,
        &layer3_1_conv3_weight_dim_2,
        &conv2d_bias_relu_30_0_dim_1,
        &conv2d_bias_relu_30_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_31_0_dim_1,
        &conv2d_bias_add_relu_31_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_31" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_32" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_31_0,
        layer3_2_conv1_weight,
        conv2d_bias_relu_32_0,

        layer3_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_31_0_dim_3,
        &layer3_2_conv1_weight_dim_1,
        &layer3_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_31_0_dim_1,
        &conv2d_bias_add_relu_31_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_32_0_dim_1,
        &conv2d_bias_relu_32_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_32" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"256\", \"1\", \"1\", \"1024\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_33" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_30(
        conv2d_bias_relu_32_0,
        layer3_2_conv2_weight,
        conv2d_bias_relu_33_0,

        layer3_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv2_weight_dim_0,
        &conv2d_bias_relu_32_0_dim_3,
        &layer3_2_conv2_weight_dim_1,
        &layer3_2_conv2_weight_dim_2,
        &conv2d_bias_relu_32_0_dim_1,
        &conv2d_bias_relu_32_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_33_0_dim_1,
        &conv2d_bias_relu_33_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_33" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_34" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_33_0,
        layer3_2_conv3_weight,
        conv2d_bias_add_relu_34_0,

        layer3_2_conv3_bias,
        conv2d_bias_add_relu_31_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_2_conv3_weight_dim_0,
        &conv2d_bias_relu_33_0_dim_3,
        &layer3_2_conv3_weight_dim_1,
        &layer3_2_conv3_weight_dim_2,
        &conv2d_bias_relu_33_0_dim_1,
        &conv2d_bias_relu_33_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_34_0_dim_1,
        &conv2d_bias_add_relu_34_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_34" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_35" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_34_0,
        layer3_3_conv1_weight,
        conv2d_bias_relu_35_0,

        layer3_3_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv1_weight_dim_0,
        &conv2d_bias_add_relu_34_0_dim_3,
        &layer3_3_conv1_weight_dim_1,
        &layer3_3_conv1_weight_dim_2,
        &conv2d_bias_add_relu_34_0_dim_1,
        &conv2d_bias_add_relu_34_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_35_0_dim_1,
        &conv2d_bias_relu_35_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_35" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"256\", \"1\", \"1\", \"1024\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_36" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_30(
        conv2d_bias_relu_35_0,
        layer3_3_conv2_weight,
        conv2d_bias_relu_36_0,

        layer3_3_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv2_weight_dim_0,
        &conv2d_bias_relu_35_0_dim_3,
        &layer3_3_conv2_weight_dim_1,
        &layer3_3_conv2_weight_dim_2,
        &conv2d_bias_relu_35_0_dim_1,
        &conv2d_bias_relu_35_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_36_0_dim_1,
        &conv2d_bias_relu_36_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_36" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_37" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_36_0,
        layer3_3_conv3_weight,
        conv2d_bias_add_relu_37_0,

        layer3_3_conv3_bias,
        conv2d_bias_add_relu_34_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_3_conv3_weight_dim_0,
        &conv2d_bias_relu_36_0_dim_3,
        &layer3_3_conv3_weight_dim_1,
        &layer3_3_conv3_weight_dim_2,
        &conv2d_bias_relu_36_0_dim_1,
        &conv2d_bias_relu_36_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_37_0_dim_1,
        &conv2d_bias_add_relu_37_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_37" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_38" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_37_0,
        layer3_4_conv1_weight,
        conv2d_bias_relu_38_0,

        layer3_4_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv1_weight_dim_0,
        &conv2d_bias_add_relu_37_0_dim_3,
        &layer3_4_conv1_weight_dim_1,
        &layer3_4_conv1_weight_dim_2,
        &conv2d_bias_add_relu_37_0_dim_1,
        &conv2d_bias_add_relu_37_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_38_0_dim_1,
        &conv2d_bias_relu_38_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_38" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"256\", \"1\", \"1\", \"1024\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_39" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_30(
        conv2d_bias_relu_38_0,
        layer3_4_conv2_weight,
        conv2d_bias_relu_39_0,

        layer3_4_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv2_weight_dim_0,
        &conv2d_bias_relu_38_0_dim_3,
        &layer3_4_conv2_weight_dim_1,
        &layer3_4_conv2_weight_dim_2,
        &conv2d_bias_relu_38_0_dim_1,
        &conv2d_bias_relu_38_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_39_0_dim_1,
        &conv2d_bias_relu_39_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_39" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_40" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_39_0,
        layer3_4_conv3_weight,
        conv2d_bias_add_relu_40_0,

        layer3_4_conv3_bias,
        conv2d_bias_add_relu_37_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_4_conv3_weight_dim_0,
        &conv2d_bias_relu_39_0_dim_3,
        &layer3_4_conv3_weight_dim_1,
        &layer3_4_conv3_weight_dim_2,
        &conv2d_bias_relu_39_0_dim_1,
        &conv2d_bias_relu_39_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_40_0_dim_1,
        &conv2d_bias_add_relu_40_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_40" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_41" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_29(
        conv2d_bias_add_relu_40_0,
        layer3_5_conv1_weight,
        conv2d_bias_relu_41_0,

        layer3_5_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv1_weight_dim_0,
        &conv2d_bias_add_relu_40_0_dim_3,
        &layer3_5_conv1_weight_dim_1,
        &layer3_5_conv1_weight_dim_2,
        &conv2d_bias_add_relu_40_0_dim_1,
        &conv2d_bias_add_relu_40_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_41_0_dim_1,
        &conv2d_bias_relu_41_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_41" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"256\", \"1\", \"1\", \"1024\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_42" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_30(
        conv2d_bias_relu_41_0,
        layer3_5_conv2_weight,
        conv2d_bias_relu_42_0,

        layer3_5_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv2_weight_dim_0,
        &conv2d_bias_relu_41_0_dim_3,
        &layer3_5_conv2_weight_dim_1,
        &layer3_5_conv2_weight_dim_2,
        &conv2d_bias_relu_41_0_dim_1,
        &conv2d_bias_relu_41_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_42_0_dim_1,
        &conv2d_bias_relu_42_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_42" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"256\", \"3\", \"3\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_43" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_28(
        conv2d_bias_relu_42_0,
        layer3_5_conv3_weight,
        conv2d_bias_add_relu_43_0,

        layer3_5_conv3_bias,
        conv2d_bias_add_relu_40_0,

        global_workspace_,
        &input0_dim_0,
        &layer3_5_conv3_weight_dim_0,
        &conv2d_bias_relu_42_0_dim_3,
        &layer3_5_conv3_weight_dim_1,
        &layer3_5_conv3_weight_dim_2,
        &conv2d_bias_relu_42_0_dim_1,
        &conv2d_bias_relu_42_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_43" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"256\"], [\"1024\", \"1\", \"1\", \"256\"], [\"1024\"], [\"1\", \"14\", \"14\", \"1024\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_44" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_44(
        conv2d_bias_add_relu_43_0,
        layer4_0_conv1_weight,
        conv2d_bias_relu_44_0,

        layer4_0_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv1_weight_dim_0,
        &conv2d_bias_add_relu_43_0_dim_3,
        &layer4_0_conv1_weight_dim_1,
        &layer4_0_conv1_weight_dim_2,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_44_0_dim_1,
        &conv2d_bias_relu_44_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_44" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"512\", \"1\", \"1\", \"1024\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"14\", \"14\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_45" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_45(
        conv2d_bias_add_relu_43_0,
        layer4_0_downsample_0_weight,
        conv2d_bias_45_0,

        layer4_0_downsample_0_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_downsample_0_weight_dim_0,
        &conv2d_bias_add_relu_43_0_dim_3,
        &layer4_0_downsample_0_weight_dim_1,
        &layer4_0_downsample_0_weight_dim_2,
        &conv2d_bias_add_relu_43_0_dim_1,
        &conv2d_bias_add_relu_43_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_45_0_dim_1,
        &conv2d_bias_45_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_45" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"1024\"], [\"2048\", \"1\", \"1\", \"1024\"], [\"2048\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_46" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_46(
        conv2d_bias_relu_44_0,
        layer4_0_conv2_weight,
        conv2d_bias_relu_46_0,

        layer4_0_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv2_weight_dim_0,
        &conv2d_bias_relu_44_0_dim_3,
        &layer4_0_conv2_weight_dim_1,
        &layer4_0_conv2_weight_dim_2,
        &conv2d_bias_relu_44_0_dim_1,
        &conv2d_bias_relu_44_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_46_0_dim_1,
        &conv2d_bias_relu_46_0_dim_2,
        2,
        1,
        1,
        2,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_46" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"14\", \"14\", \"512\"], [\"512\", \"3\", \"3\", \"512\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_47" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_46_0,
        layer4_0_conv3_weight,
        conv2d_bias_add_relu_47_0,

        layer4_0_conv3_bias,
        conv2d_bias_45_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_0_conv3_weight_dim_0,
        &conv2d_bias_relu_46_0_dim_3,
        &layer4_0_conv3_weight_dim_1,
        &layer4_0_conv3_weight_dim_2,
        &conv2d_bias_relu_46_0_dim_1,
        &conv2d_bias_relu_46_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_47_0_dim_1,
        &conv2d_bias_add_relu_47_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_47" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"], [\"2048\", \"1\", \"1\", \"512\"], [\"2048\"], [\"1\", \"7\", \"7\", \"2048\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_48" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_48(
        conv2d_bias_add_relu_47_0,
        layer4_1_conv1_weight,
        conv2d_bias_relu_48_0,

        layer4_1_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv1_weight_dim_0,
        &conv2d_bias_add_relu_47_0_dim_3,
        &layer4_1_conv1_weight_dim_1,
        &layer4_1_conv1_weight_dim_2,
        &conv2d_bias_add_relu_47_0_dim_1,
        &conv2d_bias_add_relu_47_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_48_0_dim_1,
        &conv2d_bias_relu_48_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_48" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"], [\"512\", \"1\", \"1\", \"2048\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_49" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_49(
        conv2d_bias_relu_48_0,
        layer4_1_conv2_weight,
        conv2d_bias_relu_49_0,

        layer4_1_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv2_weight_dim_0,
        &conv2d_bias_relu_48_0_dim_3,
        &layer4_1_conv2_weight_dim_1,
        &layer4_1_conv2_weight_dim_2,
        &conv2d_bias_relu_48_0_dim_1,
        &conv2d_bias_relu_48_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_49_0_dim_1,
        &conv2d_bias_relu_49_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_49" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"], [\"512\", \"3\", \"3\", \"512\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_50" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_49_0,
        layer4_1_conv3_weight,
        conv2d_bias_add_relu_50_0,

        layer4_1_conv3_bias,
        conv2d_bias_add_relu_47_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_1_conv3_weight_dim_0,
        &conv2d_bias_relu_49_0_dim_3,
        &layer4_1_conv3_weight_dim_1,
        &layer4_1_conv3_weight_dim_2,
        &conv2d_bias_relu_49_0_dim_1,
        &conv2d_bias_relu_49_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_50_0_dim_1,
        &conv2d_bias_add_relu_50_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_50" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"], [\"2048\", \"1\", \"1\", \"512\"], [\"2048\"], [\"1\", \"7\", \"7\", \"2048\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_51" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_48(
        conv2d_bias_add_relu_50_0,
        layer4_2_conv1_weight,
        conv2d_bias_relu_51_0,

        layer4_2_conv1_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv1_weight_dim_0,
        &conv2d_bias_add_relu_50_0_dim_3,
        &layer4_2_conv1_weight_dim_1,
        &layer4_2_conv1_weight_dim_2,
        &conv2d_bias_add_relu_50_0_dim_1,
        &conv2d_bias_add_relu_50_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_51_0_dim_1,
        &conv2d_bias_relu_51_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_51" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"], [\"512\", \"1\", \"1\", \"2048\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_relu_52" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_relu_49(
        conv2d_bias_relu_51_0,
        layer4_2_conv2_weight,
        conv2d_bias_relu_52_0,

        layer4_2_conv2_bias,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv2_weight_dim_0,
        &conv2d_bias_relu_51_0_dim_3,
        &layer4_2_conv2_weight_dim_1,
        &layer4_2_conv2_weight_dim_2,
        &conv2d_bias_relu_51_0_dim_1,
        &conv2d_bias_relu_51_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_relu_52_0_dim_1,
        &conv2d_bias_relu_52_0_dim_2,
        1,
        1,
        1,
        1,
        1,
        1,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_relu_52" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"], [\"512\", \"3\", \"3\", \"512\"], [\"512\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"1\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "conv2d_bias_add_relu_53" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    conv2d_bias_add_relu_47(
        conv2d_bias_relu_52_0,
        layer4_2_conv3_weight,
        conv2d_bias_add_relu_53_0,

        layer4_2_conv3_bias,
        conv2d_bias_add_relu_50_0,

        global_workspace_,
        &input0_dim_0,
        &layer4_2_conv3_weight_dim_0,
        &conv2d_bias_relu_52_0_dim_3,
        &layer4_2_conv3_weight_dim_1,
        &layer4_2_conv3_weight_dim_2,
        &conv2d_bias_relu_52_0_dim_1,
        &conv2d_bias_relu_52_0_dim_2,
        &input0_dim_0,
        &conv2d_bias_add_relu_53_0_dim_1,
        &conv2d_bias_add_relu_53_0_dim_2,
        1,
        1,
        0,
        1,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "conv2d_bias_add_relu_53" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"512\"], [\"2048\", \"1\", \"1\", \"512\"], [\"2048\"], [\"1\", \"7\", \"7\", \"2048\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"1\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "avg_pool2d_54" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    avg_pool2d_54(
        conv2d_bias_add_relu_53_0,
        avg_pool2d_54_0,
        &input0_dim_0,
        &conv2d_bias_add_relu_53_0_dim_3,
        &conv2d_bias_add_relu_53_0_dim_1,
        &conv2d_bias_add_relu_53_0_dim_2,
        &input0_dim_0,
        &avg_pool2d_54_0_dim_1,
        &avg_pool2d_54_0_dim_2,
        7,
        7,
        1,
        0,
        threadpool_.get()
    );
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "avg_pool2d_54" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"7\", \"7\", \"2048\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"1\", \"1\", \"2048\"]]"
        
          << ", \"stride\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"kernel_size\": " << "\"7\""
        
          << ", \"reduce_func\": " << "\"avg\""
        
           << " } ";
        
          ss << ",\n";
        
      }
      
      {
        std::cout << "Profiling: " << "gemm_rcr_bias_56" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
    {
    

    gemm_rcr_bias_56(
        avg_pool2d_54_0,
        fc_weight,

        fc_bias,

        output_0,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,


        &fc_weight_dim_0,

        &fc_weight_dim_1,


        &reshape_55_0_dim_0,

        &fc_weight_dim_0,

        threadpool_.get());
    }
            clock_gettime(CLOCK_MONOTONIC, &end);
            milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "gemm_rcr_bias_56" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"1\", \"2048\"], [\"1000\", \"2048\"], [\"1000\"]]"
           << ", \"output_sizes\": " << "[[\"1\", \"1000\"]]"
        
           << " } ";
        
          ss << "\n";
        
      }
      
      ss << "}\n";

      std::cout << "AIT per op profiling finished." << std::endl;
#endif
    }

    static std::unique_ptr<Model> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<Model>(
          7225344,
          0 * (1 + 0),
          0 * (1 + 0),
          1,
          1,
          108,
          constants,
          allocator
      );
    }

  private:
   void* input0 {nullptr};
   void* stem_conv1_weight {nullptr};
   void* stem_conv1_bias {nullptr};
   void* layer1_0_conv1_weight {nullptr};
   void* layer1_0_conv1_bias {nullptr};
   void* layer1_0_conv2_weight {nullptr};
   void* layer1_0_conv2_bias {nullptr};
   void* layer1_0_downsample_0_weight {nullptr};
   void* layer1_0_downsample_0_bias {nullptr};
   void* layer1_0_conv3_weight {nullptr};
   void* layer1_0_conv3_bias {nullptr};
   void* layer1_1_conv1_weight {nullptr};
   void* layer1_1_conv1_bias {nullptr};
   void* layer1_1_conv2_weight {nullptr};
   void* layer1_1_conv2_bias {nullptr};
   void* layer1_1_conv3_weight {nullptr};
   void* layer1_1_conv3_bias {nullptr};
   void* layer1_2_conv1_weight {nullptr};
   void* layer1_2_conv1_bias {nullptr};
   void* layer1_2_conv2_weight {nullptr};
   void* layer1_2_conv2_bias {nullptr};
   void* layer1_2_conv3_weight {nullptr};
   void* layer1_2_conv3_bias {nullptr};
   void* layer2_0_conv1_weight {nullptr};
   void* layer2_0_conv1_bias {nullptr};
   void* layer2_0_conv2_weight {nullptr};
   void* layer2_0_conv2_bias {nullptr};
   void* layer2_0_downsample_0_weight {nullptr};
   void* layer2_0_downsample_0_bias {nullptr};
   void* layer2_0_conv3_weight {nullptr};
   void* layer2_0_conv3_bias {nullptr};
   void* layer2_1_conv1_weight {nullptr};
   void* layer2_1_conv1_bias {nullptr};
   void* layer2_1_conv2_weight {nullptr};
   void* layer2_1_conv2_bias {nullptr};
   void* layer2_1_conv3_weight {nullptr};
   void* layer2_1_conv3_bias {nullptr};
   void* layer2_2_conv1_weight {nullptr};
   void* layer2_2_conv1_bias {nullptr};
   void* layer2_2_conv2_weight {nullptr};
   void* layer2_2_conv2_bias {nullptr};
   void* layer2_2_conv3_weight {nullptr};
   void* layer2_2_conv3_bias {nullptr};
   void* layer2_3_conv1_weight {nullptr};
   void* layer2_3_conv1_bias {nullptr};
   void* layer2_3_conv2_weight {nullptr};
   void* layer2_3_conv2_bias {nullptr};
   void* layer2_3_conv3_weight {nullptr};
   void* layer2_3_conv3_bias {nullptr};
   void* layer3_0_conv1_weight {nullptr};
   void* layer3_0_conv1_bias {nullptr};
   void* layer3_0_conv2_weight {nullptr};
   void* layer3_0_conv2_bias {nullptr};
   void* layer3_0_downsample_0_weight {nullptr};
   void* layer3_0_downsample_0_bias {nullptr};
   void* layer3_0_conv3_weight {nullptr};
   void* layer3_0_conv3_bias {nullptr};
   void* layer3_1_conv1_weight {nullptr};
   void* layer3_1_conv1_bias {nullptr};
   void* layer3_1_conv2_weight {nullptr};
   void* layer3_1_conv2_bias {nullptr};
   void* layer3_1_conv3_weight {nullptr};
   void* layer3_1_conv3_bias {nullptr};
   void* layer3_2_conv1_weight {nullptr};
   void* layer3_2_conv1_bias {nullptr};
   void* layer3_2_conv2_weight {nullptr};
   void* layer3_2_conv2_bias {nullptr};
   void* layer3_2_conv3_weight {nullptr};
   void* layer3_2_conv3_bias {nullptr};
   void* layer3_3_conv1_weight {nullptr};
   void* layer3_3_conv1_bias {nullptr};
   void* layer3_3_conv2_weight {nullptr};
   void* layer3_3_conv2_bias {nullptr};
   void* layer3_3_conv3_weight {nullptr};
   void* layer3_3_conv3_bias {nullptr};
   void* layer3_4_conv1_weight {nullptr};
   void* layer3_4_conv1_bias {nullptr};
   void* layer3_4_conv2_weight {nullptr};
   void* layer3_4_conv2_bias {nullptr};
   void* layer3_4_conv3_weight {nullptr};
   void* layer3_4_conv3_bias {nullptr};
   void* layer3_5_conv1_weight {nullptr};
   void* layer3_5_conv1_bias {nullptr};
   void* layer3_5_conv2_weight {nullptr};
   void* layer3_5_conv2_bias {nullptr};
   void* layer3_5_conv3_weight {nullptr};
   void* layer3_5_conv3_bias {nullptr};
   void* layer4_0_conv1_weight {nullptr};
   void* layer4_0_conv1_bias {nullptr};
   void* layer4_0_conv2_weight {nullptr};
   void* layer4_0_conv2_bias {nullptr};
   void* layer4_0_downsample_0_weight {nullptr};
   void* layer4_0_downsample_0_bias {nullptr};
   void* layer4_0_conv3_weight {nullptr};
   void* layer4_0_conv3_bias {nullptr};
   void* layer4_1_conv1_weight {nullptr};
   void* layer4_1_conv1_bias {nullptr};
   void* layer4_1_conv2_weight {nullptr};
   void* layer4_1_conv2_bias {nullptr};
   void* layer4_1_conv3_weight {nullptr};
   void* layer4_1_conv3_bias {nullptr};
   void* layer4_2_conv1_weight {nullptr};
   void* layer4_2_conv1_bias {nullptr};
   void* layer4_2_conv2_weight {nullptr};
   void* layer4_2_conv2_bias {nullptr};
   void* layer4_2_conv3_weight {nullptr};
   void* layer4_2_conv3_bias {nullptr};
   void* fc_weight {nullptr};
   void* fc_bias {nullptr};
   void* conv2d_bias_relu_0_0 {nullptr};
   void* max_pool2d_1_0 {nullptr};
   void* conv2d_bias_2_0 {nullptr};
   void* conv2d_bias_relu_3_0 {nullptr};
   void* conv2d_bias_relu_4_0 {nullptr};
   void* conv2d_bias_add_relu_5_0 {nullptr};
   void* conv2d_bias_relu_6_0 {nullptr};
   void* conv2d_bias_relu_7_0 {nullptr};
   void* conv2d_bias_add_relu_8_0 {nullptr};
   void* conv2d_bias_relu_9_0 {nullptr};
   void* conv2d_bias_relu_10_0 {nullptr};
   void* conv2d_bias_add_relu_11_0 {nullptr};
   void* conv2d_bias_relu_12_0 {nullptr};
   void* conv2d_bias_13_0 {nullptr};
   void* conv2d_bias_relu_14_0 {nullptr};
   void* conv2d_bias_add_relu_15_0 {nullptr};
   void* conv2d_bias_relu_16_0 {nullptr};
   void* conv2d_bias_relu_17_0 {nullptr};
   void* conv2d_bias_add_relu_18_0 {nullptr};
   void* conv2d_bias_relu_19_0 {nullptr};
   void* conv2d_bias_relu_20_0 {nullptr};
   void* conv2d_bias_add_relu_21_0 {nullptr};
   void* conv2d_bias_relu_22_0 {nullptr};
   void* conv2d_bias_relu_23_0 {nullptr};
   void* conv2d_bias_add_relu_24_0 {nullptr};
   void* conv2d_bias_relu_25_0 {nullptr};
   void* conv2d_bias_26_0 {nullptr};
   void* conv2d_bias_relu_27_0 {nullptr};
   void* conv2d_bias_add_relu_28_0 {nullptr};
   void* conv2d_bias_relu_29_0 {nullptr};
   void* conv2d_bias_relu_30_0 {nullptr};
   void* conv2d_bias_add_relu_31_0 {nullptr};
   void* conv2d_bias_relu_32_0 {nullptr};
   void* conv2d_bias_relu_33_0 {nullptr};
   void* conv2d_bias_add_relu_34_0 {nullptr};
   void* conv2d_bias_relu_35_0 {nullptr};
   void* conv2d_bias_relu_36_0 {nullptr};
   void* conv2d_bias_add_relu_37_0 {nullptr};
   void* conv2d_bias_relu_38_0 {nullptr};
   void* conv2d_bias_relu_39_0 {nullptr};
   void* conv2d_bias_add_relu_40_0 {nullptr};
   void* conv2d_bias_relu_41_0 {nullptr};
   void* conv2d_bias_relu_42_0 {nullptr};
   void* conv2d_bias_add_relu_43_0 {nullptr};
   void* conv2d_bias_relu_44_0 {nullptr};
   void* conv2d_bias_45_0 {nullptr};
   void* conv2d_bias_relu_46_0 {nullptr};
   void* conv2d_bias_add_relu_47_0 {nullptr};
   void* conv2d_bias_relu_48_0 {nullptr};
   void* conv2d_bias_relu_49_0 {nullptr};
   void* conv2d_bias_add_relu_50_0 {nullptr};
   void* conv2d_bias_relu_51_0 {nullptr};
   void* conv2d_bias_relu_52_0 {nullptr};
   void* conv2d_bias_add_relu_53_0 {nullptr};
   void* avg_pool2d_54_0 {nullptr};
   void* output_0 {nullptr};
   int64_t input0_dim_0 { 1 };
   int64_t input0_dim_1 { 224 };
   int64_t input0_dim_2 { 224 };
   int64_t input0_dim_3 { 3 };
   int64_t stem_conv1_weight_dim_0 { 64 };
   int64_t stem_conv1_weight_dim_1 { 7 };
   int64_t stem_conv1_weight_dim_2 { 7 };
   int64_t stem_conv1_weight_dim_3 { 3 };
   int64_t stem_conv1_bias_dim_0 { 64 };
   int64_t layer1_0_conv1_weight_dim_0 { 64 };
   int64_t layer1_0_conv1_weight_dim_1 { 1 };
   int64_t layer1_0_conv1_weight_dim_2 { 1 };
   int64_t layer1_0_conv1_weight_dim_3 { 64 };
   int64_t layer1_0_conv1_bias_dim_0 { 64 };
   int64_t layer1_0_conv2_weight_dim_0 { 64 };
   int64_t layer1_0_conv2_weight_dim_1 { 3 };
   int64_t layer1_0_conv2_weight_dim_2 { 3 };
   int64_t layer1_0_conv2_weight_dim_3 { 64 };
   int64_t layer1_0_conv2_bias_dim_0 { 64 };
   int64_t layer1_0_downsample_0_weight_dim_0 { 256 };
   int64_t layer1_0_downsample_0_weight_dim_1 { 1 };
   int64_t layer1_0_downsample_0_weight_dim_2 { 1 };
   int64_t layer1_0_downsample_0_weight_dim_3 { 64 };
   int64_t layer1_0_downsample_0_bias_dim_0 { 256 };
   int64_t layer1_0_conv3_weight_dim_0 { 256 };
   int64_t layer1_0_conv3_weight_dim_1 { 1 };
   int64_t layer1_0_conv3_weight_dim_2 { 1 };
   int64_t layer1_0_conv3_weight_dim_3 { 64 };
   int64_t layer1_0_conv3_bias_dim_0 { 256 };
   int64_t layer1_1_conv1_weight_dim_0 { 64 };
   int64_t layer1_1_conv1_weight_dim_1 { 1 };
   int64_t layer1_1_conv1_weight_dim_2 { 1 };
   int64_t layer1_1_conv1_weight_dim_3 { 256 };
   int64_t layer1_1_conv1_bias_dim_0 { 64 };
   int64_t layer1_1_conv2_weight_dim_0 { 64 };
   int64_t layer1_1_conv2_weight_dim_1 { 3 };
   int64_t layer1_1_conv2_weight_dim_2 { 3 };
   int64_t layer1_1_conv2_weight_dim_3 { 64 };
   int64_t layer1_1_conv2_bias_dim_0 { 64 };
   int64_t layer1_1_conv3_weight_dim_0 { 256 };
   int64_t layer1_1_conv3_weight_dim_1 { 1 };
   int64_t layer1_1_conv3_weight_dim_2 { 1 };
   int64_t layer1_1_conv3_weight_dim_3 { 64 };
   int64_t layer1_1_conv3_bias_dim_0 { 256 };
   int64_t layer1_2_conv1_weight_dim_0 { 64 };
   int64_t layer1_2_conv1_weight_dim_1 { 1 };
   int64_t layer1_2_conv1_weight_dim_2 { 1 };
   int64_t layer1_2_conv1_weight_dim_3 { 256 };
   int64_t layer1_2_conv1_bias_dim_0 { 64 };
   int64_t layer1_2_conv2_weight_dim_0 { 64 };
   int64_t layer1_2_conv2_weight_dim_1 { 3 };
   int64_t layer1_2_conv2_weight_dim_2 { 3 };
   int64_t layer1_2_conv2_weight_dim_3 { 64 };
   int64_t layer1_2_conv2_bias_dim_0 { 64 };
   int64_t layer1_2_conv3_weight_dim_0 { 256 };
   int64_t layer1_2_conv3_weight_dim_1 { 1 };
   int64_t layer1_2_conv3_weight_dim_2 { 1 };
   int64_t layer1_2_conv3_weight_dim_3 { 64 };
   int64_t layer1_2_conv3_bias_dim_0 { 256 };
   int64_t layer2_0_conv1_weight_dim_0 { 128 };
   int64_t layer2_0_conv1_weight_dim_1 { 1 };
   int64_t layer2_0_conv1_weight_dim_2 { 1 };
   int64_t layer2_0_conv1_weight_dim_3 { 256 };
   int64_t layer2_0_conv1_bias_dim_0 { 128 };
   int64_t layer2_0_conv2_weight_dim_0 { 128 };
   int64_t layer2_0_conv2_weight_dim_1 { 3 };
   int64_t layer2_0_conv2_weight_dim_2 { 3 };
   int64_t layer2_0_conv2_weight_dim_3 { 128 };
   int64_t layer2_0_conv2_bias_dim_0 { 128 };
   int64_t layer2_0_downsample_0_weight_dim_0 { 512 };
   int64_t layer2_0_downsample_0_weight_dim_1 { 1 };
   int64_t layer2_0_downsample_0_weight_dim_2 { 1 };
   int64_t layer2_0_downsample_0_weight_dim_3 { 256 };
   int64_t layer2_0_downsample_0_bias_dim_0 { 512 };
   int64_t layer2_0_conv3_weight_dim_0 { 512 };
   int64_t layer2_0_conv3_weight_dim_1 { 1 };
   int64_t layer2_0_conv3_weight_dim_2 { 1 };
   int64_t layer2_0_conv3_weight_dim_3 { 128 };
   int64_t layer2_0_conv3_bias_dim_0 { 512 };
   int64_t layer2_1_conv1_weight_dim_0 { 128 };
   int64_t layer2_1_conv1_weight_dim_1 { 1 };
   int64_t layer2_1_conv1_weight_dim_2 { 1 };
   int64_t layer2_1_conv1_weight_dim_3 { 512 };
   int64_t layer2_1_conv1_bias_dim_0 { 128 };
   int64_t layer2_1_conv2_weight_dim_0 { 128 };
   int64_t layer2_1_conv2_weight_dim_1 { 3 };
   int64_t layer2_1_conv2_weight_dim_2 { 3 };
   int64_t layer2_1_conv2_weight_dim_3 { 128 };
   int64_t layer2_1_conv2_bias_dim_0 { 128 };
   int64_t layer2_1_conv3_weight_dim_0 { 512 };
   int64_t layer2_1_conv3_weight_dim_1 { 1 };
   int64_t layer2_1_conv3_weight_dim_2 { 1 };
   int64_t layer2_1_conv3_weight_dim_3 { 128 };
   int64_t layer2_1_conv3_bias_dim_0 { 512 };
   int64_t layer2_2_conv1_weight_dim_0 { 128 };
   int64_t layer2_2_conv1_weight_dim_1 { 1 };
   int64_t layer2_2_conv1_weight_dim_2 { 1 };
   int64_t layer2_2_conv1_weight_dim_3 { 512 };
   int64_t layer2_2_conv1_bias_dim_0 { 128 };
   int64_t layer2_2_conv2_weight_dim_0 { 128 };
   int64_t layer2_2_conv2_weight_dim_1 { 3 };
   int64_t layer2_2_conv2_weight_dim_2 { 3 };
   int64_t layer2_2_conv2_weight_dim_3 { 128 };
   int64_t layer2_2_conv2_bias_dim_0 { 128 };
   int64_t layer2_2_conv3_weight_dim_0 { 512 };
   int64_t layer2_2_conv3_weight_dim_1 { 1 };
   int64_t layer2_2_conv3_weight_dim_2 { 1 };
   int64_t layer2_2_conv3_weight_dim_3 { 128 };
   int64_t layer2_2_conv3_bias_dim_0 { 512 };
   int64_t layer2_3_conv1_weight_dim_0 { 128 };
   int64_t layer2_3_conv1_weight_dim_1 { 1 };
   int64_t layer2_3_conv1_weight_dim_2 { 1 };
   int64_t layer2_3_conv1_weight_dim_3 { 512 };
   int64_t layer2_3_conv1_bias_dim_0 { 128 };
   int64_t layer2_3_conv2_weight_dim_0 { 128 };
   int64_t layer2_3_conv2_weight_dim_1 { 3 };
   int64_t layer2_3_conv2_weight_dim_2 { 3 };
   int64_t layer2_3_conv2_weight_dim_3 { 128 };
   int64_t layer2_3_conv2_bias_dim_0 { 128 };
   int64_t layer2_3_conv3_weight_dim_0 { 512 };
   int64_t layer2_3_conv3_weight_dim_1 { 1 };
   int64_t layer2_3_conv3_weight_dim_2 { 1 };
   int64_t layer2_3_conv3_weight_dim_3 { 128 };
   int64_t layer2_3_conv3_bias_dim_0 { 512 };
   int64_t layer3_0_conv1_weight_dim_0 { 256 };
   int64_t layer3_0_conv1_weight_dim_1 { 1 };
   int64_t layer3_0_conv1_weight_dim_2 { 1 };
   int64_t layer3_0_conv1_weight_dim_3 { 512 };
   int64_t layer3_0_conv1_bias_dim_0 { 256 };
   int64_t layer3_0_conv2_weight_dim_0 { 256 };
   int64_t layer3_0_conv2_weight_dim_1 { 3 };
   int64_t layer3_0_conv2_weight_dim_2 { 3 };
   int64_t layer3_0_conv2_weight_dim_3 { 256 };
   int64_t layer3_0_conv2_bias_dim_0 { 256 };
   int64_t layer3_0_downsample_0_weight_dim_0 { 1024 };
   int64_t layer3_0_downsample_0_weight_dim_1 { 1 };
   int64_t layer3_0_downsample_0_weight_dim_2 { 1 };
   int64_t layer3_0_downsample_0_weight_dim_3 { 512 };
   int64_t layer3_0_downsample_0_bias_dim_0 { 1024 };
   int64_t layer3_0_conv3_weight_dim_0 { 1024 };
   int64_t layer3_0_conv3_weight_dim_1 { 1 };
   int64_t layer3_0_conv3_weight_dim_2 { 1 };
   int64_t layer3_0_conv3_weight_dim_3 { 256 };
   int64_t layer3_0_conv3_bias_dim_0 { 1024 };
   int64_t layer3_1_conv1_weight_dim_0 { 256 };
   int64_t layer3_1_conv1_weight_dim_1 { 1 };
   int64_t layer3_1_conv1_weight_dim_2 { 1 };
   int64_t layer3_1_conv1_weight_dim_3 { 1024 };
   int64_t layer3_1_conv1_bias_dim_0 { 256 };
   int64_t layer3_1_conv2_weight_dim_0 { 256 };
   int64_t layer3_1_conv2_weight_dim_1 { 3 };
   int64_t layer3_1_conv2_weight_dim_2 { 3 };
   int64_t layer3_1_conv2_weight_dim_3 { 256 };
   int64_t layer3_1_conv2_bias_dim_0 { 256 };
   int64_t layer3_1_conv3_weight_dim_0 { 1024 };
   int64_t layer3_1_conv3_weight_dim_1 { 1 };
   int64_t layer3_1_conv3_weight_dim_2 { 1 };
   int64_t layer3_1_conv3_weight_dim_3 { 256 };
   int64_t layer3_1_conv3_bias_dim_0 { 1024 };
   int64_t layer3_2_conv1_weight_dim_0 { 256 };
   int64_t layer3_2_conv1_weight_dim_1 { 1 };
   int64_t layer3_2_conv1_weight_dim_2 { 1 };
   int64_t layer3_2_conv1_weight_dim_3 { 1024 };
   int64_t layer3_2_conv1_bias_dim_0 { 256 };
   int64_t layer3_2_conv2_weight_dim_0 { 256 };
   int64_t layer3_2_conv2_weight_dim_1 { 3 };
   int64_t layer3_2_conv2_weight_dim_2 { 3 };
   int64_t layer3_2_conv2_weight_dim_3 { 256 };
   int64_t layer3_2_conv2_bias_dim_0 { 256 };
   int64_t layer3_2_conv3_weight_dim_0 { 1024 };
   int64_t layer3_2_conv3_weight_dim_1 { 1 };
   int64_t layer3_2_conv3_weight_dim_2 { 1 };
   int64_t layer3_2_conv3_weight_dim_3 { 256 };
   int64_t layer3_2_conv3_bias_dim_0 { 1024 };
   int64_t layer3_3_conv1_weight_dim_0 { 256 };
   int64_t layer3_3_conv1_weight_dim_1 { 1 };
   int64_t layer3_3_conv1_weight_dim_2 { 1 };
   int64_t layer3_3_conv1_weight_dim_3 { 1024 };
   int64_t layer3_3_conv1_bias_dim_0 { 256 };
   int64_t layer3_3_conv2_weight_dim_0 { 256 };
   int64_t layer3_3_conv2_weight_dim_1 { 3 };
   int64_t layer3_3_conv2_weight_dim_2 { 3 };
   int64_t layer3_3_conv2_weight_dim_3 { 256 };
   int64_t layer3_3_conv2_bias_dim_0 { 256 };
   int64_t layer3_3_conv3_weight_dim_0 { 1024 };
   int64_t layer3_3_conv3_weight_dim_1 { 1 };
   int64_t layer3_3_conv3_weight_dim_2 { 1 };
   int64_t layer3_3_conv3_weight_dim_3 { 256 };
   int64_t layer3_3_conv3_bias_dim_0 { 1024 };
   int64_t layer3_4_conv1_weight_dim_0 { 256 };
   int64_t layer3_4_conv1_weight_dim_1 { 1 };
   int64_t layer3_4_conv1_weight_dim_2 { 1 };
   int64_t layer3_4_conv1_weight_dim_3 { 1024 };
   int64_t layer3_4_conv1_bias_dim_0 { 256 };
   int64_t layer3_4_conv2_weight_dim_0 { 256 };
   int64_t layer3_4_conv2_weight_dim_1 { 3 };
   int64_t layer3_4_conv2_weight_dim_2 { 3 };
   int64_t layer3_4_conv2_weight_dim_3 { 256 };
   int64_t layer3_4_conv2_bias_dim_0 { 256 };
   int64_t layer3_4_conv3_weight_dim_0 { 1024 };
   int64_t layer3_4_conv3_weight_dim_1 { 1 };
   int64_t layer3_4_conv3_weight_dim_2 { 1 };
   int64_t layer3_4_conv3_weight_dim_3 { 256 };
   int64_t layer3_4_conv3_bias_dim_0 { 1024 };
   int64_t layer3_5_conv1_weight_dim_0 { 256 };
   int64_t layer3_5_conv1_weight_dim_1 { 1 };
   int64_t layer3_5_conv1_weight_dim_2 { 1 };
   int64_t layer3_5_conv1_weight_dim_3 { 1024 };
   int64_t layer3_5_conv1_bias_dim_0 { 256 };
   int64_t layer3_5_conv2_weight_dim_0 { 256 };
   int64_t layer3_5_conv2_weight_dim_1 { 3 };
   int64_t layer3_5_conv2_weight_dim_2 { 3 };
   int64_t layer3_5_conv2_weight_dim_3 { 256 };
   int64_t layer3_5_conv2_bias_dim_0 { 256 };
   int64_t layer3_5_conv3_weight_dim_0 { 1024 };
   int64_t layer3_5_conv3_weight_dim_1 { 1 };
   int64_t layer3_5_conv3_weight_dim_2 { 1 };
   int64_t layer3_5_conv3_weight_dim_3 { 256 };
   int64_t layer3_5_conv3_bias_dim_0 { 1024 };
   int64_t layer4_0_conv1_weight_dim_0 { 512 };
   int64_t layer4_0_conv1_weight_dim_1 { 1 };
   int64_t layer4_0_conv1_weight_dim_2 { 1 };
   int64_t layer4_0_conv1_weight_dim_3 { 1024 };
   int64_t layer4_0_conv1_bias_dim_0 { 512 };
   int64_t layer4_0_conv2_weight_dim_0 { 512 };
   int64_t layer4_0_conv2_weight_dim_1 { 3 };
   int64_t layer4_0_conv2_weight_dim_2 { 3 };
   int64_t layer4_0_conv2_weight_dim_3 { 512 };
   int64_t layer4_0_conv2_bias_dim_0 { 512 };
   int64_t layer4_0_downsample_0_weight_dim_0 { 2048 };
   int64_t layer4_0_downsample_0_weight_dim_1 { 1 };
   int64_t layer4_0_downsample_0_weight_dim_2 { 1 };
   int64_t layer4_0_downsample_0_weight_dim_3 { 1024 };
   int64_t layer4_0_downsample_0_bias_dim_0 { 2048 };
   int64_t layer4_0_conv3_weight_dim_0 { 2048 };
   int64_t layer4_0_conv3_weight_dim_1 { 1 };
   int64_t layer4_0_conv3_weight_dim_2 { 1 };
   int64_t layer4_0_conv3_weight_dim_3 { 512 };
   int64_t layer4_0_conv3_bias_dim_0 { 2048 };
   int64_t layer4_1_conv1_weight_dim_0 { 512 };
   int64_t layer4_1_conv1_weight_dim_1 { 1 };
   int64_t layer4_1_conv1_weight_dim_2 { 1 };
   int64_t layer4_1_conv1_weight_dim_3 { 2048 };
   int64_t layer4_1_conv1_bias_dim_0 { 512 };
   int64_t layer4_1_conv2_weight_dim_0 { 512 };
   int64_t layer4_1_conv2_weight_dim_1 { 3 };
   int64_t layer4_1_conv2_weight_dim_2 { 3 };
   int64_t layer4_1_conv2_weight_dim_3 { 512 };
   int64_t layer4_1_conv2_bias_dim_0 { 512 };
   int64_t layer4_1_conv3_weight_dim_0 { 2048 };
   int64_t layer4_1_conv3_weight_dim_1 { 1 };
   int64_t layer4_1_conv3_weight_dim_2 { 1 };
   int64_t layer4_1_conv3_weight_dim_3 { 512 };
   int64_t layer4_1_conv3_bias_dim_0 { 2048 };
   int64_t layer4_2_conv1_weight_dim_0 { 512 };
   int64_t layer4_2_conv1_weight_dim_1 { 1 };
   int64_t layer4_2_conv1_weight_dim_2 { 1 };
   int64_t layer4_2_conv1_weight_dim_3 { 2048 };
   int64_t layer4_2_conv1_bias_dim_0 { 512 };
   int64_t layer4_2_conv2_weight_dim_0 { 512 };
   int64_t layer4_2_conv2_weight_dim_1 { 3 };
   int64_t layer4_2_conv2_weight_dim_2 { 3 };
   int64_t layer4_2_conv2_weight_dim_3 { 512 };
   int64_t layer4_2_conv2_bias_dim_0 { 512 };
   int64_t layer4_2_conv3_weight_dim_0 { 2048 };
   int64_t layer4_2_conv3_weight_dim_1 { 1 };
   int64_t layer4_2_conv3_weight_dim_2 { 1 };
   int64_t layer4_2_conv3_weight_dim_3 { 512 };
   int64_t layer4_2_conv3_bias_dim_0 { 2048 };
   int64_t fc_weight_dim_0 { 1000 };
   int64_t fc_weight_dim_1 { 2048 };
   int64_t fc_bias_dim_0 { 1000 };
   int64_t conv2d_bias_relu_0_0_dim_1 { 112 };
   int64_t conv2d_bias_relu_0_0_dim_2 { 112 };
   int64_t conv2d_bias_relu_0_0_dim_3 { 64 };
   int64_t max_pool2d_1_0_dim_1 { 56 };
   int64_t max_pool2d_1_0_dim_2 { 56 };
   int64_t max_pool2d_1_0_dim_3 { 64 };
   int64_t conv2d_bias_2_0_dim_1 { 56 };
   int64_t conv2d_bias_2_0_dim_2 { 56 };
   int64_t conv2d_bias_2_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_3_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_3_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_3_0_dim_3 { 64 };
   int64_t conv2d_bias_relu_4_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_4_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_4_0_dim_3 { 64 };
   int64_t conv2d_bias_add_relu_5_0_dim_1 { 56 };
   int64_t conv2d_bias_add_relu_5_0_dim_2 { 56 };
   int64_t conv2d_bias_add_relu_5_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_6_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_6_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_6_0_dim_3 { 64 };
   int64_t conv2d_bias_relu_7_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_7_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_7_0_dim_3 { 64 };
   int64_t conv2d_bias_add_relu_8_0_dim_1 { 56 };
   int64_t conv2d_bias_add_relu_8_0_dim_2 { 56 };
   int64_t conv2d_bias_add_relu_8_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_9_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_9_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_9_0_dim_3 { 64 };
   int64_t conv2d_bias_relu_10_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_10_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_10_0_dim_3 { 64 };
   int64_t conv2d_bias_add_relu_11_0_dim_1 { 56 };
   int64_t conv2d_bias_add_relu_11_0_dim_2 { 56 };
   int64_t conv2d_bias_add_relu_11_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_12_0_dim_1 { 56 };
   int64_t conv2d_bias_relu_12_0_dim_2 { 56 };
   int64_t conv2d_bias_relu_12_0_dim_3 { 128 };
   int64_t conv2d_bias_13_0_dim_1 { 28 };
   int64_t conv2d_bias_13_0_dim_2 { 28 };
   int64_t conv2d_bias_13_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_14_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_14_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_14_0_dim_3 { 128 };
   int64_t conv2d_bias_add_relu_15_0_dim_1 { 28 };
   int64_t conv2d_bias_add_relu_15_0_dim_2 { 28 };
   int64_t conv2d_bias_add_relu_15_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_16_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_16_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_16_0_dim_3 { 128 };
   int64_t conv2d_bias_relu_17_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_17_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_17_0_dim_3 { 128 };
   int64_t conv2d_bias_add_relu_18_0_dim_1 { 28 };
   int64_t conv2d_bias_add_relu_18_0_dim_2 { 28 };
   int64_t conv2d_bias_add_relu_18_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_19_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_19_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_19_0_dim_3 { 128 };
   int64_t conv2d_bias_relu_20_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_20_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_20_0_dim_3 { 128 };
   int64_t conv2d_bias_add_relu_21_0_dim_1 { 28 };
   int64_t conv2d_bias_add_relu_21_0_dim_2 { 28 };
   int64_t conv2d_bias_add_relu_21_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_22_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_22_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_22_0_dim_3 { 128 };
   int64_t conv2d_bias_relu_23_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_23_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_23_0_dim_3 { 128 };
   int64_t conv2d_bias_add_relu_24_0_dim_1 { 28 };
   int64_t conv2d_bias_add_relu_24_0_dim_2 { 28 };
   int64_t conv2d_bias_add_relu_24_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_25_0_dim_1 { 28 };
   int64_t conv2d_bias_relu_25_0_dim_2 { 28 };
   int64_t conv2d_bias_relu_25_0_dim_3 { 256 };
   int64_t conv2d_bias_26_0_dim_1 { 14 };
   int64_t conv2d_bias_26_0_dim_2 { 14 };
   int64_t conv2d_bias_26_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_27_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_27_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_27_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_28_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_28_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_28_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_29_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_29_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_29_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_30_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_30_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_30_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_31_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_31_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_31_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_32_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_32_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_32_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_33_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_33_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_33_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_34_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_34_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_34_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_35_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_35_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_35_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_36_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_36_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_36_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_37_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_37_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_37_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_38_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_38_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_38_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_39_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_39_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_39_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_40_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_40_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_40_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_41_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_41_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_41_0_dim_3 { 256 };
   int64_t conv2d_bias_relu_42_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_42_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_42_0_dim_3 { 256 };
   int64_t conv2d_bias_add_relu_43_0_dim_1 { 14 };
   int64_t conv2d_bias_add_relu_43_0_dim_2 { 14 };
   int64_t conv2d_bias_add_relu_43_0_dim_3 { 1024 };
   int64_t conv2d_bias_relu_44_0_dim_1 { 14 };
   int64_t conv2d_bias_relu_44_0_dim_2 { 14 };
   int64_t conv2d_bias_relu_44_0_dim_3 { 512 };
   int64_t conv2d_bias_45_0_dim_1 { 7 };
   int64_t conv2d_bias_45_0_dim_2 { 7 };
   int64_t conv2d_bias_45_0_dim_3 { 2048 };
   int64_t conv2d_bias_relu_46_0_dim_1 { 7 };
   int64_t conv2d_bias_relu_46_0_dim_2 { 7 };
   int64_t conv2d_bias_relu_46_0_dim_3 { 512 };
   int64_t conv2d_bias_add_relu_47_0_dim_1 { 7 };
   int64_t conv2d_bias_add_relu_47_0_dim_2 { 7 };
   int64_t conv2d_bias_add_relu_47_0_dim_3 { 2048 };
   int64_t conv2d_bias_relu_48_0_dim_1 { 7 };
   int64_t conv2d_bias_relu_48_0_dim_2 { 7 };
   int64_t conv2d_bias_relu_48_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_49_0_dim_1 { 7 };
   int64_t conv2d_bias_relu_49_0_dim_2 { 7 };
   int64_t conv2d_bias_relu_49_0_dim_3 { 512 };
   int64_t conv2d_bias_add_relu_50_0_dim_1 { 7 };
   int64_t conv2d_bias_add_relu_50_0_dim_2 { 7 };
   int64_t conv2d_bias_add_relu_50_0_dim_3 { 2048 };
   int64_t conv2d_bias_relu_51_0_dim_1 { 7 };
   int64_t conv2d_bias_relu_51_0_dim_2 { 7 };
   int64_t conv2d_bias_relu_51_0_dim_3 { 512 };
   int64_t conv2d_bias_relu_52_0_dim_1 { 7 };
   int64_t conv2d_bias_relu_52_0_dim_2 { 7 };
   int64_t conv2d_bias_relu_52_0_dim_3 { 512 };
   int64_t conv2d_bias_add_relu_53_0_dim_1 { 7 };
   int64_t conv2d_bias_add_relu_53_0_dim_2 { 7 };
   int64_t conv2d_bias_add_relu_53_0_dim_3 { 2048 };
   int64_t avg_pool2d_54_0_dim_1 { 1 };
   int64_t avg_pool2d_54_0_dim_2 { 1 };
   int64_t avg_pool2d_54_0_dim_3 { 2048 };
   int64_t reshape_55_0_dim_0 { 1 };
   int64_t output_0_dim_1 { 1 };
   int64_t output_0_dim_2 { 1 };
   int64_t reshape_55_0_dim_1 { 2048 };


std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};
} // namespace ait