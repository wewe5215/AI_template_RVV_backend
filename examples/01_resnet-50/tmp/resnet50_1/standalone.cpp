//  Copyright (c) Meta Platforms, Inc. and affiliates.

//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

// This file is used for generating a standalone executable for a model.
// It only invokes the C++ model interface. We can directly invoke the
// generated executable without going through Python bindings. Because it
// aims for assisting debugging, we make a number of simplifications:
//   * we use the maximum input shapes;
//   * we only generate random inputs with a fixed seed;
//   * we assume that outputs exist on the host;
//   * we disable graph_mode;
//   * etc...
// Once the file is copied into the intemediate working dir (e.g.,
// ./tmp/test_gemm_rcr) along with other files, users are free to make any
// changes to the code. We do not try to predict users' actions.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include "macros.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "make_random_data.h"
#include "model_container.h"
using namespace ait;
std::vector<const char*> param_names_;
std::vector<std::vector<int64_t>> max_param_shapes_;
void setup_constant(){
  param_names_.resize(110);
  max_param_shapes_.resize(110);
  param_names_[0] = "input0";
  param_names_[2] = "stem_conv1_weight";
  param_names_[3] = "stem_conv1_bias";
  param_names_[4] = "layer1_0_conv1_weight";
  param_names_[5] = "layer1_0_conv1_bias";
  param_names_[6] = "layer1_0_conv2_weight";
  param_names_[7] = "layer1_0_conv2_bias";
  param_names_[8] = "layer1_0_downsample_0_weight";
  param_names_[9] = "layer1_0_downsample_0_bias";
  param_names_[10] = "layer1_0_conv3_weight";
  param_names_[11] = "layer1_0_conv3_bias";
  param_names_[12] = "layer1_1_conv1_weight";
  param_names_[13] = "layer1_1_conv1_bias";
  param_names_[14] = "layer1_1_conv2_weight";
  param_names_[15] = "layer1_1_conv2_bias";
  param_names_[16] = "layer1_1_conv3_weight";
  param_names_[17] = "layer1_1_conv3_bias";
  param_names_[18] = "layer1_2_conv1_weight";
  param_names_[19] = "layer1_2_conv1_bias";
  param_names_[20] = "layer1_2_conv2_weight";
  param_names_[21] = "layer1_2_conv2_bias";
  param_names_[22] = "layer1_2_conv3_weight";
  param_names_[23] = "layer1_2_conv3_bias";
  param_names_[24] = "layer2_0_conv1_weight";
  param_names_[25] = "layer2_0_conv1_bias";
  param_names_[26] = "layer2_0_conv2_weight";
  param_names_[27] = "layer2_0_conv2_bias";
  param_names_[28] = "layer2_0_downsample_0_weight";
  param_names_[29] = "layer2_0_downsample_0_bias";
  param_names_[30] = "layer2_0_conv3_weight";
  param_names_[31] = "layer2_0_conv3_bias";
  param_names_[32] = "layer2_1_conv1_weight";
  param_names_[33] = "layer2_1_conv1_bias";
  param_names_[34] = "layer2_1_conv2_weight";
  param_names_[35] = "layer2_1_conv2_bias";
  param_names_[36] = "layer2_1_conv3_weight";
  param_names_[37] = "layer2_1_conv3_bias";
  param_names_[38] = "layer2_2_conv1_weight";
  param_names_[39] = "layer2_2_conv1_bias";
  param_names_[40] = "layer2_2_conv2_weight";
  param_names_[41] = "layer2_2_conv2_bias";
  param_names_[42] = "layer2_2_conv3_weight";
  param_names_[43] = "layer2_2_conv3_bias";
  param_names_[44] = "layer2_3_conv1_weight";
  param_names_[45] = "layer2_3_conv1_bias";
  param_names_[46] = "layer2_3_conv2_weight";
  param_names_[47] = "layer2_3_conv2_bias";
  param_names_[48] = "layer2_3_conv3_weight";
  param_names_[49] = "layer2_3_conv3_bias";
  param_names_[50] = "layer3_0_conv1_weight";
  param_names_[51] = "layer3_0_conv1_bias";
  param_names_[52] = "layer3_0_conv2_weight";
  param_names_[53] = "layer3_0_conv2_bias";
  param_names_[54] = "layer3_0_downsample_0_weight";
  param_names_[55] = "layer3_0_downsample_0_bias";
  param_names_[56] = "layer3_0_conv3_weight";
  param_names_[57] = "layer3_0_conv3_bias";
  param_names_[58] = "layer3_1_conv1_weight";
  param_names_[59] = "layer3_1_conv1_bias";
  param_names_[60] = "layer3_1_conv2_weight";
  param_names_[61] = "layer3_1_conv2_bias";
  param_names_[62] = "layer3_1_conv3_weight";
  param_names_[63] = "layer3_1_conv3_bias";
  param_names_[64] = "layer3_2_conv1_weight";
  param_names_[65] = "layer3_2_conv1_bias";
  param_names_[66] = "layer3_2_conv2_weight";
  param_names_[67] = "layer3_2_conv2_bias";
  param_names_[68] = "layer3_2_conv3_weight";
  param_names_[69] = "layer3_2_conv3_bias";
  param_names_[70] = "layer3_3_conv1_weight";
  param_names_[71] = "layer3_3_conv1_bias";
  param_names_[72] = "layer3_3_conv2_weight";
  param_names_[73] = "layer3_3_conv2_bias";
  param_names_[74] = "layer3_3_conv3_weight";
  param_names_[75] = "layer3_3_conv3_bias";
  param_names_[76] = "layer3_4_conv1_weight";
  param_names_[77] = "layer3_4_conv1_bias";
  param_names_[78] = "layer3_4_conv2_weight";
  param_names_[79] = "layer3_4_conv2_bias";
  param_names_[80] = "layer3_4_conv3_weight";
  param_names_[81] = "layer3_4_conv3_bias";
  param_names_[82] = "layer3_5_conv1_weight";
  param_names_[83] = "layer3_5_conv1_bias";
  param_names_[84] = "layer3_5_conv2_weight";
  param_names_[85] = "layer3_5_conv2_bias";
  param_names_[86] = "layer3_5_conv3_weight";
  param_names_[87] = "layer3_5_conv3_bias";
  param_names_[88] = "layer4_0_conv1_weight";
  param_names_[89] = "layer4_0_conv1_bias";
  param_names_[90] = "layer4_0_conv2_weight";
  param_names_[91] = "layer4_0_conv2_bias";
  param_names_[92] = "layer4_0_downsample_0_weight";
  param_names_[93] = "layer4_0_downsample_0_bias";
  param_names_[94] = "layer4_0_conv3_weight";
  param_names_[95] = "layer4_0_conv3_bias";
  param_names_[96] = "layer4_1_conv1_weight";
  param_names_[97] = "layer4_1_conv1_bias";
  param_names_[98] = "layer4_1_conv2_weight";
  param_names_[99] = "layer4_1_conv2_bias";
  param_names_[100] = "layer4_1_conv3_weight";
  param_names_[101] = "layer4_1_conv3_bias";
  param_names_[102] = "layer4_2_conv1_weight";
  param_names_[103] = "layer4_2_conv1_bias";
  param_names_[104] = "layer4_2_conv2_weight";
  param_names_[105] = "layer4_2_conv2_bias";
  param_names_[106] = "layer4_2_conv3_weight";
  param_names_[107] = "layer4_2_conv3_bias";
  param_names_[108] = "fc_weight";
  param_names_[109] = "fc_bias";
  param_names_[1] = "output_0";

  max_param_shapes_[0] = {1, 224, 224, 3};
  max_param_shapes_[2] = {64, 7, 7, 3};
  max_param_shapes_[3] = {64};
  max_param_shapes_[4] = {64, 1, 1, 64};
  max_param_shapes_[5] = {64};
  max_param_shapes_[6] = {64, 3, 3, 64};
  max_param_shapes_[7] = {64};
  max_param_shapes_[8] = {256, 1, 1, 64};
  max_param_shapes_[9] = {256};
  max_param_shapes_[10] = {256, 1, 1, 64};
  max_param_shapes_[11] = {256};
  max_param_shapes_[12] = {64, 1, 1, 256};
  max_param_shapes_[13] = {64};
  max_param_shapes_[14] = {64, 3, 3, 64};
  max_param_shapes_[15] = {64};
  max_param_shapes_[16] = {256, 1, 1, 64};
  max_param_shapes_[17] = {256};
  max_param_shapes_[18] = {64, 1, 1, 256};
  max_param_shapes_[19] = {64};
  max_param_shapes_[20] = {64, 3, 3, 64};
  max_param_shapes_[21] = {64};
  max_param_shapes_[22] = {256, 1, 1, 64};
  max_param_shapes_[23] = {256};
  max_param_shapes_[24] = {128, 1, 1, 256};
  max_param_shapes_[25] = {128};
  max_param_shapes_[26] = {128, 3, 3, 128};
  max_param_shapes_[27] = {128};
  max_param_shapes_[28] = {512, 1, 1, 256};
  max_param_shapes_[29] = {512};
  max_param_shapes_[30] = {512, 1, 1, 128};
  max_param_shapes_[31] = {512};
  max_param_shapes_[32] = {128, 1, 1, 512};
  max_param_shapes_[33] = {128};
  max_param_shapes_[34] = {128, 3, 3, 128};
  max_param_shapes_[35] = {128};
  max_param_shapes_[36] = {512, 1, 1, 128};
  max_param_shapes_[37] = {512};
  max_param_shapes_[38] = {128, 1, 1, 512};
  max_param_shapes_[39] = {128};
  max_param_shapes_[40] = {128, 3, 3, 128};
  max_param_shapes_[41] = {128};
  max_param_shapes_[42] = {512, 1, 1, 128};
  max_param_shapes_[43] = {512};
  max_param_shapes_[44] = {128, 1, 1, 512};
  max_param_shapes_[45] = {128};
  max_param_shapes_[46] = {128, 3, 3, 128};
  max_param_shapes_[47] = {128};
  max_param_shapes_[48] = {512, 1, 1, 128};
  max_param_shapes_[49] = {512};
  max_param_shapes_[50] = {256, 1, 1, 512};
  max_param_shapes_[51] = {256};
  max_param_shapes_[52] = {256, 3, 3, 256};
  max_param_shapes_[53] = {256};
  max_param_shapes_[54] = {1024, 1, 1, 512};
  max_param_shapes_[55] = {1024};
  max_param_shapes_[56] = {1024, 1, 1, 256};
  max_param_shapes_[57] = {1024};
  max_param_shapes_[58] = {256, 1, 1, 1024};
  max_param_shapes_[59] = {256};
  max_param_shapes_[60] = {256, 3, 3, 256};
  max_param_shapes_[61] = {256};
  max_param_shapes_[62] = {1024, 1, 1, 256};
  max_param_shapes_[63] = {1024};
  max_param_shapes_[64] = {256, 1, 1, 1024};
  max_param_shapes_[65] = {256};
  max_param_shapes_[66] = {256, 3, 3, 256};
  max_param_shapes_[67] = {256};
  max_param_shapes_[68] = {1024, 1, 1, 256};
  max_param_shapes_[69] = {1024};
  max_param_shapes_[70] = {256, 1, 1, 1024};
  max_param_shapes_[71] = {256};
  max_param_shapes_[72] = {256, 3, 3, 256};
  max_param_shapes_[73] = {256};
  max_param_shapes_[74] = {1024, 1, 1, 256};
  max_param_shapes_[75] = {1024};
  max_param_shapes_[76] = {256, 1, 1, 1024};
  max_param_shapes_[77] = {256};
  max_param_shapes_[78] = {256, 3, 3, 256};
  max_param_shapes_[79] = {256};
  max_param_shapes_[80] = {1024, 1, 1, 256};
  max_param_shapes_[81] = {1024};
  max_param_shapes_[82] = {256, 1, 1, 1024};
  max_param_shapes_[83] = {256};
  max_param_shapes_[84] = {256, 3, 3, 256};
  max_param_shapes_[85] = {256};
  max_param_shapes_[86] = {1024, 1, 1, 256};
  max_param_shapes_[87] = {1024};
  max_param_shapes_[88] = {512, 1, 1, 1024};
  max_param_shapes_[89] = {512};
  max_param_shapes_[90] = {512, 3, 3, 512};
  max_param_shapes_[91] = {512};
  max_param_shapes_[92] = {2048, 1, 1, 1024};
  max_param_shapes_[93] = {2048};
  max_param_shapes_[94] = {2048, 1, 1, 512};
  max_param_shapes_[95] = {2048};
  max_param_shapes_[96] = {512, 1, 1, 2048};
  max_param_shapes_[97] = {512};
  max_param_shapes_[98] = {512, 3, 3, 512};
  max_param_shapes_[99] = {512};
  max_param_shapes_[100] = {2048, 1, 1, 512};
  max_param_shapes_[101] = {2048};
  max_param_shapes_[102] = {512, 1, 1, 2048};
  max_param_shapes_[103] = {512};
  max_param_shapes_[104] = {512, 3, 3, 512};
  max_param_shapes_[105] = {512};
  max_param_shapes_[106] = {2048, 1, 1, 512};
  max_param_shapes_[107] = {2048};
  max_param_shapes_[108] = {1000, 2048};
  max_param_shapes_[109] = {1000};
  max_param_shapes_[1] = {1, 1, 1, 1000};
  return;
}

using OutputDataPtr = std::unique_ptr<void, std::function<void(void*)>>;

struct OutputData {
  OutputData(
      OutputDataPtr& data_in,
      std::unique_ptr<int64_t[]>& shape_ptr_in,
      int shape_size_in,
      int index_in,
      AITemplateDtype dtype_in,
      const char* name_in)
      : data(std::move(data_in)),
        shape_ptr(std::move(shape_ptr_in)),
        shape_size(shape_size_in),
        index(index_in),
        dtype(dtype_in),
        name(name_in) {}

  OutputData(OutputData&& other) noexcept
      : data(std::move(other.data)),
        shape_ptr(std::move(other.shape_ptr)),
        shape_size(other.shape_size),
        index(other.index),
        dtype(other.dtype),
        name(std::move(other.name)) {}

  OutputDataPtr data;
  std::unique_ptr<int64_t[]> shape_ptr;
  int shape_size;
  int index;
  AITemplateDtype dtype;
  std::string name;
};

static AITemplateError run(
    AITemplateModelHandle handle,
    std::vector<OutputData>& outputs) {
  size_t num_outputs = 0;
  AITemplateModelContainerGetNumOutputs(handle, &num_outputs);

  outputs.reserve(num_outputs);
  std::vector<AITData> ait_outputs;
  ait_outputs.reserve(num_outputs);
  std::vector<int64_t*> ait_output_shapes_out;
  ait_output_shapes_out.reserve(num_outputs);

  for (unsigned i = 0; i < num_outputs; i++) {
    const char* name;
    AITemplateModelContainerGetOutputName(handle, i, &name);
    AITemplateParamShape shape;
    AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
    AITemplateDtype dtype;
    AITemplateModelContainerGetOutputDtype(handle, i, &dtype);

    std::unique_ptr<int64_t[]> shape_ptr =
        std::make_unique<int64_t[]>(shape.size);
    ait_output_shapes_out.push_back(shape_ptr.get());
    size_t num_bytes = shape.Numel() * AITemplateDtypeSizeBytes(dtype);
    void* h_data = (void*)malloc(num_bytes);
    ait_outputs.emplace_back(h_data, shape, dtype);
    auto deleter = [](void* data) { free(data); };
    OutputDataPtr h_output_ptr(h_data, deleter);
    outputs.emplace_back(
        h_output_ptr, shape_ptr, (int)shape.size, (int)i, dtype, name);
  }

  size_t num_inputs = 0;
  AITemplateModelContainerGetNumInputs(handle, &num_inputs);
  // Holding unique_ptr(s) that will be auto-released.
  std::vector<Ptr> input_ptrs;
  input_ptrs.reserve(num_inputs);

  std::map<std::string, unsigned> input_name_to_index;
  std::vector<AITData> inputs(num_inputs);
  std::mt19937 rnd_generator(1234);
  // set up the name-to-index map each input
  for (unsigned i = 0; i < num_inputs; i++) {
    const char* name;
    AITemplateModelContainerGetInputName(handle, i, &name);
    input_name_to_index.insert({name, i});
    std::cout << "input: " << name << ", at idx: " << i << "\n";

    AITemplateParamShape shape;
    AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);
    AITemplateDtype dtype;
    AITemplateModelContainerGetInputDtype(handle, i, &dtype);
    // This file aims for helping debugging so we make the code logic
    // simple. Instead of asking the user to pass input names along with
    // shapes, we just use the shape with the largest dimension values
    // to make a random input. Once this code is copied into the test's
    // tmp folder, the person who will be diagnosing the issue could make any
    // changes to the code. We don't force us to predict the user's behavior.
    input_ptrs.emplace_back(
        make_random_data(rnd_generator, shape, dtype));
    inputs[i] = AITData(input_ptrs.back().get(), shape, dtype);
  }

  bool graph_mode = false;
  return AITemplateModelContainerRunWithOutputsOnHost(
      handle,
      inputs.data(),
      num_inputs,
      ait_outputs.data(),
      num_outputs,
      graph_mode,
      ait_output_shapes_out.data());
}

template <typename T>
void read_element(std::ifstream& fh, T& elem) {
  if (!fh.good()) {
    throw std::runtime_error("Input stream is not in good state.");
  }
  fh.read(reinterpret_cast<char*>(&elem), sizeof(T));
  if (fh.fail()) {
    throw std::runtime_error("Failed to read binary data");
  }
}

struct AITStandaloneTestcase {
  std::vector<AITData> expected_outputs;
  std::vector<AITData> host_outputs;
  std::vector<AITData> gpu_outputs;

  std::vector<int64_t*> ait_output_shapes_out;

  std::vector<AITData>
      inputs; // this will be filled the AITData instances for the inputs

  std::vector<int64_t> shape_data_owner;
  std::vector<Ptr> data_owner;

  const std::string test_data_path; // path to test data file
  AITemplateModelHandle& handle;

  float atol;
  float rtol;

  AITStandaloneTestcase(
      const char* test_data_path_,
      AITemplateModelHandle& handle_)
      : handle(handle_),
        test_data_path(test_data_path_) {
    _load();
  }

  void _load() { // relative error tolerance
    size_t num_outputs = 0;
    size_t num_inputs = 0;
    AITemplateModelContainerGetNumInputs(handle, &num_inputs);
    AITemplateModelContainerGetNumOutputs(handle, &num_outputs);
    ait_output_shapes_out.reserve(num_outputs);
    expected_outputs.reserve(num_outputs);
    host_outputs.reserve(num_outputs);
    gpu_outputs.reserve(num_outputs);
    std::ifstream fh(test_data_path);
    read_element(fh, atol); // absolute error tolerance
    read_element(fh, rtol); // relative error tolerance

    data_owner.reserve(num_inputs + num_outputs);
    ait_output_shapes_out.reserve(num_outputs);

    std::map<std::string, unsigned> input_name_to_index;
    size_t total_dim_count =
        0; // the sum of shape.ndims for all input and output tensors
    // calculate total_dim_count
    for (unsigned i = 0; i < num_inputs; i++) {
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);
      total_dim_count += shape.size;
    }
    for (unsigned i = 0; i < num_outputs; i++) {
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
      total_dim_count += shape.size * 2; // allocation required twice
    }
    // this is just a vector that owns the memory for the shape.shape_data
    // values
    shape_data_owner.reserve(total_dim_count);
    size_t shape_offset = 0; // offset into the shape_data_owner array
    for (unsigned i = 0; i < num_inputs; i++) {
      // for each input tensor
      const char* name;
      AITemplateModelContainerGetInputName(handle, i, &name);
      AITemplateDtype dtype;
      AITemplateModelContainerGetInputDtype(handle, i, &dtype);
      size_t dtype_size = AITemplateDtypeSizeBytes(dtype);
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumInputShape(handle, i, &shape);

      input_name_to_index.insert({name, i});
      std::cout << "Loading input: " << name << ", at idx: " << i;

      // Read metadata for test case
      unsigned int read_dtype;
      unsigned int read_dtype_size;
      unsigned int read_ndims;
      size_t read_total_tensor_bytes;
      read_element(fh, read_dtype);
      std::cout << ", dtype=" << read_dtype;
      read_element(fh, read_dtype_size);
      std::cout << ", sizeof(dtype)=" << read_dtype_size;
      read_element(fh, read_ndims);
      std::cout << ", ndims=" << read_ndims;

      if (static_cast<AITemplateDtype>(read_dtype) != dtype) {
        throw std::runtime_error(
            "Mismatch between dtype of input in testcase data and in model");
      }

      if (dtype_size != static_cast<size_t>(read_dtype_size)) {
        throw std::runtime_error(
            "Mismatch between sizeof(dtype) in testcase data and in model");
      }

      // Obtain maximum shape from model and verify the testcase data has valid
      // shape
      if (read_ndims != shape.size) {
        throw std::runtime_error(
            "Mismatch between number of input dimensions in testcase data and in model");
      }
      std::cout << ", shape=(";
      for (unsigned j = 0; j < read_ndims; j++) {
        size_t dim;
        read_element(fh, dim);
        shape_data_owner.push_back(dim);
        std::cout << dim << ", ";
        if (dim > shape.shape_data[j]) {
          throw std::runtime_error(
              "Shape in testcase data exceeds maximum shape.");
        }
      }
      std::cout << ")";

      // Set the shape of the input to the actual, and not the maximum shape.
      // the previous shape.shape_data may not be deleted as it's owned by the
      // model.
      shape.shape_data = shape_data_owner.data() + shape_offset;
      shape_offset += read_ndims; // move offset to the next unused space

      // total number of bytes of tensor raw data
      read_element(fh, read_total_tensor_bytes);

      size_t numel = shape.Numel();
      size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
      std::cout << ", total_tensor_bytes=" << read_total_tensor_bytes
                << " - model expects " << num_bytes << "\n";
      if (num_bytes != read_total_tensor_bytes) {
        throw std::runtime_error("Tensor data total size mismatch.");
      }
      // allocate memory for tensor raw data on host
      void* h_data = (void*)malloc(num_bytes);
      // read tensor raw data from file
      fh.read(reinterpret_cast<char*>(h_data), read_total_tensor_bytes);
      // Allocate corresponding device memory and copy tensor raw data to device
      data_owner.emplace_back(RAII_DeviceMalloc(num_bytes));
      std::memcpy(data_owner.back().get(), h_data, num_bytes);
      free(h_data);
      inputs.push_back(AITData(data_owner.back().get(), shape, dtype));
    }
    std::cout << "Finished loading testcase inputs." << "\n";
    if (fh.peek() == std::ifstream::traits_type::eof()) {
      std::cout << "No expected outputs in testcase." << "\n";
      return;
    }
    if (inputs.size() != num_inputs) {
      throw std::runtime_error("Number of inputs mismatches with expected.");
    }
    // read expected outputs from file
    for (unsigned i = 0; i < num_outputs; i++) {
      // for each input tensor
      const char* name;
      AITemplateModelContainerGetOutputName(handle, i, &name);
      AITemplateDtype dtype;
      AITemplateModelContainerGetOutputDtype(handle, i, &dtype);
      size_t dtype_size = AITemplateDtypeSizeBytes(dtype);
      AITemplateParamShape shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &shape);
      AITemplateParamShape max_shape;
      AITemplateModelContainerGetMaximumOutputShape(handle, i, &max_shape);

      size_t max_numel = shape.Numel();
      size_t max_num_bytes = max_numel * AITemplateDtypeSizeBytes(dtype);

      data_owner.emplace_back(RAII_DeviceMalloc(max_num_bytes));
      gpu_outputs.push_back(
          AITData(data_owner.back().get(), max_shape, dtype));

      std::cout << "Loading expected output: " << name << ", at idx: " << i;

      // Read metadata for test case
      unsigned int read_dtype;
      unsigned int read_dtype_size;
      unsigned int read_ndims;
      size_t read_total_tensor_bytes;
      read_element(fh, read_dtype);
      std::cout << ", dtype=" << read_dtype;
      read_element(fh, read_dtype_size);
      std::cout << ", sizeof(dtype)=" << read_dtype_size;
      read_element(fh, read_ndims);
      std::cout << ", ndims=" << read_ndims;

      if (static_cast<AITemplateDtype>(read_dtype) != dtype) {
        throw std::runtime_error(
            "Mismatch between dtype of input in testcase data and in model");
      }

      if (dtype_size != static_cast<size_t>(read_dtype_size)) {
        throw std::runtime_error(
            "Mismatch between sizeof(dtype) in testcase data and in model");
      }

      // Obtain maximum shape from model and verify the testcase data has valid
      // shape
      if (read_ndims != shape.size) {
        throw std::runtime_error(
            "Mismatch between number of input dimensions in testcase data and in model");
      }
      std::cout << ", shape=(";
      for (unsigned j = 0; j < read_ndims; j++) {
        size_t dim;
        read_element(fh, dim);
        shape_data_owner.push_back(dim);
        std::cout << dim << ", ";
        if (dim > shape.shape_data[j]) {
          throw std::runtime_error(
              "Shape in testcase data exceeds maximum shape.");
        }
      }
      std::cout << ")";

      // Set the shape of the input to the actual, and not the maximum shape.
      // the previous shape.shape_data may not be deleted as it's owned by the
      // model.
      shape.shape_data = shape_data_owner.data() + shape_offset;
      shape_offset += read_ndims; // move offset to the next unused space

      // total number of bytes of tensor raw data
      read_element(fh, read_total_tensor_bytes);

      size_t numel = shape.Numel();
      size_t num_bytes = numel * AITemplateDtypeSizeBytes(dtype);
      std::cout << ", total_tensor_bytes=" << read_total_tensor_bytes
                << " - model expects " << num_bytes << "\n";
      if (num_bytes != read_total_tensor_bytes) {
        throw std::runtime_error("Tensor data total size mismatch.");
      }
      // allocate memory for tensor raw data on host
      void* h_data_expected = (void*)malloc(num_bytes);
      void* h_data = (void*)malloc(max_num_bytes);
      // read tensor raw data from file
      fh.read(
          reinterpret_cast<char*>(h_data_expected), read_total_tensor_bytes);

      // ---
      // Memory to place output tensors on host
      host_outputs.emplace_back(h_data, shape, dtype);
      ait_output_shapes_out.push_back(shape_data_owner.data());
      shape_offset += read_ndims;
      expected_outputs.emplace_back(h_data_expected, shape, dtype);
    }
  }

  AITemplateError run(
      AITemplateModelHandle handle) {
    bool graph_mode = false;

    return AITemplateModelContainerRunWithOutputsOnHost(
        handle,
        inputs.data(),
        inputs.size(),
        host_outputs.data(),
        host_outputs.size(),
        graph_mode,
        ait_output_shapes_out.data());
  }

  float benchmark(
      AITemplateModelHandle handle,
      size_t count,
      size_t num_threads) {
    bool graph_mode = false;
    float runtime_ms = -999.0f;
    AITemplateError err = AITemplateModelContainerBenchmark(
        handle,
        inputs.data(),
        inputs.size(),
        gpu_outputs.data(),
        gpu_outputs.size(),
        graph_mode,
        count,
        num_threads,
        true,
        &runtime_ms,
        ait_output_shapes_out.data());
    if (err != AITemplateError::AITemplateSuccess) {
      std::cout << "Benchmark failed with error " << static_cast<int>(err)
                << std::endl;
      return -1.0f;
    }
    return runtime_ms;
  }

  bool compare_results_to_expected() {
    bool passed = true;
    size_t num_outputs = 0;
    AITemplateModelContainerGetNumOutputs(handle, &num_outputs);
    for (unsigned output_idx = 0; output_idx < num_outputs; ++output_idx) {
      switch (expected_outputs[output_idx].dtype) {
        case AITemplateDtype::kInt:
          passed = passed and _compare_results_to_expected<int32_t>(output_idx);
          break;
        case AITemplateDtype::kLong:
          passed = passed and _compare_results_to_expected<int64_t>(output_idx);
          break;
        case AITemplateDtype::kFloat:
          passed = passed and _compare_results_to_expected<float>(output_idx);
          break;
        case AITemplateDtype::kBFloat16:
          passed =
              passed and _compare_results_to_expected<bfloat16>(output_idx);
          break;
        case AITemplateDtype::kHalf:
          passed = passed and _compare_results_to_expected<_Float16>(output_idx);
          break;
        case AITemplateDtype::kBool:
          passed = passed and _compare_results_to_expected<bool>(output_idx);
          break;
        default:
          std::cerr << "Unsupported output dtype! "
                    << static_cast<int>(expected_outputs[output_idx].dtype)
                    << std::endl;
          throw std::runtime_error("unsupported dtype for comparisons");
      }
    }
    return passed;
  }

  template <typename T>
  bool _compare_results_to_expected(unsigned output_idx) {
    unsigned ndims = host_outputs[output_idx].shape.size;
    // check the actual output shape
    for (unsigned i = 0; i < ndims; ++i) {
      if (expected_outputs[output_idx].shape.shape_data[i] !=
          ait_output_shapes_out[output_idx][i]) {
        std::cout
            << "Mismatch between expected output shape and actual shape after inference of output #"
            << i << " at dimension " << i << " expected shape[i]=="
            << host_outputs[output_idx].shape.shape_data[i]
            << " actual shape[i]==" << ait_output_shapes_out[output_idx][i]
            << std::endl;
        return false;
      }
    }
    size_t numel = host_outputs[output_idx].shape.Numel();
    T* data = reinterpret_cast<T*>(host_outputs[output_idx].ptr);
    T* expected_data = reinterpret_cast<T*>(expected_outputs[output_idx].ptr);
    size_t violations = 0;
    int worst_idx = -1;
    double worst_abs_diff = 0.0;

    for (size_t i = 0; i < numel; ++i) {
      double val = static_cast<double>(data[i]);
      double expected = static_cast<double>(expected_data[i]);
      double actual_diff = std::abs(val - expected);
      double tolerated_diff = atol +
          rtol * std::abs(expected); // as defined by torch.testing.assert_close
      if (actual_diff > worst_abs_diff) {
        worst_abs_diff = actual_diff;
      }
      if (actual_diff > tolerated_diff) {
        violations++;
      }
    }
    if (violations > 0) {
      std::cout
          << "Actual output and expected output are not equal for output with index "
          << output_idx << " of " << numel << " elements, " << violations
          << " differed by more than the tolerance of atol=" << atol
          << " and rtol=" << rtol << rtol << "\n";
      return false;
    }
    return true;
  }
};

int run_testcase(const char* input_file, bool benchmark) {
  std::cout << "Starting single test run with input " << input_file << "\n";
  {
    AITemplateModelHandle handle;
    AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
    auto deleter = [](void* data) { free(data); };
    AITStandaloneTestcase test(input_file, handle);

    AIT_ERROR_CHECK(test.run(handle));
    std::cout << "Finished test run with input " << input_file << "\n";
    int retval = -1;
    if (!test.compare_results_to_expected()) {
      std::cout << "Test failed. " << std::endl;
      return 1;
    }
    std::cout << "Test succeeded. " << std::endl;
  }
  if (benchmark) {
    std::cout << "Benchmarking with testcase " << input_file << "\n";
    AITemplateModelHandle handle;
    AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
    auto deleter = [](void* data) { free(data); };
    AITStandaloneTestcase benchmarker(input_file, handle);
    float runtime_ms = benchmarker.benchmark(handle, 10, 1);
    if (runtime_ms >= 0.0) {
      std::cout << "Benchmark result: " << input_file
                << " repetitions: 10, ms/iter: " << runtime_ms << "\n";
    }
  }

  return 0;
}

void free_tensors(AITData* tensors, int num_tensors) {
    if (tensors != nullptr) {
        for (int i = 0; i < num_tensors; i++) {
            if (tensors[i].ptr != nullptr) {
                free(tensors[i].ptr);  // Free the memory for data_generated
                tensors[i].ptr = nullptr;
            }
        }
        free(tensors);  // Free the memory for tensors array
        tensors = nullptr;
    }
}

int run_with_random_inputs() {
  AITemplateModelHandle handle;
  AITemplateModelContainerCreate(&handle, /*num_runtimes*/ 1);
  auto deleter = [](void* data) { free(data); };

  std::vector<OutputData> outputs;
  setup_constant();
  size_t num_params_ = 110;
  AITData* tensors = (AITData*)malloc(sizeof(AITData)*(num_params_-2));
  for(int i = 2; i < num_params_; i++){
    auto& input_shape = max_param_shapes_[i];
    auto shape = AITemplateParamShape{input_shape.data(), input_shape.size()};
    int64_t* data_generated = (int64_t*)malloc(shape.Numel() * sizeof(int64_t));
    std::uniform_int_distribution<> dist(-10, 10);
    std::mt19937 rnd_generator(1234);
    for (size_t j = 0; j < shape.Numel(); j++) {
      data_generated[j] = static_cast<int64_t>(dist(rnd_generator));
    }
    tensors[i-2] = AITData((void*)data_generated, shape, AITemplateDtype::kFloat);
  }
  const AITData* tensors_pointer = tensors;
  AITemplateModelContainerSetManyConstants(handle, &param_names_[2], tensors_pointer, num_params_-2);
  AITemplateModelContainerFoldConstantsInDoubleBuffer(handle, true);
  AIT_ERROR_CHECK(run(handle, outputs));

  // print out something
  for (const auto& output : outputs) {
    std::cout << "output: " << output.name << " at idx: " << output.index
              << " with shape: ";
    for (int i = 0; i < output.shape_size; i++) {
      std::cout << output.shape_ptr[i] << ",";
    }
    std::cout << "\n";
  }
  free_tensors(tensors, num_params_-2);
  // We are done and delete the handle.
  AITemplateModelContainerDelete(handle);
  return 0;
}

int main(int argc, char* argv[]) {
  try {
    if (argc <= 1) {
      std::cout
          << "No action provided on commandline. Running model with random maximum size inputs."
          << std::endl;

      return run_with_random_inputs();
    }
    std::string action(argv[1]);
    if ((action == "--help") or (action == "help")) {
      std::cout << "AITemplate standalone test runner usage:" << std::endl
                << " run with random input:   " << argv[0] << std::endl
                << " run single tests:        " << argv[0]
                << " test <testcase-file-1> ... <testcase-file-N>" << std::endl
                << " run tests and benchmark: " << argv[0]
                << " benchmark <testcase-file-1> ... <testcase-file-N>"
                << std::endl;
    }
    if ((action == "test") or (action == "benchmark")) {
      if (argc < 3) {
        std::cout
            << "Invalid number of arguments. Require at least one test case as argument"
            << std::endl;
      }
      int failure_count = 0;
      for (int i = 2; i < argc; i++) {
        if (run_testcase(argv[i], action == "benchmark") != 0) {
          failure_count++;
        }
      }
      if (failure_count == 0) {
        std::cout << "All tests succeeded." << std::endl;
      } else {
        std::cout << "Failed tests: " << failure_count << " of " << (argc - 2)
                  << std::endl;
      }
      return failure_count;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return -99;
  }
}
