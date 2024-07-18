
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
// TODO: to be deleted later

void transposed_conv2d_bias_relu_3(
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
  int
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
            constants) {
    
    auto* blob_ptr = static_cast<uint8_t*>(blob_.get());
    
    
         params_[0].shape_ptrs = {ParamDim(4, 4, &input_0_dim_0), ParamDim(14, 14, &input_0_dim_1), ParamDim(14, 14, &input_0_dim_2), ParamDim(256, 256, &input_0_dim_3)};
     params_[1].shape_ptrs = {ParamDim(256, 256, &input_1_dim_0), ParamDim(2, 2, &input_1_dim_1), ParamDim(2, 2, &input_1_dim_2), ParamDim(256, 256, &input_1_dim_3)};
     params_[2].shape_ptrs = {ParamDim(256, 256, &input_2_dim_0)};
     params_[3].shape_ptrs = {ParamDim(4, 4, &output_0_dim_0), ParamDim(28, 28, &output_0_dim_1), ParamDim(28, 28, &output_0_dim_2), ParamDim(256, 256, &output_0_dim_3)};

      
      
    }

    ~Model() {
      
      
    }

    void SetUpInputsOutputs() {
             input_0 = static_cast<decltype(input_0)>(params_[0].ptr);

if (input_0 == nullptr) {
    throw std::runtime_error("Constant input_0 was not set! Set the value with set_constant.");
}
    
     input_1 = static_cast<decltype(input_1)>(params_[1].ptr);

if (input_1 == nullptr) {
    throw std::runtime_error("Constant input_1 was not set! Set the value with set_constant.");
}
    
     input_2 = static_cast<decltype(input_2)>(params_[2].ptr);

if (input_2 == nullptr) {
    throw std::runtime_error("Constant input_2 was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params_[3].ptr);

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
  
    }


    ///////////////////////////////////////////////////////////////////////////
    // default RunImpl implemenation
    void RunImpl(StreamType stream) {
        
  
  
    transposed_conv2d_bias_relu_3(
        input_0,
        input_1,
        output_0,

        input_2,

        global_workspace_,
        &input_0_dim_0,
        &input_1_dim_3,
        &input_0_dim_3,
        &input_1_dim_1,
        &input_1_dim_2,
        &input_0_dim_1,
        &input_0_dim_2,
        &output_0_dim_0,
        &output_0_dim_1,
        &output_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0
    );
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
        std::cout << "Profiling: " << "transposed_conv2d_bias_relu_3" << " (" << iters << " iterations)" << std::endl;
        float milliseconds = 0.0;
        for (int i = 0; i < iters; i ++) {
          struct timespec start, end;
          clock_gettime(CLOCK_MONOTONIC, &start);
    transposed_conv2d_bias_relu_3(
        input_0,
        input_1,
        output_0,

        input_2,

        global_workspace_,
        &input_0_dim_0,
        &input_1_dim_3,
        &input_0_dim_3,
        &input_1_dim_1,
        &input_1_dim_2,
        &input_0_dim_1,
        &input_0_dim_2,
        &output_0_dim_0,
        &output_0_dim_1,
        &output_0_dim_2,
        2,
        1,
        0,
        2,
        1,
        0
    );
          clock_gettime(CLOCK_MONOTONIC, &end);
          milliseconds += ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6);
        }
        ss << "\"" << "transposed_conv2d_bias_relu_3" << "\": { \"ms_per_iter\": "
           << std::setprecision(4) << (milliseconds/iters)
           << ", \"qps\": " << 1000 * iters / milliseconds
           << ", \"input_sizes\": " << "[[\"4\", \"14\", \"14\", \"256\"], [\"256\", \"2\", \"2\", \"256\"], [\"256\"]]"
           << ", \"output_sizes\": " << "[[\"4\", \"28\", \"28\", \"256\"]]"
        
          << ", \"dilate\": " << "\"1\""
        
          << ", \"group\": " << "\"1\""
        
          << ", \"pad\": " << "\"0\""
        
          << ", \"stride\": " << "\"2\""
        
           << " } ";
        
          ss << "\n";
        
      }
      
      ss << "}\n";
#endif
    }

    static std::unique_ptr<Model> Create(
      AITemplateAllocator& allocator,
      uint8_t* constants
    ) {
      return std::make_unique<Model>(
          2531840,
          0 * (1 + 0),
          0 * (1 + 0),
          3,
          1,
          0,
          constants,
          allocator
      );
    }

  private:
   void* input_0 {nullptr};
   void* input_1 {nullptr};
   void* input_2 {nullptr};
   void* output_0 {nullptr};
   int64_t input_0_dim_0 { 4 };
   int64_t input_0_dim_1 { 14 };
   int64_t input_0_dim_2 { 14 };
   int64_t input_0_dim_3 { 256 };
   int64_t input_1_dim_0 { 256 };
   int64_t input_1_dim_1 { 2 };
   int64_t input_1_dim_2 { 2 };
   int64_t input_1_dim_3 { 256 };
   int64_t input_2_dim_0 { 256 };
   int64_t output_0_dim_0 { 4 };
   int64_t output_0_dim_1 { 28 };
   int64_t output_0_dim_2 { 28 };
   int64_t output_0_dim_3 { 256 };


};
} // namespace ait