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
#pragma once

#include <stdexcept>
#include <string>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace ait {

// This serves as a base class for AIT runtime objects, e.g. the compiled
// model and the constant folder. It uses CRTP as a mechanism to call into
// a few base class methods (dynamic dispatch is not needed in ModelContainer,
// so there's no need to add a vtable). Inheriting classes should implement
// the following methods:
// - RunImpl(StreamType):    The bulk of the compiled model's kernel invocations
//                           go here.
// - SetUpInputsOutputs():   Check the provided input/output pointers dtypes &
//                           sizes
// - DeviceToDeviceCopies(): Called at the end of infernece, copy views of
//                           inputs/constants to the provided output pointer.
//
// In practice, inheriting classes are generated via MODEL_TEMPLATE in
// python/aitemplate/backend/main_templates.py.
template <typename ModelType>
class ModelBase {
 protected:
  // Should not be constructed directly, use the base class' factory function
  // instead.
  ModelBase(
      size_t blob_size,
      size_t workspace_size,
      size_t unique_workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      uint8_t* constants)
      : blob_(RAII_DeviceMalloc(blob_size)),
        workspace_(RAII_DeviceMalloc(workspace_size)),
        params_(num_inputs + num_outputs + num_unbound_constants),
        workspace_size_{workspace_size},
        unique_workspace_size_{unique_workspace_size},
        num_inputs_(num_inputs),
        num_outputs_(num_outputs),
        constants_(constants),
        run_finished_(true) {
    global_workspace_ =
        static_cast<uint8_t*>(workspace_.get()) + unique_workspace_size;
    unique_workspace_ = static_cast<uint8_t*>(workspace_.get());
  }

 public:
  virtual ~ModelBase() {
  }

  ModelBase(ModelBase&&) = delete;
  ModelBase& operator=(ModelBase&&) = delete;
  ModelBase(const ModelBase&) = delete;
  ModelBase& operator=(const ModelBase&) = delete;

  void Run(StreamType stream, bool graph_mode) {
    auto* model = static_cast<ModelType*>(this);
    model->SetUpInputsOutputs();
    {
      std::lock_guard<std::mutex> lk(cv_m);
      run_finished_.store(false);
    }
    model->RunImpl(stream);
    {
      std::lock_guard<std::mutex> lk(cv_m);
      run_finished_.store(true);
    }
    cv.notify_one();
  }

  void Profile(StreamType stream, size_t iters, const std::string& filename) {
    auto* model = static_cast<ModelType*>(this);
    model->SetUpInputsOutputs();
    model->ProfileImpl(stream, iters, filename);
  }

  bool IsPending() {
    std::unique_lock<std::mutex> lk(cv_m);
    auto query = run_finished_.load();
    if (query == false) {
      return true;
    }
    if (query != true) {
      LOG(WARNING) << "Pending model run did not finish successfully.";
    }
    return false;
  }

  void WaitForCompletion() {
    std::unique_lock<std::mutex> lk(cv_m);
    cv.wait(lk, [this] { return run_finished_.load(); });
    std::cout << "CPU synchronization code executed.\n";
  }

  size_t NumInputs() const {
    return num_inputs_;
  }

  size_t NumOutputs() const {
    return num_outputs_;
  }

  void SetParam(const void* src, size_t param_idx) {
    CHECK_VECTOR_ACCESS(params_, param_idx)
    // const_cast is not ideal here, but it is unfortunately
    // necessary:
    // 1) We store outputs and inputs in the same vector,
    //    and outputs cannot be const.
    // 2) Most of the codegen is not const-correct (most ops
    //    require non-const pointers). So even if we put const
    //    pointers into params, a const_cast would be required
    //    somewhere else.
    params_[param_idx].ptr = const_cast<void*>(src);
  }

  void SetInput(
      const void* src,
      const AITemplateParamShape& shape,
      size_t idx) {
    SetInputShape(shape, idx);
    SetParam(src, idx);
  }

  void SetOutput(void* src, size_t idx) {
    SetParam(src, idx + num_inputs_);
  }

  // Write the (possibly dynamic) output shape to the given pointer.
  // Note that this should be called _after_ the shape inference in
  // Run() is finished. output_shape_out should be able to store
  // at least GetOutputMaximumShape(idx).size values.
  void GetOutputShape(size_t idx, int64_t* output_shape_out) {
    const auto param_idx = idx + num_inputs_;
    CHECK_VECTOR_ACCESS(params_, param_idx);
    const auto& shape_ptrs = params_[param_idx].shape_ptrs;
    for (size_t i = 0; i < shape_ptrs.size(); ++i) {
      output_shape_out[i] = shape_ptrs[i].GetValue();
    }
  }

  void SetConstant(const char* name, const void* src) {
    auto it = constant_name_to_ptr_.find(name);
    if (it == constant_name_to_ptr_.end()) {
      throw std::out_of_range(std::string("Could not find constant ") + name);
    }
    const void** ptr = it->second;
    *ptr = src;
  }

 private:
  void SetInputShape(const AITemplateParamShape& shape, size_t idx) {
    auto& param = params_[idx];
    if (shape.size != param.shape_ptrs.size()) {
      throw std::runtime_error(
          "[SetInputShape] Got wrong param shape for input " +
          std::to_string(idx) + "; expected " +
          std::to_string(param.shape_ptrs.size()) + ", got " +
          std::to_string(shape.size));
    }
    for (size_t i = 0; i < param.shape_ptrs.size(); ++i) {
      param.shape_ptrs[i].SetValue(shape.shape_data[i], param.name);
    }
  }

 protected:
  std::condition_variable cv;
  std::mutex cv_m;
  std::atomic<bool> run_finished_;
  Ptr blob_;
  // Memory for constants that were folded into the *.so. Unowned by Model,
  // owned by ModelContainer.
  // TODO: make this const. It can't be const right now because we derive
  // tensor pointers from it, and no tensor pointers are const.
  uint8_t* constants_;
  size_t num_inputs_;
  size_t num_outputs_;

  // These values are preserved for multi-stream needs.
  size_t workspace_size_;
  size_t unique_workspace_size_;
  // The workspace blob is used as scratch memory. See
  // _generate_workspace in memory planning for more information.
  Ptr workspace_;
  uint8_t* global_workspace_{nullptr};
  uint8_t* unique_workspace_{nullptr};

  class ParamDim {
   public:
    ParamDim(int64_t lower_bound, int64_t upper_bound, int64_t* value)
        : lower_bound_(lower_bound), upper_bound_(upper_bound), value_(value) {}

    void SetValue(int64_t new_value, const char* name = nullptr) {
      if (new_value < lower_bound_ || new_value > upper_bound_) {
        throw std::out_of_range(
            "[SetValue] Dimension got value out of bounds; expected value to be in [" +
            std::to_string(lower_bound_) + ", " + std::to_string(upper_bound_) +
            "], but got " + std::to_string(new_value) +
            (name ? ". Variable name: " + std::string(name) : "") + ".");
      }
      *value_ = new_value;
    }

    int64_t GetValue() const {
      return *value_;
    }

   private:
    int64_t lower_bound_;
    int64_t upper_bound_;
    int64_t* value_;
  };

  struct ParamInfo {
    void* ptr = nullptr;
    // TODO add offset
    const char* name;
    std::vector<ParamDim> shape_ptrs;
  };

  // Contains info for all tensors marked as inputs
  // or outputs. The first num_inputs elements are the inputs.
  // Constants are not included.
  std::vector<ParamInfo> params_;

  std::unordered_map<std::string, const void**> constant_name_to_ptr_;
};

} // namespace ait
