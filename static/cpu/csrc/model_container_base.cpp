
#include "../include/model_container.h"
#include "../include/owned_constants.h"

namespace ait {
namespace {

// TODO : to be deleted later

// Contains the metadata for each constant.
constexpr std::array<ConstantInfo, 0> owned_constants = {
  
};
} // namespace

ModelContainerBase::ModelContainerBase(
    size_t num_inputs,
    size_t num_outputs,
    size_t num_bound_constants,
    size_t num_unbound_constants,
    size_t params_size,
    AITemplateAllocator& allocator)
    : constants_size_(params_size),
      constants_primary_(RAII_DeviceMalloc(constants_size_)),
      constants_secondary_(nullptr),
      use_constants_primary_buffer_(true),
      buffer_state_(BufferState::CLEAN),
      bound_constant_size_(num_bound_constants),
      bound_constant_dtypes_(num_bound_constants),
      num_params_(num_inputs + num_outputs + num_unbound_constants),
      param_names_(num_params_),
      param_dtypes_(num_params_),
      max_param_shapes_(num_params_),
      max_param_numel_(num_params_),
      max_param_storage_bytes_(num_params_) {


     param_names_[0] = "input_0";
     param_names_[1] = "input_1";
     param_names_[2] = "input_2";
     param_names_[3] = "output_0";
     param_dtypes_[0] = AITemplateDtype::kHalf;
     param_dtypes_[1] = AITemplateDtype::kHalf;
     param_dtypes_[2] = AITemplateDtype::kHalf;
     param_dtypes_[3] = AITemplateDtype::kHalf;


     max_param_shapes_[0] = {4, 14, 14, 256};
     max_param_shapes_[1] = {256, 2, 2, 256};
     max_param_shapes_[2] = {256};
     max_param_shapes_[3] = {4, 28, 28, 256};
  for (size_t i = 0; i < num_params_; ++i) {
    max_param_numel_[i] = std::accumulate(
      max_param_shapes_[i].begin(),
      max_param_shapes_[i].end(),
      1,
      std::multiplies<int64_t>()
    );
    max_param_storage_bytes_[i] = max_param_numel_[i] * AITemplateDtypeSizeBytes(param_dtypes_[i]);
  }




  const auto binary_constants_bin_size = static_cast<size_t>(_binary_constants_bin_end - _binary_constants_bin_start);
  const uint8_t* const binary_constants_bin_start = _binary_constants_bin_start;


  auto* constants_ptr = static_cast<uint8_t*>(constants_primary_.get());
  for (auto& constant_info : owned_constants) {
    auto* dst = constants_ptr + constant_info.internal_offset;
    if (constant_info.data_offset + constant_info.num_bytes > binary_constants_bin_size) {
      throw std::runtime_error(std::string("Copying constant ") + constant_info.name + " would overflow constant buffer");
    }
    std::memcpy(dst, binary_constants_bin_start + constant_info.data_offset, constant_info.num_bytes);
  }
}

ModelContainer* CreateModelContainer(size_t num_runtimes, AITemplateAllocator& allocator) {
  // num_runtimes, num_inputs, num_outputs, num_bound_constants, num_unbound_constants, params_size, allocator
  return new ModelContainer(num_runtimes, 3, 1, 0, 0, 0, allocator);
}
} // namespace ait