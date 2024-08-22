
#include "model_container.h"
#include "owned_constants.h"

namespace ait {
namespace {



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
     unbound_constant_name_to_idx_["stem_conv1_weight"] = 0;
     unbound_constant_name_to_idx_["stem_conv1_bias"] = 1;
     unbound_constant_name_to_idx_["layer1_0_conv1_weight"] = 2;
     unbound_constant_name_to_idx_["layer1_0_conv1_bias"] = 3;
     unbound_constant_name_to_idx_["layer1_0_conv2_weight"] = 4;
     unbound_constant_name_to_idx_["layer1_0_conv2_bias"] = 5;
     unbound_constant_name_to_idx_["layer1_0_downsample_0_weight"] = 6;
     unbound_constant_name_to_idx_["layer1_0_downsample_0_bias"] = 7;
     unbound_constant_name_to_idx_["layer1_0_conv3_weight"] = 8;
     unbound_constant_name_to_idx_["layer1_0_conv3_bias"] = 9;
     unbound_constant_name_to_idx_["layer1_1_conv1_weight"] = 10;
     unbound_constant_name_to_idx_["layer1_1_conv1_bias"] = 11;
     unbound_constant_name_to_idx_["layer1_1_conv2_weight"] = 12;
     unbound_constant_name_to_idx_["layer1_1_conv2_bias"] = 13;
     unbound_constant_name_to_idx_["layer1_1_conv3_weight"] = 14;
     unbound_constant_name_to_idx_["layer1_1_conv3_bias"] = 15;
     unbound_constant_name_to_idx_["layer1_2_conv1_weight"] = 16;
     unbound_constant_name_to_idx_["layer1_2_conv1_bias"] = 17;
     unbound_constant_name_to_idx_["layer1_2_conv2_weight"] = 18;
     unbound_constant_name_to_idx_["layer1_2_conv2_bias"] = 19;
     unbound_constant_name_to_idx_["layer1_2_conv3_weight"] = 20;
     unbound_constant_name_to_idx_["layer1_2_conv3_bias"] = 21;
     unbound_constant_name_to_idx_["layer2_0_conv1_weight"] = 22;
     unbound_constant_name_to_idx_["layer2_0_conv1_bias"] = 23;
     unbound_constant_name_to_idx_["layer2_0_conv2_weight"] = 24;
     unbound_constant_name_to_idx_["layer2_0_conv2_bias"] = 25;
     unbound_constant_name_to_idx_["layer2_0_downsample_0_weight"] = 26;
     unbound_constant_name_to_idx_["layer2_0_downsample_0_bias"] = 27;
     unbound_constant_name_to_idx_["layer2_0_conv3_weight"] = 28;
     unbound_constant_name_to_idx_["layer2_0_conv3_bias"] = 29;
     unbound_constant_name_to_idx_["layer2_1_conv1_weight"] = 30;
     unbound_constant_name_to_idx_["layer2_1_conv1_bias"] = 31;
     unbound_constant_name_to_idx_["layer2_1_conv2_weight"] = 32;
     unbound_constant_name_to_idx_["layer2_1_conv2_bias"] = 33;
     unbound_constant_name_to_idx_["layer2_1_conv3_weight"] = 34;
     unbound_constant_name_to_idx_["layer2_1_conv3_bias"] = 35;
     unbound_constant_name_to_idx_["layer2_2_conv1_weight"] = 36;
     unbound_constant_name_to_idx_["layer2_2_conv1_bias"] = 37;
     unbound_constant_name_to_idx_["layer2_2_conv2_weight"] = 38;
     unbound_constant_name_to_idx_["layer2_2_conv2_bias"] = 39;
     unbound_constant_name_to_idx_["layer2_2_conv3_weight"] = 40;
     unbound_constant_name_to_idx_["layer2_2_conv3_bias"] = 41;
     unbound_constant_name_to_idx_["layer2_3_conv1_weight"] = 42;
     unbound_constant_name_to_idx_["layer2_3_conv1_bias"] = 43;
     unbound_constant_name_to_idx_["layer2_3_conv2_weight"] = 44;
     unbound_constant_name_to_idx_["layer2_3_conv2_bias"] = 45;
     unbound_constant_name_to_idx_["layer2_3_conv3_weight"] = 46;
     unbound_constant_name_to_idx_["layer2_3_conv3_bias"] = 47;
     unbound_constant_name_to_idx_["layer3_0_conv1_weight"] = 48;
     unbound_constant_name_to_idx_["layer3_0_conv1_bias"] = 49;
     unbound_constant_name_to_idx_["layer3_0_conv2_weight"] = 50;
     unbound_constant_name_to_idx_["layer3_0_conv2_bias"] = 51;
     unbound_constant_name_to_idx_["layer3_0_downsample_0_weight"] = 52;
     unbound_constant_name_to_idx_["layer3_0_downsample_0_bias"] = 53;
     unbound_constant_name_to_idx_["layer3_0_conv3_weight"] = 54;
     unbound_constant_name_to_idx_["layer3_0_conv3_bias"] = 55;
     unbound_constant_name_to_idx_["layer3_1_conv1_weight"] = 56;
     unbound_constant_name_to_idx_["layer3_1_conv1_bias"] = 57;
     unbound_constant_name_to_idx_["layer3_1_conv2_weight"] = 58;
     unbound_constant_name_to_idx_["layer3_1_conv2_bias"] = 59;
     unbound_constant_name_to_idx_["layer3_1_conv3_weight"] = 60;
     unbound_constant_name_to_idx_["layer3_1_conv3_bias"] = 61;
     unbound_constant_name_to_idx_["layer3_2_conv1_weight"] = 62;
     unbound_constant_name_to_idx_["layer3_2_conv1_bias"] = 63;
     unbound_constant_name_to_idx_["layer3_2_conv2_weight"] = 64;
     unbound_constant_name_to_idx_["layer3_2_conv2_bias"] = 65;
     unbound_constant_name_to_idx_["layer3_2_conv3_weight"] = 66;
     unbound_constant_name_to_idx_["layer3_2_conv3_bias"] = 67;
     unbound_constant_name_to_idx_["layer3_3_conv1_weight"] = 68;
     unbound_constant_name_to_idx_["layer3_3_conv1_bias"] = 69;
     unbound_constant_name_to_idx_["layer3_3_conv2_weight"] = 70;
     unbound_constant_name_to_idx_["layer3_3_conv2_bias"] = 71;
     unbound_constant_name_to_idx_["layer3_3_conv3_weight"] = 72;
     unbound_constant_name_to_idx_["layer3_3_conv3_bias"] = 73;
     unbound_constant_name_to_idx_["layer3_4_conv1_weight"] = 74;
     unbound_constant_name_to_idx_["layer3_4_conv1_bias"] = 75;
     unbound_constant_name_to_idx_["layer3_4_conv2_weight"] = 76;
     unbound_constant_name_to_idx_["layer3_4_conv2_bias"] = 77;
     unbound_constant_name_to_idx_["layer3_4_conv3_weight"] = 78;
     unbound_constant_name_to_idx_["layer3_4_conv3_bias"] = 79;
     unbound_constant_name_to_idx_["layer3_5_conv1_weight"] = 80;
     unbound_constant_name_to_idx_["layer3_5_conv1_bias"] = 81;
     unbound_constant_name_to_idx_["layer3_5_conv2_weight"] = 82;
     unbound_constant_name_to_idx_["layer3_5_conv2_bias"] = 83;
     unbound_constant_name_to_idx_["layer3_5_conv3_weight"] = 84;
     unbound_constant_name_to_idx_["layer3_5_conv3_bias"] = 85;
     unbound_constant_name_to_idx_["layer4_0_conv1_weight"] = 86;
     unbound_constant_name_to_idx_["layer4_0_conv1_bias"] = 87;
     unbound_constant_name_to_idx_["layer4_0_conv2_weight"] = 88;
     unbound_constant_name_to_idx_["layer4_0_conv2_bias"] = 89;
     unbound_constant_name_to_idx_["layer4_0_downsample_0_weight"] = 90;
     unbound_constant_name_to_idx_["layer4_0_downsample_0_bias"] = 91;
     unbound_constant_name_to_idx_["layer4_0_conv3_weight"] = 92;
     unbound_constant_name_to_idx_["layer4_0_conv3_bias"] = 93;
     unbound_constant_name_to_idx_["layer4_1_conv1_weight"] = 94;
     unbound_constant_name_to_idx_["layer4_1_conv1_bias"] = 95;
     unbound_constant_name_to_idx_["layer4_1_conv2_weight"] = 96;
     unbound_constant_name_to_idx_["layer4_1_conv2_bias"] = 97;
     unbound_constant_name_to_idx_["layer4_1_conv3_weight"] = 98;
     unbound_constant_name_to_idx_["layer4_1_conv3_bias"] = 99;
     unbound_constant_name_to_idx_["layer4_2_conv1_weight"] = 100;
     unbound_constant_name_to_idx_["layer4_2_conv1_bias"] = 101;
     unbound_constant_name_to_idx_["layer4_2_conv2_weight"] = 102;
     unbound_constant_name_to_idx_["layer4_2_conv2_bias"] = 103;
     unbound_constant_name_to_idx_["layer4_2_conv3_weight"] = 104;
     unbound_constant_name_to_idx_["layer4_2_conv3_bias"] = 105;
     unbound_constant_name_to_idx_["fc_weight"] = 106;
     unbound_constant_name_to_idx_["fc_bias"] = 107;

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
     param_dtypes_[0] = AITemplateDtype::kFloat;
     param_dtypes_[2] = AITemplateDtype::kFloat;
     param_dtypes_[3] = AITemplateDtype::kFloat;
     param_dtypes_[4] = AITemplateDtype::kFloat;
     param_dtypes_[5] = AITemplateDtype::kFloat;
     param_dtypes_[6] = AITemplateDtype::kFloat;
     param_dtypes_[7] = AITemplateDtype::kFloat;
     param_dtypes_[8] = AITemplateDtype::kFloat;
     param_dtypes_[9] = AITemplateDtype::kFloat;
     param_dtypes_[10] = AITemplateDtype::kFloat;
     param_dtypes_[11] = AITemplateDtype::kFloat;
     param_dtypes_[12] = AITemplateDtype::kFloat;
     param_dtypes_[13] = AITemplateDtype::kFloat;
     param_dtypes_[14] = AITemplateDtype::kFloat;
     param_dtypes_[15] = AITemplateDtype::kFloat;
     param_dtypes_[16] = AITemplateDtype::kFloat;
     param_dtypes_[17] = AITemplateDtype::kFloat;
     param_dtypes_[18] = AITemplateDtype::kFloat;
     param_dtypes_[19] = AITemplateDtype::kFloat;
     param_dtypes_[20] = AITemplateDtype::kFloat;
     param_dtypes_[21] = AITemplateDtype::kFloat;
     param_dtypes_[22] = AITemplateDtype::kFloat;
     param_dtypes_[23] = AITemplateDtype::kFloat;
     param_dtypes_[24] = AITemplateDtype::kFloat;
     param_dtypes_[25] = AITemplateDtype::kFloat;
     param_dtypes_[26] = AITemplateDtype::kFloat;
     param_dtypes_[27] = AITemplateDtype::kFloat;
     param_dtypes_[28] = AITemplateDtype::kFloat;
     param_dtypes_[29] = AITemplateDtype::kFloat;
     param_dtypes_[30] = AITemplateDtype::kFloat;
     param_dtypes_[31] = AITemplateDtype::kFloat;
     param_dtypes_[32] = AITemplateDtype::kFloat;
     param_dtypes_[33] = AITemplateDtype::kFloat;
     param_dtypes_[34] = AITemplateDtype::kFloat;
     param_dtypes_[35] = AITemplateDtype::kFloat;
     param_dtypes_[36] = AITemplateDtype::kFloat;
     param_dtypes_[37] = AITemplateDtype::kFloat;
     param_dtypes_[38] = AITemplateDtype::kFloat;
     param_dtypes_[39] = AITemplateDtype::kFloat;
     param_dtypes_[40] = AITemplateDtype::kFloat;
     param_dtypes_[41] = AITemplateDtype::kFloat;
     param_dtypes_[42] = AITemplateDtype::kFloat;
     param_dtypes_[43] = AITemplateDtype::kFloat;
     param_dtypes_[44] = AITemplateDtype::kFloat;
     param_dtypes_[45] = AITemplateDtype::kFloat;
     param_dtypes_[46] = AITemplateDtype::kFloat;
     param_dtypes_[47] = AITemplateDtype::kFloat;
     param_dtypes_[48] = AITemplateDtype::kFloat;
     param_dtypes_[49] = AITemplateDtype::kFloat;
     param_dtypes_[50] = AITemplateDtype::kFloat;
     param_dtypes_[51] = AITemplateDtype::kFloat;
     param_dtypes_[52] = AITemplateDtype::kFloat;
     param_dtypes_[53] = AITemplateDtype::kFloat;
     param_dtypes_[54] = AITemplateDtype::kFloat;
     param_dtypes_[55] = AITemplateDtype::kFloat;
     param_dtypes_[56] = AITemplateDtype::kFloat;
     param_dtypes_[57] = AITemplateDtype::kFloat;
     param_dtypes_[58] = AITemplateDtype::kFloat;
     param_dtypes_[59] = AITemplateDtype::kFloat;
     param_dtypes_[60] = AITemplateDtype::kFloat;
     param_dtypes_[61] = AITemplateDtype::kFloat;
     param_dtypes_[62] = AITemplateDtype::kFloat;
     param_dtypes_[63] = AITemplateDtype::kFloat;
     param_dtypes_[64] = AITemplateDtype::kFloat;
     param_dtypes_[65] = AITemplateDtype::kFloat;
     param_dtypes_[66] = AITemplateDtype::kFloat;
     param_dtypes_[67] = AITemplateDtype::kFloat;
     param_dtypes_[68] = AITemplateDtype::kFloat;
     param_dtypes_[69] = AITemplateDtype::kFloat;
     param_dtypes_[70] = AITemplateDtype::kFloat;
     param_dtypes_[71] = AITemplateDtype::kFloat;
     param_dtypes_[72] = AITemplateDtype::kFloat;
     param_dtypes_[73] = AITemplateDtype::kFloat;
     param_dtypes_[74] = AITemplateDtype::kFloat;
     param_dtypes_[75] = AITemplateDtype::kFloat;
     param_dtypes_[76] = AITemplateDtype::kFloat;
     param_dtypes_[77] = AITemplateDtype::kFloat;
     param_dtypes_[78] = AITemplateDtype::kFloat;
     param_dtypes_[79] = AITemplateDtype::kFloat;
     param_dtypes_[80] = AITemplateDtype::kFloat;
     param_dtypes_[81] = AITemplateDtype::kFloat;
     param_dtypes_[82] = AITemplateDtype::kFloat;
     param_dtypes_[83] = AITemplateDtype::kFloat;
     param_dtypes_[84] = AITemplateDtype::kFloat;
     param_dtypes_[85] = AITemplateDtype::kFloat;
     param_dtypes_[86] = AITemplateDtype::kFloat;
     param_dtypes_[87] = AITemplateDtype::kFloat;
     param_dtypes_[88] = AITemplateDtype::kFloat;
     param_dtypes_[89] = AITemplateDtype::kFloat;
     param_dtypes_[90] = AITemplateDtype::kFloat;
     param_dtypes_[91] = AITemplateDtype::kFloat;
     param_dtypes_[92] = AITemplateDtype::kFloat;
     param_dtypes_[93] = AITemplateDtype::kFloat;
     param_dtypes_[94] = AITemplateDtype::kFloat;
     param_dtypes_[95] = AITemplateDtype::kFloat;
     param_dtypes_[96] = AITemplateDtype::kFloat;
     param_dtypes_[97] = AITemplateDtype::kFloat;
     param_dtypes_[98] = AITemplateDtype::kFloat;
     param_dtypes_[99] = AITemplateDtype::kFloat;
     param_dtypes_[100] = AITemplateDtype::kFloat;
     param_dtypes_[101] = AITemplateDtype::kFloat;
     param_dtypes_[102] = AITemplateDtype::kFloat;
     param_dtypes_[103] = AITemplateDtype::kFloat;
     param_dtypes_[104] = AITemplateDtype::kFloat;
     param_dtypes_[105] = AITemplateDtype::kFloat;
     param_dtypes_[106] = AITemplateDtype::kFloat;
     param_dtypes_[107] = AITemplateDtype::kFloat;
     param_dtypes_[108] = AITemplateDtype::kFloat;
     param_dtypes_[109] = AITemplateDtype::kFloat;
     param_dtypes_[1] = AITemplateDtype::kFloat;


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
  return new ModelContainer(num_runtimes, 1, 1, 0, 108, 0, allocator);
}
} // namespace ait