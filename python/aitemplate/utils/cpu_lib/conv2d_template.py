import jinja2

template = jinja2.Template(
            """
{{indent}}//{{name}}
{% if is_transpose %}
{{indent}}{{DataName}}* tmp_out = ({{DataName}}*)malloc(NI * HO * WO * CO * sizeof({{DataName}}));
{% endif %}
{{indent}}xnn_operator_t op_conv = nullptr;
{{indent}}const xnn_status status = xnn_create_{{Conv2DSpecialization}}(
{{indent}}  PH, PW, PH, PW, i32_kernel_h, i32_kernel_w,
{{indent}}  SH, SW, DH, DW, 1, CI,
{{indent}}  CO, 1 * CI, 1 * CO, ({{DataName}}*)(weight_ptr), ({{DataName}}*)(bias_ptr),
{% if is_relu %}
{{indent}}  0, std::numeric_limits<{{DataName}}>::infinity(),
{% elif is_relu6 %}
{{indent}}  0, 6,
{% else %}
{{indent}}  -std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(),
{% endif %}
{{indent}}  /*flags=*/0, nullptr, nullptr, &op_conv);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_conv(op_conv, xnn_delete_operator);
{{indent}}CHECK_EQ(status, xnn_status_success);
{{indent}}CHECK_NE(op_conv, nullptr);
{{indent}}size_t workspace_size = SIZE_MAX;
{{indent}}size_t workspace_alignment = SIZE_MAX;
{{indent}}CHECK_EQ(
{{indent}}  xnn_reshape_{{Conv2DSpecialization}}(
{{indent}}    op_conv, i32_batch, i32_in_h, i32_in_w,
{{indent}}    &workspace_size, &workspace_alignment,
{{indent}}    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}    /*threadpool=*/pthreadpool_), xnn_status_success);
{{indent}}CHECK_EQ(workspace_size, 0);
{{indent}}CHECK_EQ(workspace_alignment, 1);
{{indent}}CHECK_EQ(xnn_setup_{{Conv2DSpecialization}}(
{{indent}}    op_conv, 
{{indent}}    /*workspace=*/nullptr, 
{{indent}}    ({{DataName}}*)(in_ptr), 
{% if is_transpose %}
{{indent}}    ({{DataName}}*)(tmp_out)), xnn_status_success);
{% else %}
{{indent}}    ({{DataName}}*)(out_ptr)), xnn_status_success);
{% endif %}
{{indent}}CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/pthreadpool_), xnn_status_success);
            """
        )
template_with_pruning = jinja2.Template(
            """
{{indent}}//{{name}}
{{indent}}xnn_operator_t op_conv = nullptr;
{{indent}}const xnn_status status = xnn_create_input_T_pruned_convolution2d_nhwc_f32_x2v(
{{indent}}  PH, PW, PH, PW, i32_kernel_h, i32_kernel_w,
{{indent}}  SH, SW, DH, DW, 1, CI,
{{indent}}  CO, 1 * CI, 1 * CO, ({{DataName}}*)(weight_ptr), ({{DataName}}*)(bias_ptr),
{% if is_relu %}
{{indent}}  0, std::numeric_limits<{{DataName}}>::infinity(),
{% elif is_relu6 %}
{{indent}}  0, 6,
{% else %}
{{indent}}  -std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(),
{% endif %}
{{indent}}  /*flags=*/0, nullptr, nullptr, &op_conv);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_conv(op_conv, xnn_delete_operator);
{{indent}}CHECK_EQ(status, xnn_status_success);
{{indent}}CHECK_NE(op_conv, nullptr);
{{indent}}size_t workspace_size = SIZE_MAX;
{{indent}}size_t workspace_alignment = SIZE_MAX;
{{indent}}CHECK_EQ(
{{indent}}  xnn_reshape_input_T_pruned_convolution2d_nhwc_f32(
{{indent}}    op_conv, i32_batch, i32_in_h, i32_in_w,
{{indent}}    &workspace_size, &workspace_alignment,
{{indent}}    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}    /*threadpool=*/pthreadpool_, pruning_ratio), xnn_status_success);
{{indent}}CHECK_EQ(workspace_size, 0);
{{indent}}CHECK_EQ(workspace_alignment, 1);
{{indent}}CHECK_EQ(xnn_setup_input_T_pruned_convolution2d_nhwc_f32(
{{indent}}    op_conv, 
{{indent}}    /*workspace=*/nullptr, 
{{indent}}    ({{DataName}}*)(in_ptr), 
{{indent}}    ({{DataName}}*)(out_ptr), 
{{indent}}    (uint16_t*)(weight_indice_ptr),
{{indent}}    /*lmul=*/2), xnn_status_success);
{{indent}}CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/pthreadpool_), xnn_status_success);
            """
        )
template_depthwise = jinja2.Template(
            """
{{indent}}//{{name}}
{% if is_transpose %}
{{indent}}{{DataName}}* tmp_out = ({{DataName}}*)malloc(NI * HO * WO * CO * sizeof({{DataName}}));
{% endif %}
{{indent}}xnn_operator_t op_conv = nullptr;
{{indent}}const xnn_status status = xnn_create_{{Conv2DSpecialization}}(
{{indent}}  PH, PW, PH, PW, i32_kernel_h, i32_kernel_w,
{{indent}}  SH, SW, DH, DW, CI, 1,
{{indent}}  1, 1 * CI, 1 * CO, ({{DataName}}*)(weight_ptr), ({{DataName}}*)(bias_ptr),
{% if is_relu %}
{{indent}}  0, std::numeric_limits<{{DataName}}>::infinity(),
{% elif is_relu6 %}
{{indent}}  0, 6,
{% else %}
{{indent}}  -std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(),
{% endif %}
{{indent}}  /*flags=*/0, nullptr, nullptr, &op_conv);
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op_conv(op_conv, xnn_delete_operator);
{{indent}}CHECK_EQ(status, xnn_status_success);
{{indent}}CHECK_NE(op_conv, nullptr);
{{indent}}size_t workspace_size = SIZE_MAX;
{{indent}}size_t workspace_alignment = SIZE_MAX;
{{indent}}CHECK_EQ(
{{indent}}  xnn_reshape_{{Conv2DSpecialization}}(
{{indent}}    op_conv, i32_batch, i32_in_h, i32_in_w,
{{indent}}    &workspace_size, &workspace_alignment,
{{indent}}    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
{{indent}}    /*threadpool=*/pthreadpool_), xnn_status_success);
{{indent}}CHECK_EQ(workspace_size, 0);
{{indent}}CHECK_EQ(workspace_alignment, 1);
{{indent}}CHECK_EQ(xnn_setup_{{Conv2DSpecialization}}(
{{indent}}    op_conv, 
{{indent}}    /*workspace=*/nullptr, 
{{indent}}    ({{DataName}}*)(in_ptr), 
{% if is_transpose %}
{{indent}}    ({{DataName}}*)(tmp_out)), xnn_status_success);
{% else %}
{{indent}}    ({{DataName}}*)(out_ptr)), xnn_status_success);
{% endif %}
{{indent}}CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/pthreadpool_), xnn_status_success);
            """
        )
code_snippet = jinja2.Template(
"""
{% if not is_bias %}
{{indent}}{{DataName}}* bias_ptr = ({{DataName}}*)malloc(i32_out_ch * sizeof({{DataName}}));
{{indent}}std::memset(bias_ptr, 0, i32_out_ch * sizeof({{DataName}}));
{% endif %}
{{conv2d}}
{{extra_kind}}

{% if not is_bias %}
{{indent}}free(bias_ptr);
{% endif %}
"""
        )
# add, sub, mul, div (f16, f32)
binary_func_minmax_flag_op = jinja2.Template(
"""
{{indent}}xnn_operator_t binary_func_minmax_flag_op = nullptr;
{% if is_relu %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, std::numeric_limits<{{DataName}}>::infinity(), 0, &binary_func_minmax_flag_op));
{% elif is_relu6 %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, 6, 0, &binary_func_minmax_flag_op));
{% else %}
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(-std::numeric_limits<{{DataName}}>::infinity(), std::numeric_limits<{{DataName}}>::infinity(), 0, &binary_func_minmax_flag_op));
{% endif %}
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_binary_func_minmax_flag_op(binary_func_minmax_flag_op, xnn_delete_operator);
{{indent}}const size_t a_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}const size_t b_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_reshape_{{operation}}_{{DataType}}(
{{indent}}                        binary_func_minmax_flag_op, 4, a_shape, 4, b_shape,
{{indent}}                        /*threadpool=*/pthreadpool_));
{{indent}}CHECK_EQ(
{{indent}}  xnn_status_success, xnn_setup_{{operation}}_{{DataType}}(binary_func_minmax_flag_op, ({{DataName}}*)(res_ptr), ({{DataName}}*)(out_ptr), ({{DataName}}*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(binary_func_minmax_flag_op, /*threadpool=*/pthreadpool_));
"""
)
# copysign(f32), maximum(f16, f32), minimum(f16, f32), squared_difference(f16, f32), mul(s32), 
binary_func_flag_op = jinja2.Template(
"""
{{indent}}xnn_operator_t binary_func_flag_op = nullptr;
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, &binary_func_flag_op));
{{indent}}std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_binary_func_flag_op(binary_func_flag_op, xnn_delete_operator);
{{indent}}const size_t a_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}const size_t b_shape[] = { (size_t)i32_out_batch, (size_t)i32_out_h, (size_t)i32_out_w, (size_t)i32_out_ch};
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_reshape_{{operation}}_{{DataType}}(
{{indent}}                        binary_func_flag_op, a_shape, input1_dims.data(), b_shape, input2_dims.data(),
{{indent}}                        /*threadpool=*/pthreadpool_));
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_setup_{{operation}}_{{DataType}}(binary_func_flag_op, ({{DataName}}*)(res_ptr), ({{DataName}}*)(out_ptr), ({{DataName}}*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(binary_func_flag_op, /*threadpool=*/pthreadpool_));
"""
)
transpose_func = jinja2.Template(
"""
{{indent}}xnn_operator_t transpose_op = nullptr;
{{indent}}std::vector<size_t> shape = { (size_t)(NI * HO * WO), (size_t)CO};
{{indent}}std::vector<size_t> perm = {1, 0};
{{indent}}CHECK_EQ(xnn_status_success, xnn_create_{{operation}}_{{DataType}}(0, &transpose_op));
{{indent}}CHECK_NE(nullptr, transpose_op);
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_reshape_{{operation}}_{{DataType}}(
{{indent}} transpose_op, shape.size(), shape.data(), perm.data(), pthreadpool_));
{{indent}}CHECK_EQ(
{{indent}}xnn_status_success, xnn_setup_{{operation}}_{{DataType}}(transpose_op, tmp_out, ({{DataName}}*)(out_ptr)));
{{indent}}CHECK_EQ(xnn_status_success, xnn_run_operator(transpose_op, /*threadpool=*/pthreadpool_));
{{indent}}free(tmp_out);
"""
)
