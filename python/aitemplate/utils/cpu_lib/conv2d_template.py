import jinja2

template = jinja2.Template(
            """
{{indent}}//{{name}}
{% if is_transpose %}
{{indent}}{{DataName}}* tmp_out = ({{DataName}}*)malloc(NI * HO * i32_out_w * CO * sizeof({{DataName}}));
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
{{indent}}const xnn_status status = xnn_create_input_T_pruned_convolution2d_nhwc_f32_x{{LMUL}}v(
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
{{indent}}    /*lmul=*/{{LMUL}}), xnn_status_success);
{{indent}}CHECK_EQ(xnn_run_operator(op_conv, /*threadpool=*/pthreadpool_), xnn_status_success);
            """
        )
template_choose_merge_im2col_packing_setting_x1v = jinja2.Template("""
{{indent}}if(SH == 2){
{{indent}}    if(PH){
{{indent}}      xnn_x32_packa_in_T_gemm_im2col_s2_d1_x1v(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}      );
{{indent}}    }
{{indent}}    else{
{{indent}}      xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr, nr
{{indent}}      );
{{indent}}    }
{{indent}}  }
{{indent}}else if(SH == 1){
{{indent}}  	xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x1v(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}      );
{{indent}}}
{{microkernel_computation}}
"""
)

template_choose_merge_im2col_packing_setting_x2v = jinja2.Template("""
{{indent}}if(SH == 2){
{{indent}}    if(PH){
{{indent}}      xnn_x32_packa_in_T_gemm_im2col_s2_d1_x2v(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}      );
{{indent}}    }
{{indent}}    else{
{{indent}}      xnn_pack_f32_with_im2col_input_T_nr pack;
{{indent}}      if (i32_out_w >= (3 * nr) >> 2)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x2v
{{indent}}      else
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v
{{indent}}
{{indent}}      pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr, nr
{{indent}}      );
{{indent}}    }
{{indent}}  }
{{indent}}else if(SH == 1){
{{indent}}  xnn_pack_f32_with_im2col_input_T pack;
{{indent}}  if (i32_out_w >= (3 * nr) >> 2)
{{indent}}      pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x2v
{{indent}}  else
{{indent}}      pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x2v
{{indent}}  pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch,
{{indent}}      i32_out_h, i32_out_w,
{{indent}}      i32_kernel_h, i32_kernel_w, SH, SW,
{{indent}}      DH, DW, PH, PW,
{{indent}}      reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}  );
{{indent}}}
{{microkernel_computation}}
"""
)

template_choose_merge_im2col_packing_setting_x4v = jinja2.Template("""
{{indent}}if(SH == 2){
{{indent}}    if(PH){
{{indent}}        xnn_pack_f32_with_im2col_input_T pack;
{{indent}}        if (i32_out_w >= (3 * nr) >> 2)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1
{{indent}}        else if (i32_out_w >= (3 * nr) >> 3)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_2x4v
{{indent}}        else
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_1x4v
{{indent}}
{{indent}}        pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}            i32_out_h, i32_out_w,
{{indent}}            i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}            DH, DW, PH, PW, \
{{indent}}            reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}            );
{{indent}}    }
{{indent}}    else{
{{indent}}        xnn_pack_f32_with_im2col_input_T_nr pack;
{{indent}}        if (i32_out_w >= (3 * nr) >> 2)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x4v
{{indent}}        else if (i32_out_w >= (3 * nr) >> 3)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x2v
{{indent}}        else
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v
{{indent}}
{{indent}}        pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}            i32_out_h, i32_out_w,
{{indent}}            i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}            DH, DW, PH, PW, \
{{indent}}            reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr, nr
{{indent}}        );
{{indent}}    }
{{indent}}  }
{{indent}}else if(SH == 1){
{{indent}}    xnn_pack_f32_with_im2col_input_T pack;
{{indent}}    if (i32_out_w >= (3 * nr) >> 2)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x4v
{{indent}}    else if (i32_out_w >= (3 * nr) >> 3)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x4v
{{indent}}    else
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x4v
{{indent}}
{{indent}}    pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}    );
{{indent}}}
{{microkernel_computation}}
"""
)

template_choose_merge_im2col_packing_setting_x8v = jinja2.Template("""
{{indent}}if(SH == 2){
{{indent}}    if(PH){
{{indent}}        xnn_x32_packa_in_T_gemm_im2col_s2_d1_x8v(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}            i32_out_h, i32_out_w,
{{indent}}            i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}            DH, DW, PH, PW, \
{{indent}}            reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}        );
{{indent}}    }
{{indent}}    else{
{{indent}}        xnn_pack_f32_with_im2col_input_T_nr pack;
{{indent}}        if (i32_out_w >= (3 * nr) >> 3)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x4v
{{indent}}        else if (i32_out_w >= (3 * nr) >> 4)
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x2v
{{indent}}        else
{{indent}}            pack = xnn_x32_packa_in_T_gemm_im2col_s2_d1_p0_x1v
{{indent}}
{{indent}}        pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}            i32_out_h, i32_out_w,
{{indent}}            i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}            DH, DW, PH, PW, \
{{indent}}            reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr, nr
{{indent}}        );
{{indent}}    }
{{indent}}}
{{indent}}else if(SH == 1){
{{indent}}    xnn_pack_f32_with_im2col_input_T pack;
{{indent}}    if (i32_out_w >= (3 * nr) >> 2)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_8x8v
{{indent}}    else if (i32_out_w >= (3 * nr) >> 3)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_4x8v
{{indent}}    else if (i32_out_w >= (3 * nr) >> 4)
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_2x8v
{{indent}}    else
{{indent}}        pack = xnn_x32_packa_in_T_gemm_im2col_s1_d1_1x4v
{{indent}}
{{indent}}    pack(i32_batch, i32_in_h, i32_in_w, i32_in_ch, \
{{indent}}        i32_out_h, i32_out_w,
{{indent}}        i32_kernel_h, i32_kernel_w, SH, SW, \
{{indent}}        DH, DW, PH, PW, \
{{indent}}        reinterpret_cast<uint32_t*>(in_ptr), input_packed_ptr
{{indent}}    );
{{indent}}}
{{microkernel_computation}}
"""
)

microkernel_computation = jinja2.Template(
    """
{{indent}}const size_t num_threads = std::thread::hardware_concurrency();
{{indent}}uint32_t kernel_size = i32_kernel_h * i32_kernel_w;
{{indent}}uint32_t batch_output_size = i32_batch * i32_out_h * i32_out_w * i32_out_ch;
{{indent}}uint32_t nr = __riscv_vsetvlmax_e32m{{LMUL}}();
{{indent}}uint32_t* input_packed_ptr = (uint32_t*)malloc(kernel_size * i32_in_ch * round_up(batch_output_size, nr));
{{indent}}{{merge_im2col_packing}}
{{indent}}{{microkernel_lambda_func}}
{{indent}}const size_t im2col_row_cnt = kernel_size * CI;
{{indent}}struct function_context context = (struct function_context){
{{indent}}{{indent}}.input = (float*)(in_ptr),
{{indent}}{{indent}}.bias = (float*)(bias_ptr),
{{indent}}{{indent}}.pruned_weight = (float*)(weight_ptr),
{{indent}}{{indent}}.output = (float*)(out_ptr),
{{indent}}{{indent}}.input_channel = CI,
{{indent}}{{indent}}.output_channel = CO,
{{indent}}{{indent}}.output_height = i32_out_h,
{{indent}}{{indent}}.output_width = i32_out_w,
{{indent}}{{indent}}.mr = {{MR}},
{{indent}}{{indent}}.nr = nr,
{{indent}}{{indent}}.im2col_packing = input_packed_ptr,
{{indent}}{{indent}}.indice = (uint16_t*)(weight_indice_ptr),
{{indent}}{{indent}}.microkernel = xnn_f32_gemm_ukernel_{{MR}}x{{LMUL}}v_columnwise_pruned__rvv,
{{indent}}{{indent}}.a_stride  = (im2col_row_cnt << 2),
{{indent}}{{indent}}.cm_stride = (batch_output_size << 2),
{{indent}}{{indent}}.cn_stride = (nr << 2),
{{indent}}{{indent}}.k_scaled = (im2col_row_cnt << 2),
{{indent}}{{indent}}.w_stride = ((round_up_po2(im2col_row_cnt, 1 * 1)) << 2),// bias + transposed weight[out_ch][in_ch]
{{indent}}};
{{indent}}const size_t num_other_tiles = 1 * divide_round_up(CO, {{MR}});
{{indent}}const size_t target_tiles_per_thread = 5;
{{indent}}const size_t max_nc = divide_round_up(CO * num_other_tiles, num_threads * target_tiles_per_thread);
{{indent}}size_t nc = batch_output_size;
{{indent}}if (max_nc < nc) {
{{indent}}  nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
{{indent}}}

{{indent}}void conv2d_columnwise_pruning_vector = [](function_context* context, \
{{indent}}    size_t mr_block_start, size_t nr_block_start, size_t mr_block_size, size_t nr_block_size){
{{indent}}    uint32_t nr = context->nr;
{{indent}}    uint32_t w_stride = context->w_stride;
{{indent}}};

{{indent}}pthreadpool_parallelize_2d_tile_2d(
{{indent}}    pthreadpool_,
{{indent}}    (pthreadpool_task_2d_tile_2d_t)conv2d_columnwise_pruning_vector,
{{indent}}    (void*) ((uintptr_t) &context),
{{indent}}    context.output_channel, batch_output_size,
{{indent}}    {{MR}}, nc,
{{indent}}    0x00000001);

"""
)

microkernel_lambda_func = jinja2.Template(
    """
{% set PARAMS = {"LINEAR":"union xnn_f32_default_params", "RELU": "union xnn_f32_relu_params", "MINMAX": "union xnn_f32_minmax_params"}[ACTIVATION] %}
void xnn_f32_gemm_ukernel_{{MR}}x{{LMUL}}v_columnwise_pruned__rvv = [](
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    size_t w_stride,
    const float*  w,
    const float*  bias,
    float*  c,
    size_t cm_stride,
    size_t cn_stride,
    const uint16_t* indice,
    const {{ PARAMS }} params[1])
{
  assert(mr != 0);
  assert(mr <= {{MR}});
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a    != NULL);
  assert(bias != NULL);
  assert(w    != NULL);
  assert(c    != NULL);
  {{log_nr}}
  {%- if ACTIVATION == "MINMAX" %}
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  {%- elif ACTIVATION == "RELU" %}
  const float vmin = 0.0f;
  {%- endif %}
  const float* w0 = w;
  float*       c0 = c;
  const float* bias0 = bias;
  {%- for M in range(1, MR) %}
  const float* w{{ M }} = (const float*) ((uintptr_t) w{{ M-1 }} + w_stride);
  float*       c{{ M }} = (float*) ((uintptr_t) c{{ M-1 }} + cm_stride);
  const float* bias{{ M }} = (const float*) ((uintptr_t) bias{{ M-1 }} + sizeof(float));
    {%- if M % 2 == 0 %}
  if UNPREDICTABLE(mr <= {{ M }}) {
    {%- elif M + 1 == MR %}
  if UNPREDICTABLE(mr != {{ M+1 }}) {
    {%- else %}
  if UNPREDICTABLE(mr < {{ M+1 }}) {
    {%- endif %}
        w{{ M }}    = w{{ M-1 }};
        c{{ M }}    = c{{ M-1 }};
        bias{{ M }} = bias{{ M-1 }};
      }
  {%- endfor %}
  const size_t nr = __riscv_vsetvlmax_e32m{{ LMUL }}();
  size_t vl = nr;
  do {
    if UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m{{LMUL}}(nc);
    }
    nc -= vl;
    {%- for M in range(MR) %}
    vfloat32m{{ LMUL }}_t vacc{{ M }} = __riscv_vfmv_v_f_f32m{{ LMUL }}(*bias{{ M }}, vl);
    {%- endfor %}
    size_t k = w_stride;
    size_t idx_indice_arr = 0;
    do {
      {%- for M in range(MR) %}
      const float vw{{ M }} = *w{{ M }}++;
      {%- endfor %}
      vfloat32m{{ LMUL }}_t vb = __riscv_vle32_v_f32m{{ LMUL }}(a + (indice[idx_indice_arr] << log_nr), vl);
      idx_indice_arr++;
      {%- for M in range(MR) %}
      vacc{{ M }} = __riscv_vfmacc_vf_f32m{{ LMUL }}(vacc{{ M }}, vw{{ M }}, vb, vl);
      {%- endfor %}
      k -= sizeof(float);
    } while (k != 0);
    {%- if ACTIVATION == "MINMAX" %}
      {%- for M in range(MR) %}
    vacc{{ M }} = __riscv_vfmax_vf_f32m{{LMUL}}(vacc{{ M }}, vmin, vl);
      {%- endfor %}
      {%- for M in range(MR) %}
    vacc{{ M }} = __riscv_vfmin_vf_f32m{{ LMUL }}(vacc{{ M }}, vmax, vl);
      {%- endfor %}
    {%- elif ACTIVATION == "RELU" %}
      {%- for M in range(MR) %}
    vacc{{ M }} = __riscv_vfmax_vf_f32m{{ LMUL }}(vacc{{ M }}, vmin, vl);
      {%- endfor %}
    {%- endif %}

    {%- for M in range(MR) %}
    __riscv_vse32_v_f32m{{LMUL}}(c{{ M }}, vacc{{ M }}, vl);
    c{{ M }} = (float*) ((uintptr_t) c{{ M }} + cn_stride);
    {%- endfor %}

    {%- for M in range(MR) %}
    w{{ M }} = (const float*) ((uintptr_t) w{{ M }} - w_stride);
    {%- endfor %}
    a += nr * (kc >> 2);
  } while (nc != 0);
};

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
