
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <sstream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <cstring> // For memset
#include "xnnpack.h"
#include "logging.h"


void gemm_rcr_bias_56 (
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* c_ptr,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* c_dim0,
    int64_t* c_dim1,
    pthreadpool* pthreadpool_
  ) {
  
 int64_t M = (*a_dim0);

 int64_t N = (*b_dim0);

 int64_t K = (*a_dim1);
  
  int64_t input_a_batch_stride = M * K;
  int64_t input_a_stride = K;
  int64_t input_a_offset = 0; // default to 0
  int64_t input_b_batch_stride = N * K;
  int64_t input_b_stride = K;
  int64_t input_b_offset = 0; // default to 0
    
  
  
  int64_t output_stride = N;
  int64_t output_offset = 0;
  
    
  
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }

  if (!bias_ptr) {
    throw std::runtime_error("bias_ptr is null!");
  }
  
  if (M == 1 && N == 1000 && K == 2048) {
    


//gemm_bias_9_f32_f32_f32_row_column_row
xnn_operator_t gemm_op = nullptr;
const xnn_status status = xnn_create_fully_connected_nc_f32(
    K, N, K, N, 
    (float*)(b_ptr), (float*)(bias_ptr), 
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    /*flags=*/0, nullptr, nullptr, &gemm_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(gemm_op, xnn_delete_operator);
  CHECK_EQ(xnn_status_success, status);
  CHECK_NE(nullptr, gemm_op);
  CHECK_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32(gemm_op, M, /*threadpool=*/pthreadpool_));
  CHECK_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32(gemm_op, (float*)(a_ptr), (float*)(c_ptr)));
  CHECK_EQ(xnn_status_success, xnn_run_operator(gemm_op, /*threadpool=*/pthreadpool_));


  }
  return;
  throw std::runtime_error(
      "Unsupported workload for this gemm_rcr_bias_56 specialization."
  );
}