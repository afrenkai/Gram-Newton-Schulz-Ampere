#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "tvm_ffi_utils.h"
#include <cstdint>
#include <type_traits>

namespace gram_newton_schulz_ampere {

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

CUTLASS_DEVICE
cutlass::MatrixCoord symmetric_tile_pair() {
  int const physical_column =
      cutlass::gemm::threadblock::RematerializeBlockIdxX();
  int const circular_distance =
      cutlass::gemm::threadblock::RematerializeBlockIdxY();
  bool const odd_tile_count =
      static_cast<int>(gridDim.x) == 2 * static_cast<int>(gridDim.y) - 1;
  int const tile_count = odd_tile_count ? static_cast<int>(gridDim.x)
                                        : static_cast<int>(gridDim.x) - 1;

  int first_tile = physical_column;
  int second_tile = physical_column + circular_distance;
  if (physical_column == tile_count) {
    first_tile = circular_distance;
    second_tile = circular_distance + tile_count / 2;
  } else if (second_tile >= tile_count) {
    second_tile -= tile_count;
  }
  return {first_tile < second_tile ? first_tile : second_tile,
          first_tile < second_tile ? second_tile : first_tile};
}

struct SymmetricBatchedThreadblockSwizzle {
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord
  get_tiled_shape(cutlass::gemm::GemmCoord problem_size,
                  cutlass::gemm::GemmCoord tile_size, int batch_count) {
    int const row_tiles =
        (problem_size.m() + tile_size.m() - 1) / tile_size.m();
    int const column_tiles =
        (problem_size.n() + tile_size.n() - 1) / tile_size.n();
    int const grid_batches = batch_count < 65535 ? batch_count : 65535;
    return {row_tiles, column_tiles, grid_batches};
  }

  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(cutlass::gemm::GemmCoord tiled_shape) {
    int const tile_count = tiled_shape.m();
    int const grid_columns = (tile_count & 1) ? tile_count : tile_count + 1;
    int const grid_rows = tile_count / 2 + tile_count % 2;
    return dim3(grid_columns, grid_rows, tiled_shape.k());
  }

  CUTLASS_HOST_DEVICE
  static int get_log_tile(cutlass::gemm::GemmCoord) { return 0; }

  CUTLASS_DEVICE
  static cutlass::gemm::GemmCoord get_tile_offset(int) {
    cutlass::MatrixCoord const tile_pair = symmetric_tile_pair();
    return {tile_pair.row(), tile_pair.column(),
            cutlass::gemm::threadblock::RematerializeBlockIdxZ()};
  }

  CUTLASS_DEVICE
  static int get_batch_idx() {
    return cutlass::gemm::threadblock::RematerializeBlockIdxZ();
  }
};

template <typename Element, typename LayoutB, typename ThreadblockShape,
          typename WarpShape>
void launch_baddbmm(TensorView accumulator, TensorView left, TensorView right,
                    TensorView output, double alpha, double beta,
                    cudaStream_t stream) {
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Element, 128 / cutlass::sizeof_bits<Element>::value, float, float>;
  using Operation = cutlass::gemm::device::GemmBatched<
      Element, RowMajor, Element, LayoutB, Element, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
      WarpShape, InstructionShape, Epilogue,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3, 8,
      8>;

  int const batch_count = static_cast<int>(left.size(0));
  int const rows = static_cast<int>(left.size(1));
  int const inner = static_cast<int>(left.size(2));
  int const columns = static_cast<int>(right.size(2));
  int const right_leading_dimension =
      std::is_same_v<LayoutB, RowMajor> ? columns : inner;

  typename Operation::Arguments arguments(
      {rows, columns, inner},
      {reinterpret_cast<Element const *>(left.data_ptr()), inner},
      left.stride(0),
      {reinterpret_cast<Element const *>(right.data_ptr()),
       right_leading_dimension},
      right.stride(0),
      {reinterpret_cast<Element const *>(accumulator.data_ptr()), columns},
      accumulator.stride(0),
      {reinterpret_cast<Element *>(output.data_ptr()), columns},
      output.stride(0), {static_cast<float>(alpha), static_cast<float>(beta)},
      batch_count);

  cutlass::Status const support_status = Operation::can_implement(arguments);
  TVM_FFI_ICHECK(support_status == cutlass::Status::kSuccess)
      << "CUTLASS SM80 baddbmm cannot implement this alignment: "
      << static_cast<int>(support_status);
  Operation operation;
  cutlass::Status const launch_status = operation(arguments, nullptr, stream);
  TVM_FFI_ICHECK(launch_status == cutlass::Status::kSuccess)
      << "CUTLASS SM80 baddbmm launch failed: "
      << static_cast<int>(launch_status);
}

template <typename Element>
__global__ void mirror_lower_triangle(Element *output, int matrix_size,
                                      int64_t batch_stride) {
  constexpr int tile_size = 32;
  __shared__ Element tile[tile_size][tile_size + 1];

  cutlass::MatrixCoord const tile_pair = symmetric_tile_pair();
  int const row_tile = tile_pair.column();
  int const column_tile = tile_pair.row();
  int const thread_column = static_cast<int>(threadIdx.x);
  int64_t const batch_offset = static_cast<int64_t>(blockIdx.z) * batch_stride;

  for (int thread_row = static_cast<int>(threadIdx.y); thread_row < tile_size;
       thread_row += static_cast<int>(blockDim.y)) {
    int const source_row = column_tile * tile_size + thread_row;
    int const source_column = row_tile * tile_size + thread_column;
    if (source_row < matrix_size && source_column < matrix_size) {
      tile[thread_row][thread_column] =
          output[batch_offset + static_cast<int64_t>(source_row) * matrix_size +
                 source_column];
    }
  }
  __syncthreads();

  for (int thread_row = static_cast<int>(threadIdx.y); thread_row < tile_size;
       thread_row += static_cast<int>(blockDim.y)) {
    int const destination_row = row_tile * tile_size + thread_row;
    int const destination_column = column_tile * tile_size + thread_column;
    bool const is_lower_entry =
        row_tile > column_tile || thread_row > thread_column;
    if (is_lower_entry && destination_row < matrix_size &&
        destination_column < matrix_size) {
      output[batch_offset +
             static_cast<int64_t>(destination_row) * matrix_size +
             destination_column] = tile[thread_column][thread_row];
    }
  }
}

template <typename Element, typename LayoutA, typename LayoutB>
void launch_symmetric_baddbmm(TensorView accumulator, TensorView left,
                              TensorView right, TensorView output, double alpha,
                              double beta, cudaStream_t stream) {
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Element, 128 / cutlass::sizeof_bits<Element>::value, float, float>;
  using Operation = cutlass::gemm::device::GemmBatched<
      Element, LayoutA, Element, LayoutB, Element, RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
      WarpShape, InstructionShape, Epilogue, SymmetricBatchedThreadblockSwizzle,
      3, 8, 8>;

  int const batch_count = static_cast<int>(left.size(0));
  int const rows = static_cast<int>(left.size(1));
  int const inner = static_cast<int>(left.size(2));
  int const columns = static_cast<int>(right.size(2));
  int const left_leading_dimension =
      std::is_same_v<LayoutA, RowMajor> ? inner : rows;
  int const right_leading_dimension =
      std::is_same_v<LayoutB, RowMajor> ? columns : inner;

  typename Operation::Arguments arguments(
      {rows, columns, inner},
      {reinterpret_cast<Element const *>(left.data_ptr()),
       left_leading_dimension},
      left.stride(0),
      {reinterpret_cast<Element const *>(right.data_ptr()),
       right_leading_dimension},
      right.stride(0),
      {reinterpret_cast<Element const *>(accumulator.data_ptr()), columns},
      accumulator.stride(0),
      {reinterpret_cast<Element *>(output.data_ptr()), columns},
      output.stride(0), {static_cast<float>(alpha), static_cast<float>(beta)},
      batch_count);

  cutlass::Status const support_status = Operation::can_implement(arguments);
  TVM_FFI_ICHECK(support_status == cutlass::Status::kSuccess)
      << "Symmetric CUTLASS SM80 baddbmm cannot implement this alignment: "
      << static_cast<int>(support_status);
  Operation operation;
  cutlass::Status const launch_status = operation(arguments, nullptr, stream);
  TVM_FFI_ICHECK(launch_status == cutlass::Status::kSuccess)
      << "Symmetric CUTLASS SM80 baddbmm launch failed: "
      << static_cast<int>(launch_status);

  constexpr int mirror_tile_size = 32;
  int const mirror_tiles = (rows + mirror_tile_size - 1) / mirror_tile_size;
  int const mirror_grid_columns =
      (mirror_tiles & 1) ? mirror_tiles : mirror_tiles + 1;
  int const mirror_grid_rows = mirror_tiles / 2 + mirror_tiles % 2;
  dim3 const mirror_grid(mirror_grid_columns, mirror_grid_rows, batch_count);
  dim3 const mirror_block(mirror_tile_size, 8);
  mirror_lower_triangle<Element><<<mirror_grid, mirror_block, 0, stream>>>(
      reinterpret_cast<Element *>(output.data_ptr()), rows, output.stride(0));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess)
      << "Symmetric CUTLASS SM80 mirror launch failed";
}

template <typename Element, typename LayoutB>
void dispatch_strat(TensorView accumulator, TensorView left, TensorView right,
                    TensorView output, double alpha, double beta, int64_t strat,
                    cudaStream_t stream) {
  if (strat == 0) {
    launch_baddbmm<Element, LayoutB, cutlass::gemm::GemmShape<128, 128, 32>,
                   cutlass::gemm::GemmShape<64, 64, 32>>(
        accumulator, left, right, output, alpha, beta, stream);
    return;
  }
  if (strat == 1) {
    launch_baddbmm<Element, LayoutB, cutlass::gemm::GemmShape<64, 64, 32>,
                   cutlass::gemm::GemmShape<32, 32, 32>>(
        accumulator, left, right, output, alpha, beta, stream);
    return;
  }
  if (strat == 2) {
    launch_baddbmm<Element, LayoutB, cutlass::gemm::GemmShape<64, 128, 32>,
                   cutlass::gemm::GemmShape<32, 64, 32>>(
        accumulator, left, right, output, alpha, beta, stream);
    return;
  }
  TVM_FFI_LOG_AND_THROW(ValueError) << "CUTLASS strat has to be 0, 1, or 2";
}

template <typename Element>
void dispatch_layout(TensorView accumulator, TensorView left, TensorView right,
                     TensorView output, double alpha, double beta,
                     int64_t strat, bool right_column_major,
                     cudaStream_t stream) {
  if (right_column_major) {
    dispatch_strat<Element, ColumnMajor>(accumulator, left, right, output,
                                         alpha, beta, strat, stream);
  } else {
    dispatch_strat<Element, RowMajor>(accumulator, left, right, output, alpha,
                                      beta, strat, stream);
  }
}

template <typename Element>
void dispatch_symmetric_layout(TensorView accumulator, TensorView left,
                               TensorView right, TensorView output,
                               double alpha, double beta,
                               bool left_column_major, bool right_column_major,
                               cudaStream_t stream) {
  if (left_column_major) {
    launch_symmetric_baddbmm<Element, ColumnMajor, RowMajor>(
        accumulator, left, right, output, alpha, beta, stream);
    return;
  }
  if (right_column_major) {
    launch_symmetric_baddbmm<Element, RowMajor, ColumnMajor>(
        accumulator, left, right, output, alpha, beta, stream);
    return;
  }
  launch_symmetric_baddbmm<Element, RowMajor, RowMajor>(
      accumulator, left, right, output, alpha, beta, stream);
}

inline void check_baddbmm_tensors(TensorView accumulator, TensorView left,
                                  TensorView right, TensorView output) {
  CHECK_CUDA(accumulator);
  CHECK_CUDA(left);
  CHECK_CUDA(right);
  CHECK_CUDA(output);
  CHECK_DEVICE(accumulator, left);
  CHECK_DEVICE(accumulator, right);
  CHECK_DEVICE(accumulator, output);
  CHECK_SAME_DTYPE(accumulator, left);
  CHECK_SAME_DTYPE(accumulator, right);
  CHECK_SAME_DTYPE(accumulator, output);
  CHECK_DIM(3, accumulator);
  CHECK_DIM(3, left);
  CHECK_DIM(3, right);
  CHECK_DIM(3, output);
  CHECK_CONTIGUOUS(accumulator);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK_EQ(left.size(0), right.size(0));
  TVM_FFI_ICHECK_EQ(left.size(0), accumulator.size(0));
  TVM_FFI_ICHECK_EQ(left.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(left.size(2), right.size(1));
  TVM_FFI_ICHECK_EQ(left.size(1), accumulator.size(1));
  TVM_FFI_ICHECK_EQ(right.size(2), accumulator.size(2));
  CHECK_SHAPE(accumulator, output);
}

inline void check_matrix_layout(TensorView tensor, bool column_major) {
  if (column_major) {
    TVM_FFI_ICHECK_EQ(tensor.stride(1), 1);
    TVM_FFI_ICHECK_EQ(tensor.stride(2), tensor.size(1));
  } else {
    TVM_FFI_ICHECK_EQ(tensor.stride(2), 1);
    TVM_FFI_ICHECK_EQ(tensor.stride(1), tensor.size(2));
  }
}

void cutlass_baddbmm(TensorView accumulator, TensorView left, TensorView right,
                     TensorView output, double alpha, double beta,
                     int64_t strat, bool right_column_major) {
  check_baddbmm_tensors(accumulator, left, right, output);
  CHECK_CONTIGUOUS(left);
  check_matrix_layout(right, right_column_major);

  ffi::CUDADeviceGuard guard(left.device().device_id);
  cudaStream_t const stream = get_stream(left.device());
  switch (encode_dlpack_dtype(left.dtype())) {
  case float16_code:
    dispatch_layout<cutlass::half_t>(accumulator, left, right, output, alpha,
                                     beta, strat, right_column_major, stream);
    return;
  case bfloat16_code:
    dispatch_layout<cutlass::bfloat16_t>(accumulator, left, right, output,
                                         alpha, beta, strat, right_column_major,
                                         stream);
    return;
  default:
    TVM_FFI_LOG_AND_THROW(TypeError)
        << "CUTLASS SM80 baddbmm requires float16 or bfloat16 tensors (or I "
           "guess fp32 but we aren't doing that)";
  }
}

void cutlass_symmetric_baddbmm(TensorView accumulator, TensorView left,
                               TensorView right, TensorView output,
                               double alpha, double beta,
                               bool left_column_major,
                               bool right_column_major) {
  check_baddbmm_tensors(accumulator, left, right, output);
  TVM_FFI_ICHECK_EQ(accumulator.size(1), accumulator.size(2));
  TVM_FFI_ICHECK(!(left_column_major && right_column_major));
  check_matrix_layout(left, left_column_major);
  check_matrix_layout(right, right_column_major);

  ffi::CUDADeviceGuard guard(left.device().device_id);
  cudaStream_t const stream = get_stream(left.device());
  switch (encode_dlpack_dtype(left.dtype())) {
  case float16_code:
    dispatch_symmetric_layout<cutlass::half_t>(accumulator, left, right, output,
                                               alpha, beta, left_column_major,
                                               right_column_major, stream);
    return;
  case bfloat16_code:
    dispatch_symmetric_layout<cutlass::bfloat16_t>(
        accumulator, left, right, output, alpha, beta, left_column_major,
        right_column_major, stream);
    return;
  default:
    TVM_FFI_LOG_AND_THROW(TypeError)
        << "Symmetric CUTLASS SM80 baddbmm requires float16 or bfloat16 "
           "tensors";
  }
}

} // namespace gram_newton_schulz_ampere

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_baddbmm,
                              gram_newton_schulz_ampere::cutlass_baddbmm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    cutlass_symmetric_baddbmm,
    gram_newton_schulz_ampere::cutlass_symmetric_baddbmm);
