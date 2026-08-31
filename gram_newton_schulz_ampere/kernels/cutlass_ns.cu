#include <cstdint>
#include <type_traits>
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "tvm_ffi_utils.h"


namespace gram_newton_schulz_ampere {

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

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
      {reinterpret_cast<Element const*>(left.data_ptr()), inner},
      left.stride(0),
      {reinterpret_cast<Element const*>(right.data_ptr()),
       right_leading_dimension},
      right.stride(0),
      {reinterpret_cast<Element const*>(accumulator.data_ptr()), columns},
      accumulator.stride(0),
      {reinterpret_cast<Element*>(output.data_ptr()), columns},
      output.stride(0),
      {static_cast<float>(alpha), static_cast<float>(beta)}, batch_count);

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

template <typename Element, typename LayoutB>
void dispatch_strat(TensorView accumulator, TensorView left, TensorView right,
                     TensorView output, double alpha, double beta,
                     int64_t strat, cudaStream_t stream) {
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
    dispatch_strat <Element, ColumnMajor>(accumulator, left, right, output,
                                         alpha, beta, strat, stream);
  } else {
    dispatch_strat<Element, RowMajor>(accumulator, left, right, output, alpha,
                                      beta, strat, stream);
  }
}

void cutlass_baddbmm(TensorView accumulator, TensorView left, TensorView right,
                     TensorView output, double alpha, double beta,
                     int64_t strat, bool right_column_major) {
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
  CHECK_CONTIGUOUS(left);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK_EQ(left.size(0), right.size(0));
  TVM_FFI_ICHECK_EQ(left.size(0), accumulator.size(0));
  TVM_FFI_ICHECK_EQ(left.size(0), output.size(0));
  TVM_FFI_ICHECK_EQ(left.size(2), right.size(1));
  TVM_FFI_ICHECK_EQ(left.size(1), accumulator.size(1));
  TVM_FFI_ICHECK_EQ(right.size(2), accumulator.size(2));
  CHECK_SHAPE(accumulator, output);
  TVM_FFI_ICHECK_EQ(left.stride(2), 1);
  TVM_FFI_ICHECK_EQ(accumulator.stride(2), 1);
  TVM_FFI_ICHECK_EQ(output.stride(2), 1);
  if (right_column_major) {
    TVM_FFI_ICHECK_EQ(right.stride(1), 1);
    TVM_FFI_ICHECK_EQ(right.stride(2), right.size(1));
  } else {
    TVM_FFI_ICHECK_EQ(right.stride(2), 1);
    TVM_FFI_ICHECK_EQ(right.stride(1), right.size(2));
  }

  ffi::CUDADeviceGuard guard(left.device().device_id);
  cudaStream_t const stream = get_stream(left.device());
  switch (encode_dlpack_dtype(left.dtype())) {
    case float16_code:
      dispatch_layout<cutlass::half_t>(accumulator, left, right, output, alpha,
                                       beta, strat, right_column_major, stream);
      return;
    case bfloat16_code:
      dispatch_layout<cutlass::bfloat16_t>(
          accumulator, left, right, output, alpha, beta, strat,
          right_column_major, stream);
      return;
    default:
      TVM_FFI_LOG_AND_THROW(TypeError)
          << "CUTLASS SM80 baddbmm requires float16 or bfloat16 tensors (or I guess fp32 but we aren't doing that)";
  }
}

}  // namespace gram_newton_schulz_ampere

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    cutlass_baddbmm, gram_newton_schulz_ampere::cutlass_baddbmm);
