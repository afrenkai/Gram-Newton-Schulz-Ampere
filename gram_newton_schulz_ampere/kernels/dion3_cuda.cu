#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* shared_warps) {
  int const lane = threadIdx.x % 32;
  int const warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0) {
    shared_warps[warp] = value;
  }
  __syncthreads();
  float block_value = threadIdx.x < blockDim.x / 32 ? shared_warps[lane] : 0.0f;
  if (warp == 0) {
    block_value = warp_sum(block_value);
  }
  return block_value;
}

template <typename scalar_t>
__global__ void momentum_row_norm_kernel(scalar_t* momentum,
                                         scalar_t const* gradient,
                                         scalar_t* parameter,
                                         float parameter_decay,
                                         float* row_norms,
                                         int64_t columns) {
  int64_t const row = blockIdx.x;
  int64_t const row_offset = row * columns;
  float local_norm = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    int64_t const element = row_offset + column;
    float const updated = static_cast<float>(momentum[element]) +
                          static_cast<float>(gradient[element]);
    scalar_t const stored = static_cast<scalar_t>(updated);
    momentum[element] = stored;
    if (parameter_decay != 1.0f) {
      parameter[element] = static_cast<scalar_t>(
          static_cast<float>(parameter[element]) * parameter_decay);
    }
    local_norm += fabsf(static_cast<float>(stored));
  }
  __shared__ float shared_warps[8];
  float const reduced_norm = block_sum(local_norm, shared_warps);
  if (threadIdx.x == 0) {
    row_norms[row] = reduced_norm;
  }
}

template <typename scalar_t>
__global__ void gather_decay_kernel(scalar_t* momentum,
                                    int64_t const* indices,
                                    at::BFloat16* selected,
                                    float decay,
                                    int64_t rows,
                                    int64_t selected_rows,
                                    int64_t columns) {
  int64_t const selected_linear_row = blockIdx.x;
  int64_t const batch = selected_linear_row / selected_rows;
  int64_t const selected_row = selected_linear_row % selected_rows;
  int64_t const row = indices[selected_linear_row];
  if (row < 0 || row >= rows) {
    return;
  }
  int64_t const momentum_offset = (batch * rows + row) * columns;
  int64_t const selected_offset = selected_linear_row * columns;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float const value = static_cast<float>(momentum[momentum_offset + column]);
    selected[selected_offset + column] = at::BFloat16(value);
    momentum[momentum_offset + column] = static_cast<scalar_t>(value * decay);
  }
}

__global__ void normalize_rows_kernel(at::BFloat16 const* selected_update,
                                      float* variance,
                                      int64_t const* indices,
                                      float* normalized_update,
                                      float* row_squared_norms,
                                      float beta_two,
                                      float epsilon,
                                      int64_t rows,
                                      int64_t selected_rows,
                                      int64_t columns) {
  int64_t const selected_linear_row = blockIdx.x;
  int64_t const batch = selected_linear_row / selected_rows;
  int64_t const row = indices[selected_linear_row];
  if (row < 0 || row >= rows) {
    return;
  }
  int64_t const selected_offset = selected_linear_row * columns;
  float local_squared_norm = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float const value = static_cast<float>(selected_update[selected_offset + column]);
    local_squared_norm += value * value;
  }
  __shared__ float shared_warps[8];
  __shared__ float denominator;
  float const row_squared_norm = block_sum(local_squared_norm, shared_warps);
  if (threadIdx.x == 0) {
    float const neuron_variance = row_squared_norm / static_cast<float>(columns);
    int64_t const variance_offset = batch * rows + row;
    float const updated_variance =
        beta_two * variance[variance_offset] + (1.0f - beta_two) * neuron_variance;
    variance[variance_offset] = updated_variance;
    denominator = sqrtf(updated_variance) + epsilon;
    row_squared_norms[selected_linear_row * 2] = row_squared_norm;
  }
  __syncthreads();

  float local_normalized_squared_norm = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float const value = static_cast<float>(selected_update[selected_offset + column]);
    float const normalized = value / denominator;
    normalized_update[selected_offset + column] = normalized;
    local_normalized_squared_norm += normalized * normalized;
  }
  float const normalized_squared_norm =
      block_sum(local_normalized_squared_norm, shared_warps);
  if (threadIdx.x == 0) {
    row_squared_norms[selected_linear_row * 2 + 1] = normalized_squared_norm;
  }
}

__global__ void reduce_squared_norms_kernel(float const* row_squared_norms,
                                            float* squared_norms,
                                            int64_t selected_rows) {
  int64_t const batch = blockIdx.x;
  int64_t const batch_offset = batch * selected_rows * 2;
  float original_sum = 0.0f;
  float normalized_sum = 0.0f;
  for (int64_t selected_row = threadIdx.x; selected_row < selected_rows;
       selected_row += blockDim.x) {
    int64_t const row_offset = batch_offset + selected_row * 2;
    original_sum += row_squared_norms[row_offset];
    normalized_sum += row_squared_norms[row_offset + 1];
  }
  __shared__ float shared_warps[8];
  float const original_squared_norm = block_sum(original_sum, shared_warps);
  if (threadIdx.x == 0) {
    squared_norms[batch * 2] = original_squared_norm;
  }
  __syncthreads();
  float const normalized_squared_norm = block_sum(normalized_sum, shared_warps);
  if (threadIdx.x == 0) {
    squared_norms[batch * 2 + 1] = normalized_squared_norm;
  }
}

template <typename scalar_t>
__global__ void decay_parameter_kernel(scalar_t* parameter,
                                       float decay,
                                       int64_t elements) {
  for (int64_t element = blockIdx.x * blockDim.x + threadIdx.x;
       element < elements;
       element += blockDim.x * gridDim.x) {
    parameter[element] =
        static_cast<scalar_t>(static_cast<float>(parameter[element]) * decay);
  }
}

template <typename scalar_t>
__global__ void apply_rows_kernel(scalar_t* parameter,
                                  float const* normalized_update,
                                  int64_t const* indices,
                                  float const* squared_norms,
                                  float adjusted_learning_rate,
                                  float epsilon,
                                  int64_t rows,
                                  int64_t selected_rows,
                                  int64_t columns) {
  int64_t const selected_linear_row = blockIdx.x;
  int64_t const batch = selected_linear_row / selected_rows;
  int64_t const row = indices[selected_linear_row];
  if (row < 0 || row >= rows) {
    return;
  }
  float const original_norm = sqrtf(squared_norms[batch * 2]);
  float const normalized_norm =
      fmaxf(sqrtf(squared_norms[batch * 2 + 1]), epsilon);
  int64_t const parameter_offset = (batch * rows + row) * columns;
  int64_t const selected_offset = selected_linear_row * columns;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    int64_t const parameter_element = parameter_offset + column;
    float const parameter_value = static_cast<float>(parameter[parameter_element]);
    float const update = normalized_update[selected_offset + column];
    scalar_t const scaled_update = static_cast<scalar_t>(
        update * original_norm / normalized_norm);
    scalar_t const learning_rate_update = static_cast<scalar_t>(
        -adjusted_learning_rate * static_cast<float>(scaled_update));
    parameter[parameter_element] = static_cast<scalar_t>(
        parameter_value + static_cast<float>(learning_rate_update));
  }
}

void validate_matrix(torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() >= 2, name, " must have at least two dimensions");
}

std::vector<int64_t> leading_shape(torch::Tensor const& tensor) {
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end() - 1);
  return shape;
}

}  // namespace

torch::Tensor momentum_row_norm(torch::Tensor momentum, torch::Tensor gradient) {
  validate_matrix(momentum, "momentum");
  validate_matrix(gradient, "gradient");
  TORCH_CHECK(momentum.sizes() == gradient.sizes(),
              "momentum and gradient shapes must match");
  TORCH_CHECK(momentum.scalar_type() == gradient.scalar_type(),
              "momentum and gradient dtypes must match");
  c10::cuda::CUDAGuard const device_guard(momentum.device());
  int64_t const columns = momentum.size(-1);
  int64_t const row_count = momentum.numel() / columns;
  torch::Tensor row_norms = torch::empty(
      leading_shape(momentum), momentum.options().dtype(torch::kFloat32));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int const threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      momentum.scalar_type(),
      "momentum_row_norm",
      [&] {
        momentum_row_norm_kernel<scalar_t><<<row_count, threads, 0, stream>>>(
            momentum.data_ptr<scalar_t>(),
            gradient.data_ptr<scalar_t>(),
            nullptr,
            1.0f,
            row_norms.data_ptr<float>(),
            columns);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_norms;
}

torch::Tensor momentum_row_norm_decay(torch::Tensor momentum,
                                             torch::Tensor gradient,
                                             torch::Tensor parameter,
                                             double parameter_decay) {
  validate_matrix(momentum, "momentum");
  validate_matrix(gradient, "gradient");
  validate_matrix(parameter, "parameter");
  TORCH_CHECK(momentum.sizes() == gradient.sizes() &&
                  momentum.sizes() == parameter.sizes(),
              "momentum, gradient, and parameter shapes must match");
  TORCH_CHECK(momentum.scalar_type() == gradient.scalar_type() &&
                  momentum.scalar_type() == parameter.scalar_type(),
              "momentum, gradient, and parameter dtypes must match");
  c10::cuda::CUDAGuard const device_guard(momentum.device());
  int64_t const columns = momentum.size(-1);
  int64_t const row_count = momentum.numel() / columns;
  torch::Tensor row_norms = torch::empty(
      leading_shape(momentum), momentum.options().dtype(torch::kFloat32));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int const threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      momentum.scalar_type(),
      "momentum_row_norm_decay",
      [&] {
        momentum_row_norm_kernel<scalar_t><<<row_count, threads, 0, stream>>>(
            momentum.data_ptr<scalar_t>(),
            gradient.data_ptr<scalar_t>(),
            parameter.data_ptr<scalar_t>(),
            static_cast<float>(parameter_decay),
            row_norms.data_ptr<float>(),
            columns);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return row_norms;
}

torch::Tensor gather_decay(torch::Tensor momentum,
                           torch::Tensor indices,
                           double decay) {
  validate_matrix(momentum, "momentum");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(indices.dim() == momentum.dim() - 1,
              "indices must match momentum batch dimensions");
  c10::cuda::CUDAGuard const device_guard(momentum.device());
  int64_t const rows = momentum.size(-2);
  int64_t const columns = momentum.size(-1);
  int64_t const selected_rows = indices.size(-1);
  int64_t const batch = momentum.numel() / (rows * columns);
  std::vector<int64_t> selected_shape = indices.sizes().vec();
  selected_shape.push_back(columns);
  torch::Tensor selected = torch::empty(
      selected_shape, momentum.options().dtype(torch::kBFloat16));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int const threads = 256;
  int64_t const blocks = batch * selected_rows;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      momentum.scalar_type(),
      "gather_decay",
      [&] {
        gather_decay_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            momentum.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            selected.data_ptr<at::BFloat16>(),
            static_cast<float>(decay),
            rows,
            selected_rows,
            columns);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return selected;
}

std::vector<torch::Tensor> select_rows(torch::Tensor momentum,
                                       torch::Tensor gradient,
                                       torch::Tensor parameter,
                                       int64_t selected_count,
                                       double decay,
                                       double parameter_decay) {
  int64_t const rows = momentum.size(-2);
  TORCH_CHECK(selected_count > 0 && selected_count < rows,
              "selected_count must be between zero and the row count");
  torch::Tensor row_norms =
      momentum_row_norm_decay(momentum, gradient, parameter, parameter_decay);
  auto topk_result = at::topk(row_norms, selected_count, -1, true, false);
  torch::Tensor indices = std::get<1>(topk_result);
  torch::Tensor selected = gather_decay(momentum, indices, decay);
  return {selected, indices};
}

std::vector<torch::Tensor> normalize_rows(torch::Tensor selected_update,
                                          torch::Tensor variance,
                                          torch::Tensor indices,
                                          double beta_two,
                                          double epsilon) {
  validate_matrix(selected_update, "selected_update");
  validate_matrix(variance, "variance");
  TORCH_CHECK(selected_update.scalar_type() == torch::kBFloat16,
              "selected_update must be bfloat16");
  TORCH_CHECK(variance.scalar_type() == torch::kFloat32,
              "variance must be float32");
  TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(),
              "indices must be a contiguous CUDA tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  c10::cuda::CUDAGuard const device_guard(selected_update.device());
  int64_t const rows = variance.size(-2);
  int64_t const selected_rows = selected_update.size(-2);
  int64_t const columns = selected_update.size(-1);
  int64_t batch = 1;
  for (int64_t dimension = 0; dimension < selected_update.dim() - 2;
       ++dimension) {
    batch *= selected_update.size(dimension);
  }
  TORCH_CHECK(indices.numel() == batch * selected_rows,
              "indices and selected_update shapes must match");
  TORCH_CHECK(variance.numel() == batch * rows,
              "variance and selected_update batch dimensions must match");
  if (selected_rows == 0) {
    torch::Tensor normalized = torch::empty(
        selected_update.sizes(), selected_update.options().dtype(torch::kFloat32));
    torch::Tensor squared_norms = torch::zeros(
        {batch, 2}, selected_update.options().dtype(torch::kFloat32));
    return {normalized, squared_norms};
  }
  torch::Tensor normalized = torch::empty(
      selected_update.sizes(), selected_update.options().dtype(torch::kFloat32));
  torch::Tensor row_squared_norms = torch::empty(
      {batch, selected_rows, 2},
      selected_update.options().dtype(torch::kFloat32));
  torch::Tensor squared_norms = torch::empty(
      {batch, 2}, selected_update.options().dtype(torch::kFloat32));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int const threads = 256;
  int64_t const blocks = batch * selected_rows;
  normalize_rows_kernel<<<blocks, threads, 0, stream>>>(
      selected_update.data_ptr<at::BFloat16>(),
      variance.data_ptr<float>(),
      indices.data_ptr<int64_t>(),
      normalized.data_ptr<float>(),
      row_squared_norms.data_ptr<float>(),
      static_cast<float>(beta_two),
      static_cast<float>(epsilon),
      rows,
      selected_rows,
      columns);
  reduce_squared_norms_kernel<<<batch, threads, 0, stream>>>(
      row_squared_norms.data_ptr<float>(),
      squared_norms.data_ptr<float>(),
      selected_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {normalized, squared_norms};
}

void apply_rows(torch::Tensor parameter,
                torch::Tensor normalized_update,
                torch::Tensor indices,
                torch::Tensor squared_norms,
                double learning_rate,
                double weight_decay,
                double adjusted_learning_rate,
                double epsilon) {
  validate_matrix(parameter, "parameter");
  validate_matrix(normalized_update, "normalized_update");
  TORCH_CHECK(normalized_update.scalar_type() == torch::kFloat32,
              "normalized_update must be float32");
  TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(),
              "indices must be a contiguous CUDA tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(squared_norms.is_cuda() && squared_norms.is_contiguous(),
              "squared_norms must be a contiguous CUDA tensor");
  TORCH_CHECK(squared_norms.scalar_type() == torch::kFloat32,
              "squared_norms must be float32");
  c10::cuda::CUDAGuard const device_guard(parameter.device());
  int64_t const rows = parameter.size(-2);
  int64_t const columns = parameter.size(-1);
  int64_t const selected_rows = normalized_update.size(-2);
  int64_t batch = 1;
  for (int64_t dimension = 0; dimension < parameter.dim() - 2; ++dimension) {
    batch *= parameter.size(dimension);
  }
  TORCH_CHECK(normalized_update.numel() == batch * selected_rows * columns,
              "parameter and normalized_update batch dimensions must match");
  TORCH_CHECK(indices.numel() == batch * selected_rows,
              "indices and normalized_update shapes must match");
  TORCH_CHECK(squared_norms.numel() == batch * 2,
              "squared_norms must contain two values per batch");
  if (selected_rows == 0) {
    TORCH_CHECK(parameter.numel() == 0,
                "an empty selected update requires an empty parameter shard");
    return;
  }
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  int const threads = 256;
  int64_t const blocks = batch * selected_rows;
  float const parameter_decay =
      1.0f - static_cast<float>(learning_rate * weight_decay);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      parameter.scalar_type(),
      "apply_rows",
      [&] {
        if (parameter_decay != 1.0f) {
          int64_t const decay_blocks = (parameter.numel() + threads - 1) / threads;
          decay_parameter_kernel<scalar_t><<<decay_blocks, threads, 0, stream>>>(
              parameter.data_ptr<scalar_t>(), parameter_decay, parameter.numel());
        }
        apply_rows_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            parameter.data_ptr<scalar_t>(),
            normalized_update.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            squared_norms.data_ptr<float>(),
            static_cast<float>(adjusted_learning_rate),
            static_cast<float>(epsilon),
            rows,
            selected_rows,
            columns);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void normalize_apply_rows(torch::Tensor parameter,
                          torch::Tensor selected_update,
                          torch::Tensor variance,
                          torch::Tensor indices,
                          double beta_two,
                          double epsilon,
                          double learning_rate,
                          double weight_decay,
                          double adjusted_learning_rate) {
  std::vector<torch::Tensor> normalization =
      normalize_rows(selected_update, variance, indices, beta_two, epsilon);
  apply_rows(parameter,
             normalization[0],
             indices,
             normalization[1],
             learning_rate,
             weight_decay,
             adjusted_learning_rate,
             epsilon);
}

std::vector<torch::Tensor> select_rows_batch(
    std::vector<torch::Tensor> const& momenta,
    std::vector<torch::Tensor> const& gradients,
    std::vector<torch::Tensor> const& parameters,
    std::vector<int64_t> const& selected_counts,
    double decay,
    double parameter_decay) {
  size_t const tensor_count = momenta.size();
  TORCH_CHECK(gradients.size() == tensor_count &&
                  parameters.size() == tensor_count &&
                  selected_counts.size() == tensor_count,
              "batched selection inputs must have equal lengths");
  std::vector<torch::Tensor> outputs;
  outputs.reserve(tensor_count * 2);
  for (size_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
    std::vector<torch::Tensor> selection =
        select_rows(momenta[tensor_index],
                    gradients[tensor_index],
                    parameters[tensor_index],
                    selected_counts[tensor_index],
                    decay,
                    parameter_decay);
    outputs.push_back(selection[0]);
    outputs.push_back(selection[1]);
  }
  return outputs;
}

void normalize_apply_rows_batch(
    std::vector<torch::Tensor> const& parameters,
    std::vector<torch::Tensor> const& selected_updates,
    std::vector<torch::Tensor> const& variances,
    std::vector<torch::Tensor> const& indices,
    double beta_two,
    double epsilon,
    double learning_rate,
    double weight_decay,
    std::vector<double> const& adjusted_learning_rates) {
  size_t const tensor_count = parameters.size();
  TORCH_CHECK(selected_updates.size() == tensor_count &&
                  variances.size() == tensor_count &&
                  indices.size() == tensor_count &&
                  adjusted_learning_rates.size() == tensor_count,
              "batched normalization inputs must have equal lengths");
  for (size_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
    normalize_apply_rows(parameters[tensor_index],
                         selected_updates[tensor_index],
                         variances[tensor_index],
                         indices[tensor_index],
                         beta_two,
                         epsilon,
                         learning_rate,
                         weight_decay,
                         adjusted_learning_rates[tensor_index]);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("momentum_row_norm", &momentum_row_norm);
  module.def("gather_decay", &gather_decay);
  module.def("select_rows", &select_rows);
  module.def("select_rows_batch", &select_rows_batch);
  module.def("normalize_rows", &normalize_rows);
  module.def("apply_rows", &apply_rows);
  module.def("normalize_apply_rows", &normalize_apply_rows);
  module.def("normalize_apply_rows_batch", &normalize_apply_rows_batch);
}
