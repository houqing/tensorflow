
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

//#include "tensorflow/core/kernels/jamdata_op.h"
#include "tensorflow/core/kernels/matmul_op.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>


#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
namespace {
__global__ void JamDDataCudaKernel(int nthreads, const float* input, float* output, const int bits) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
                output[index] = __uint_as_float(__float_as_uint(input[index]) & (~((1 << bits) - 1)));
        }
}
}	// namespace

template <typename T>
struct JamDData<GPUDevice, T> {
  void operator()(const GPUDevice& d, const int size, const T *in, T *out, int bits) const {
          CudaLaunchConfig config = GetCudaLaunchConfig(size, d);

          JamDDataCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, out, bits);
  }
};

}	// namespace functor

template struct functor::JamDData<GPUDevice, float>;

}	// namespace tensorflow

#endif	// GOOGLE_CUDA
