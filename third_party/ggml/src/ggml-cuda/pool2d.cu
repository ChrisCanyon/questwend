#include "pool2d.cuh"
#include <cstdlib>
#include <cstring>

// Only the explicitly tagged QSA node uses this path. Generic AVG is kept
// unchanged: it scales each input before summation and is not bit-equivalent.
static __global__ void qsa_pool4_left_scale_f32(
        const float * src, float * dst, const int64_t outputs,
        const float scale, const float bias) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    constexpr int width = 128; // Guarded fixed shape; division lowers to shifts.
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < outputs; index += step) {
        const int64_t block = index / width;
        const int column = static_cast<int>(index % width);
        const int64_t base = block * (4 * width) + column;
        // Separate rounded additions reproduce three GGML ADD outputs.
        const float sum01 = __fadd_rn(src[base], src[base + width]);
        const float sum012 = __fadd_rn(sum01, src[base + 2 * width]);
        const float sum = __fadd_rn(sum012, src[base + 3 * width]);
        // Same expression and runtime arguments as scale_f32<float>.
        // Do not turn this into sum*.25 alone: bias/FMA/FTZ affect edge bits.
        dst[index] = scale * sum + bias;
    }
}


template <typename Ti, typename To>
static  __global__ void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / ow;
    const int cur_ow = idx % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;
    To* o_ptr = dst + nc * O_HW;
    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);
    const To scale = 1. / (kh * kw);
    To res = 0;

    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
            Ti cur = __ldg(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
                case GGML_OP_POOL_AVG: res += cur * scale; break;
                case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
                default: assert(false);
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
static __global__ void pool2d_nchw_kernel_warp(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti * __restrict__ src, To * __restrict__ dst, const enum ggml_op_pool op) {
    const int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    if (warp_id >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc     = warp_id / O_HW;
    const int cur_oh = warp_id % O_HW / ow;
    const int cur_ow = warp_id % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;

    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);

    const int win_w     = ew - bw;
    const int win_elems = (eh - bh) * win_w;
    const To scale = 1. / (kh * kw);

    To res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: res = 0; assert(false);
    }

    for (int t = lane; t < win_elems; t += WARP_SIZE) {
        const int i = bh + t / win_w;
        const int j = bw + t % win_w;
        const Ti cur = i_ptr[i * iw + j];
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur * scale; break;
            case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
            default: break;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const To other = __shfl_xor_sync(0xFFFFFFFF, res, offset, WARP_SIZE);
        res = (op == GGML_OP_POOL_MAX) ? max(res, other) : res + other;
    }

    if (lane == 0) {
        dst[nc * O_HW + cur_oh * ow + cur_ow] = res;
    }
}

static void pool2d_nchw_kernel_f32_f32_cuda(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {

    if (kh * kw >= CUDA_POOL2D_WARP_KERNEL_MIN_WINDOW) {
        const int warps_per_block = CUDA_POOL2D_BLOCK_SIZE / WARP_SIZE;
        const int num_blocks = (parallel_elements + warps_per_block - 1) / warps_per_block;
        pool2d_nchw_kernel_warp<<<num_blocks, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
        return;
    }

    const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool2d_nchw_kernel<<<block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    constexpr char exact_prefix[] = "qsa_pool4_exact_";
    const bool exact_node = std::strncmp(dst->name, exact_prefix, sizeof(exact_prefix) - 1) == 0;
    if (exact_node) {
        static const bool enabled = [] {
            const char * value = std::getenv("GGML_CUDA_QSA_POOL4_EXACT");
            return value && std::strcmp(value, "1") == 0;
        }();
        // An exact-tagged node must never silently fall back to generic AVG.
        GGML_ASSERT(enabled && "Exact QSA pool requires the matched opt-in CUDA backend");
        GGML_ASSERT(op == GGML_OP_POOL_AVG && k0 == 1 && k1 == 4 &&
            s0 == 1 && s1 == 4 && p0 == 0 && p1 == 0);
        GGML_ASSERT(src0->ne[0] == 128 && src0->ne[1] >= 4 && src0->ne[1] % 4 == 0 &&
            src0->ne[2] == 1 && src0->ne[3] == 1 &&
            dst->ne[0] == 128 && dst->ne[1] == src0->ne[1] / 4 &&
            dst->ne[2] == 1 && dst->ne[3] == 1 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
        const int64_t outputs = ggml_nelements(dst);
        const int64_t blocks = (outputs + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
        GGML_ASSERT(blocks > 0 && blocks <= 0x7fffffff);
        const ggml_cuda_kernel_launch_params launch = {
            dim3(static_cast<unsigned>(blocks)), dim3(CUDA_POOL2D_BLOCK_SIZE), 0, stream};
        ggml_cuda_kernel_launch(qsa_pool4_left_scale_f32, launch,
            src0_d, dst_d, outputs, 0.25f, 0.0f);
        return;
    }

    const int64_t IH = src0->ne[1];
    const int64_t IW = src0->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;

    pool2d_nchw_kernel_f32_f32_cuda(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
}
