#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <stdexcept>


#include "cuda_math.h"

#define NUM_THREADS 256


#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

__device__ __forceinline__ int sign(float x) { 
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

__device__ void clip_arc(float2 p0, float2 p1, const float c, float3* p) {
    // p: [y0,0: not clip, 1: clip, -1: drop],[y1,0: not clip, 1: clip, -1: drop]
    const float x0=p0.x;
    const float y0=p0.y;
    const float x1=p1.x;
    const float y1=p1.y;
    
    float Dx = x1-x0;
    float Dy = y1-y0;
    
    float C = c*c*(y0*y0-1)+x0*x0;
    
    if (C>0 && x0<0) {// p0 needs to be clipped
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
        
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(fmaxf(D,0.0f));
        float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// both need to be clipped
            if (D<0||t0<0||t0>1||t1<0||t1>1) {// no intersection
                p[0].z = -1;
                p[1].z = -1;
                return;
            }
            
            p[0].x = Dx*t0+x0;
            p[0].y = Dy*t0+y0;
            p[0].z = 1;
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            return;
        } 
        // only clip p0
        t0 = std::max(std::min(t0,1.0f),0.0f);
        p[0].x = Dx*t0+x0;
        p[0].y = Dy*t0+y0;
        p[0].z = 1;
        p[1].x = x1;
        p[1].y = y1;
        p[1].z = 0;
        return;
    } else {
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
    
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(fmaxf(D,0.0f));
        float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// clip p1
            t1 = std::max(std::min(t1,1.0f),0.0f);
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            p[0].x = x0;
            p[0].y = y0;
            p[0].z = 0;
            return;
        }
    }
    p[0].x = x0;
    p[0].y = y0;
    p[0].z = 0;
    p[1].x = x1;
    p[1].y = y1;
    p[1].z = 0;
    return;
}


__device__ __forceinline__ float int_arc(float y0, float y1) {
    return -0.5*((asinf(y1)-asinf(y0))+(y1*sqrtf(1-y1*y1)-y0*sqrtf(1-y0*y0)));
}

__device__ __forceinline__ float int_line(float x0, float y0, float x1, float y1, const float c) {
    float Dy = y1-y0;
    float Dx = x1-x0;
    float r = sqrtf(Dx*Dx+Dy*Dy);
    if (r<1e-6) { //point zero
        return 0.0f;
    }
    Dx/=r;Dy/=r;
    r = Dy*x0-Dx*y0;
    r = sqrtf(fmaxf(1-r*r,0.0f));
    
    float p0 = (Dx*x0+Dy*y0)/r;
    float p1 = (Dx*x1+Dy*y1)/r;
    
    float disk = Dy*r*r*(p1*sqrtf(1-p1*p1)-p0*sqrtf(1-p0*p0)+asinf(p1)-asinf(p0));
    disk = fabsf(disk)*sign(y1-y0);
    float cube = (x1+x0)*(y1-y0);
    return 0.5*(c*cube-sqrtf(1-c*c)*disk);
}

__device__ __forceinline__ void sort_vert(float2 &p0, float2 &p1, float2 &pc) {
    if (p0.y>=pc.y) {
        if (p1.y>=pc.y) {
            if (p0.x<=p1.x) {
                return;
            }
        }
    } else {
        if (p1.y<pc.y) {
            if (p0.x>=p1.x) {
                return;
            }
        } else {
            return;
        }
    }
    float2 pt = make_float2(p0.x,p0.y);
    p0.x = p1.x;
    p0.y = p1.y;
    p1.x = pt.x;
    p1.y = pt.y;
    return;
}


__global__ void project_triangle_kernel(
    const int32_t B,
    float* P0, float* P1, float* P2,
    const float c,
    float* rets
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx>=B) {
        return;
    }
    P0 += idx*2;
    P1 += idx*2;
    P2 += idx*2;
    rets += idx;
    
    float2 p0 = make_float2(P0[0],P0[1]);
    float2 p1 = make_float2(P1[0],P1[1]);
    float2 p2 = make_float2(P2[0],P2[1]);
    
    /**
    float2 pmid = (p0+p1+p2)/3.0f;
    sort_vert(p1,p0,pmid);
    sort_vert(p2,p0,pmid);
    sort_vert(p2,p1,pmid);
    */
    
    
    // clip all the edges
    float3 p[6];// 3 edges in total
    uint32_t front = 0;
    clip_arc(p0, p1, c, p);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p1, p2, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p2, p0, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    
    // eval all integrals
    float result = 0.0f;
    for (uint32_t i=0; i < front; i++) {
        //if (fabsf(p[i*2].y-p[i*2+1].y)>1e-6f)
        result += int_line(p[i*2].x, p[i*2].y, p[i*2+1].x, p[i*2+1].y,c); // line integral
        
        if (p[i*2+1].z>0 && p[((i+1)%front)*2].z>0) { // need to integate arc
            //if(fabsf(p[i*2+1].y-p[((i+1)%front)*2].y)>1e-6f)
            result += int_arc(p[i*2+1].y,p[((i+1)%front)*2].y);
        }
    }
    rets[0] = fabs(result);
}


__global__ void vis_clip_kernel(
    const int32_t B,
    float* P0, float* P1, float* P2,
    const float c,
    float* rets
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx>=B) {
        return;
    }
    P0 += idx*2;
    P1 += idx*2;
    P2 += idx*2;
    rets += idx*2*3*3;
    
    float2 p0 = make_float2(P0[0],P0[1]);
    float2 p1 = make_float2(P1[0],P1[1]);
    float2 p2 = make_float2(P2[0],P2[1]);
    
    
    // clip all the edges
    float3 p[2];// 3 edges in total
    clip_arc(p0, p1, c, p);
    for (int i=0; i < 2; i++) {
        rets[i*3] = p[i].x;
        rets[i*3+1] = p[i].y;
        rets[i*3+2] = p[i].z;
    }
    rets += 6;
    
    clip_arc(p1, p2, c, p);
    for (int i=0; i < 2; i++) {
        rets[i*3] = p[i].x;
        rets[i*3+1] = p[i].y;
        rets[i*3+2] = p[i].z;
    }
    rets += 6;
    
    clip_arc(p2, p0, c, p);
    for (int i=0; i < 2; i++) {
        rets[i*3] = p[i].x;
        rets[i*3+1] = p[i].y;
        rets[i*3+2] = p[i].z;
    }
}



void project_triangle(
    const torch::Tensor p0, const torch::Tensor p1, const torch::Tensor p2,
    const float c,
    torch::Tensor rets) {
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t B = p0.size(0);
    dim3 blocks = dim3(div_round_up<int32_t>(B,NUM_THREADS),1,1);
    
    project_triangle_kernel<<<blocks,NUM_THREADS,0,stream>>>(
        B,
        p0.data_ptr<float>(),p1.data_ptr<float>(),p2.data_ptr<float>(),
        c,
        rets.data_ptr<float>()
    );
  
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}

void vis_clip(
    const torch::Tensor p0, const torch::Tensor p1, const torch::Tensor p2,
    const float c,
    torch::Tensor rets) {
    
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t B = p0.size(0);
    dim3 blocks = dim3(div_round_up<int32_t>(B,NUM_THREADS),1,1);
    
    vis_clip_kernel<<<blocks,NUM_THREADS,0,stream>>>(
        B,
        p0.data_ptr<float>(),p1.data_ptr<float>(),p2.data_ptr<float>(),
        c,
        rets.data_ptr<float>()
    );
  
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
    
}