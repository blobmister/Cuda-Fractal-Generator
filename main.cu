#include <vector>
#include <iostream>
#include "complex.cuh"
#include <chrono>
#include <cuda_runtime.h>

struct Color {
    int r;
    int g;
    int b;
};

using color = struct Color;

int DIM = 1080;
double scale = 1.5;
complex c(-0.8, 0.156);
int depth = 200;
double threshold = 1000;

__global__ void kernel(Color* ptr, int dim, double scale, complex c_const, int depth, double threshold) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= dim || i >= dim) return;
    int offset = j + i * dim;

    float jx = scale * ((dim * 1.0)/2 - j)/((dim * 1.0)/2);
    float jy = scale * ((dim * 1.0)/2 - i)/((dim * 1.0)/2);
    complex a(jx, jy);
    int juliaValue = 1;

    for (int w = 0; w < depth; ++w) {
        a = a*a + c_const;
        if (a.mag_sq() > threshold) {
            juliaValue = 0;
            break;
        }
    }

    ptr[offset].r = 3 * juliaValue;
    ptr[offset].g = 78 * juliaValue;
    ptr[offset].b = 252 * juliaValue;
}



void render(std::vector<color>& bitmap) {
    std::cout << "P3\n" << DIM << " " << DIM << "\n255\n";
    for (int i{}; i < DIM * DIM; ++i) {
        std::cout << bitmap[i].r << " " << bitmap[i].g << " " << bitmap[i].b << '\n';
    }
}



int main() {
   size_t num_bytes = DIM * DIM * sizeof(color);
   std::vector<Color> host_bitmap(DIM * DIM);

   Color* device_bitmap;

   std::clog << "Rendering on GPU...\n";
   auto start = std::chrono::high_resolution_clock::now();

   cudaMalloc((void**)&device_bitmap, num_bytes);

   dim3 grid(DIM, DIM);
   kernel<<<grid, 1>>>(device_bitmap, DIM, scale, c, depth, threshold);

   cudaDeviceSynchronize();
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       std::cerr << "Cuda Error: " << cudaGetErrorString(err) << '\n';
   }

   auto end = std::chrono::high_resolution_clock::now();
   auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   std::clog << "GPU Render Time: " << elapsed << "ms\n";

   std::clog << "Printing to PPM...\n";
   cudaMemcpy(host_bitmap.data(), device_bitmap, num_bytes, cudaMemcpyDeviceToHost);
   cudaFree(device_bitmap);
   render(host_bitmap);
   return 0;
}
