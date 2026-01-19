#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <iostream>
#include <chrono>

#include "complex.cuh"
#include "fractal.cuh"

struct Color {
    int r;
    int g;
    int b;
};

struct ImageData {
    int dim;
    double scale;
    int depth;
    int sampleNum;
};

using color = struct Color;
using imageData = struct ImageData;

/*
* Kernel code that is run on GPU. 
*
* Takes in an array, the fractal type (with generate function defined) and the image data.
*/
template <typename Fractal>
__global__ static void kernel(Color* ptr, Fractal f, imageData d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= d.dim || i >= d.dim) return;
    int offset = j + i * d.dim;

    float total_hits = 0.0f;

    for (int sub_y = 0; sub_y < d.sampleNum; ++sub_y) {
        for (int sub_x = 0; sub_x < d.sampleNum; ++sub_x) {

            float off_x = (sub_x + 0.5f) / d.sampleNum;
            float off_y = (sub_y + 0.5f) / d.sampleNum;

            float coord_x = j + off_x;
            float coord_y = i + off_y;

            float jx = d.scale * ((d.dim * 1.0f) / 2.0f - coord_x) / ((d.dim * 1.0f) / 2.0f);
            float jy = d.scale * ((d.dim * 1.0f) / 2.0f - coord_y) / ((d.dim * 1.0f) / 2.0f);

            bool inside = f.generate(jx, jy);
            if (inside) total_hits += 1.0f;
        }
    }

    float intensity = total_hits / (d.sampleNum * d.sampleNum);

    ptr[offset].r = 3 * intensity;
    ptr[offset].g = 78 * intensity;
    ptr[offset].b = 252 * intensity;
}

class Renderer {
    public:
        Renderer(imageData data) : d(data) {}

        template <typename Fractal>
        void render(Fractal f) {
            size_t num_bytes = d.dim * d.dim * sizeof(color);
            std::vector<Color> host_bitmap(d.dim * d.dim);
            Color* device_bitmap;

            std::clog << "Rendering on GPU...\n";
            auto start = std::chrono::high_resolution_clock::now();

            cudaMalloc((void**)&device_bitmap, num_bytes);

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks (
                (d.dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (d.dim + threadsPerBlock.y - 1) / threadsPerBlock.y
            );
            kernel<<<numBlocks, threadsPerBlock>>>(device_bitmap, f, d);

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
            PPM_render(host_bitmap);
        }

    private:
        imageData d;

        void PPM_render(std::vector<color>& bitmap) {
            std::cout << "P3\n" << d.dim << " " << d.dim << "\n255\n";
            for (int i{}; i < d.dim * d.dim; ++i) {
                std::cout << bitmap[i].r << " " << bitmap[i].g << " " << bitmap[i].b << '\n';
            }
        }


        
};


#endif