#ifndef FRACTAL_H
#define FRACTAL_H

#include <cuda_runtime.h>
#include "complex.cuh"

/*
* 
* File Contains generating logic for basic fractals. To generate a fractal, create an object of the desired fractal
* and pass into the kernel function rendered on the GPU.
* 
* Generally an increased depth and threshold generates a more accurate image (with eventual diminishing returns in quality).
*/

class Julia {
    private:
        complex c_const;
        double threshold;
        int depth;

    public:
        Julia(complex c_const, double threshold, int depth) : 
        c_const(c_const), threshold(threshold), depth(depth) {};

        __device__ bool generate(float jx, float jy) {
            complex a(jx, jy);

            for (int w{}; w < depth; ++w) {
                a = a*a + c_const;
                if (a.mag_sq() > threshold) {
                    return false;
                }
            }

            return true;
        }
};

#endif