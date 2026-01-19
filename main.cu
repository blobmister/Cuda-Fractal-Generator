#include <cuda_runtime.h>

#include "complex.cuh"
#include "fractal.cuh"
#include "render.cuh"



// Global Constants for Quick Setup.
int DIM = 10000;
double scale = 1.5;
complex c(-0.8, 0.156);
int depth = 2000;
double threshold = 10000;
int sampleNum = 10;

int main() {
   // Setup Image Parameters
   imageData d = {
       DIM, scale, depth, sampleNum
   };

   // Setup fractal type and parameters
   complex c(-0.5125, 0.5123);
   Julia f(c, threshold, depth);

   // Get Render Object
   Renderer r(d);
   r.render(f);
   
   return 0;
}
