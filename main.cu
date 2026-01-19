#include <cuda_runtime.h>

#include "complex.cuh"
#include "fractal.cuh"
#include "render.cuh"


// Global Constants for Quick Setup.
int DIM = 10000;
double scale = 1.5;
complex c(-0.5125, 0.5123);
int depth = 2000;
double threshold = 1000;
int sampleNum = 10;
float colorFreq = 0.1f;
std::string filename = "image.ppm";

int main() {
   // Setup Image Parameters
   imageData d = {
       DIM, scale, depth, sampleNum, colorFreq
   };

   // Setup fractal type and parameters
   Julia f(c, threshold, depth);

   // Get Render Object
   Renderer r(d, filename);
   r.render(f);
   
   return 0;
}
