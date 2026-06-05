#include <cuda_runtime.h>

#include "complex.cuh"
#include "fractal.cuh"
#include "render.cuh"


// Image Setup
int DIM = 1584;
double scale = 1;
int sampleNum = 10;
float colorFreq = 0.005f;
float r_phase = 45.0f;
float g_phase = 23.0f;
float b_phase = 71.0f;
colourAlgos algo = colourAlgos::GLOW;
std::string filename = "image.ppm";


// Fractal Setup
complex c(0.4, 0.4);
int depth = 3000;
double threshold = 1000;

int main() {
   // Setup Image Parameters
   imageData d = {
       DIM, scale, depth, sampleNum, colorFreq, r_phase, g_phase, b_phase, algo
   };

   // Setup fractal type and parameters
   Julia f(c, threshold, depth);

   // Get Render Object
   Renderer r(d, filename);
   r.render(f);
   
   return 0;
}
