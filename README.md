A fractal generator running on GPU.

# Usage
Edit parameters in main.cu in order to customise fractal image, then to build project (CMake and CUDA toolkit Required),
```
cmake -B build
cmake --build build
```
Run ./build/main in order to generate fractal. Must have a ppm image viewer installed to view image (.png support work in progress).

# Sample image
## Julia Fractal
Parmaters used:
```C++
// Image Setup
int DIM = 2000;
double scale = 1.5;
int sampleNum = 10;
float colorFreq = 0.009f;
float r_phase = 5.0f;
float g_phase = 5.2f;
float b_phase = 2.0f;
std::string filename = "image.ppm";


// Fractal Setup
complex c(-0.5125, 0.5123);
int depth = 2000;
double threshold = 1000;
```
<img width="330" height="330" alt="image" src="https://github.com/user-attachments/assets/e1d08562-5cbb-4b57-b126-2757adf8bb45" />


