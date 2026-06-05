A fractal generator running on GPU.

# Usage
Edit parameters in main.cu in order to customise fractal image, then to build project (CMake and CUDA toolkit Required),
```
cmake -B build
cmake --build build
```
Run ./build/main in order to generate fractal. Must have a ppm image viewer installed to view image (.png support work in progress).

# Sample images
Two colouring functions are provided. A psychadelic function (which utilizes 3 sine waves, for which you may set the phase), and a glow function (for which you set a base rgb value). Examples of both are shown. The glow colouring algorithm is written to mimick the visuals shown on the [Wikipedia artical on Julia Fractals](https://en.wikipedia.org/wiki/Julia_set) as closely as possible.

## Image 1
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
colorAlgos algo = colourAlgos::WAVE;
std::string filename = "image.ppm";


// Fractal Setup
complex c(-0.5125, 0.5123);
int depth = 2000;
double threshold = 1000;
```
<img width="2000" height="2000" alt="image" src="https://github.com/user-attachments/assets/d10028ff-79d8-4205-97e2-8545ca01e80d" />

## Image 2
Paramters used:
```C++
// Image Setup
int DIM = 1584;
double scale = 1;
int sampleNum = 10;
float colorFreq = 0.005f;
float r_phase = 2.0f;
float g_phase = 2.0f;
float b_phase = 2.0f;
colorAlgos algo = colourAlgos::WAVE;
std::string filename = "image.ppm";

// Fractal Setup
complex c(0.285, 0.01);
int depth = 3000;
double threshold = 1000;
```
<img width="1584" height="1584" alt="image" src="https://github.com/user-attachments/assets/04e2017b-bf9e-4f31-9869-06a5aff9949b" />

## Image 3
Parameters used:
```C++
// Image Setup
int DIM = 1584;
double scale = 1;
int sampleNum = 10;
float colorFreq = 0.005f;
float r_phase = 8.0f;
float g_phase = 48.0f;
float b_phase = 27.0f;
colourAlgos algo = colourAlgos::GLOW;
std::string filename = "image.ppm";


// Fractal Setup
complex c(-0.835, -0.2321);
int depth = 3000;
double threshold = 1000;
```

<img width="1584" height="1584" alt="image" src="https://github.com/user-attachments/assets/3740eeb3-d1d3-4fe1-8700-0517e94979a9" />

## Image 4
Paramters used:
```C++
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
```

<img width="1584" height="1584" alt="image" src="https://github.com/user-attachments/assets/46ef14ae-700b-449a-8247-60036b71091c" />




