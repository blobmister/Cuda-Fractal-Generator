#ifndef COMPLEX_H
#define COMPLEX_H

#include <cuda_runtime.h>
#include <cwctype>

class complex {
private:
    double m_real;
    double m_imag;

public:
    __host__ __device__ complex(double r = 0.0, double i = 0.0) : m_real(r), m_imag(i) {}

    __host__ __device__ double real() const { return m_real; }
    __host__ __device__ double imaginary() const { return m_imag; }

    __host__ __device__ complex& operator+=(const complex& other) {
        m_real += other.m_real;
        m_imag += other.m_imag;
        return *this;
    }

    __host__ __device__ complex& operator-=(const complex& other) {
        m_real -= other.m_real;
        m_imag -= other.m_imag;
        return *this;
    }

    __host__ __device__ complex& operator*=(const complex& other) {
        double new_real = m_real * other.m_real - m_imag * other.m_imag;
        m_imag = m_real * other.m_imag + m_imag * other.m_real;
        m_real = new_real;
        return *this;
    }

    __host__ __device__ double const mag_sq() const {
        return m_real * m_real + m_imag * m_imag;
    }

    __host__ __device__ complex& operator/=(const complex& other);

    __host__ __device__ complex operator-() const { return complex(-m_real, -m_imag); }
};


__host__ __device__ inline complex operator+(complex lhs, const complex& rhs) { return lhs += rhs; }
__host__ __device__ inline complex operator-(complex lhs, const complex& rhs) { return lhs -= rhs; }
__host__ __device__ inline complex operator*(complex lhs, const complex& rhs) { return lhs *= rhs; }
__host__ __device__ inline complex operator/(complex lhs, const complex& rhs) { return lhs /= rhs; }

__host__ __device__ inline complex operator*(complex z, double scalar) { return complex(z.real() * scalar, z.imaginary() * scalar); }
__host__ __device__ inline complex operator*(double scalar, complex z) { return z * scalar; }
__host__ __device__ inline complex operator/(complex z, double scalar) { return complex(z.real() / scalar, z.imaginary() / scalar); }
__host__ __device__ inline complex operator/(double scalar, complex z) { return scalar * (complex(1, 0)/z); }


__host__ __device__ inline complex conjugate(const complex& z) {
    return complex(z.real(), -z.imaginary());
}

__host__ __device__ inline complex& complex::operator/=(const complex& other) {
    *this *= conjugate(other);
    *this = *this / other.mag_sq();
    return *this;
}

#endif
