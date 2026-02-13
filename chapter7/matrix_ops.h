#pragma once
#include <vector>
#include <valarray>
#include <cmath>
#include <random>

class Matrix {
private:
    std::valarray<double> data;
    size_t rows_, cols_;

public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data(rows * cols) {}
    
    Matrix(size_t rows, size_t cols, double val) : rows_(rows), cols_(cols), data(val, rows * cols) {}
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    double& operator()(size_t i, size_t j) { return data[i * cols_ + j]; }
    const double& operator()(size_t i, size_t j) const { return data[i * cols_ + j]; }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                double sum = 0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows_, cols_);
        result.data = data + other.data;
        return result;
    }
    
    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        Matrix result(rows_, cols_);
        result.data = data - other.data;
        return result;
    }
    
    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows_, cols_);
        result.data = data * scalar;
        return result;
    }
    
    // Transpose
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    // Element-wise operations using valarray
    Matrix tanh() const {
        Matrix result(rows_, cols_);
        result.data = std::tanh(data);
        return result;
    }
    
    Matrix sigmoid() const {
        Matrix result(rows_, cols_);
        result.data = 1.0 / (1.0 + std::exp(-data));
        return result;
    }
    
    Matrix tanh_derivative() const {
        Matrix result(rows_, cols_);
        auto tanh_vals = std::tanh(data);
        result.data = 1.0 - tanh_vals * tanh_vals;
        return result;
    }
    
    // Hadamard (element-wise) product
    Matrix hadamard(const Matrix& other) const {
        Matrix result(rows_, cols_);
        result.data = data * other.data;
        return result;
    }
    
    // Random initialization
    void randomize(double mean = 0.0, double stddev = 0.1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(mean, stddev);
        
        for (auto& val : data) {
            val = dist(gen);
        }
    }
    
    // Zero initialization
    void zeros() {
        data = 0.0;
    }
};