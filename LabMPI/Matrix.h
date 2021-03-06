#pragma once

#include <iostream>
#include <iomanip>

template<typename T>
struct Matrix {
	int size;
	T* main = nullptr;

	Matrix(int size) : size(size) {
		main = new T[size * size];
	}

	~Matrix() {
		delete[] main;
	}
};

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T>& mat) {
	stream << std::scientific << std::setprecision(3);

	for (int i = 0; i < mat.size; i++) {
		stream << std::setw(5) << i << ": ";
		for (int j = 0; j < mat.size; j++) {
			stream << std::setw(12) << mat.main[i * mat.size + j] << " ";
		}
		stream << std::endl;
	}

	return stream;
}

template<typename T>
void fillWithRandom(int seed, int size, T* matrix) {
	srand(seed);
	for (int i = 0; i < size * size; i++) {
		T r = static_cast<T>(rand()) / RAND_MAX;
		matrix[i] = 0.1 * (r - 0.5);
	}
}

template<typename T>
inline bool isZero(T value, T relative = 1) {
	return std::abs(value) < relative * 1e-5;
}