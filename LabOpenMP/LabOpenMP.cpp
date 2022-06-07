#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>

#include "Matrix.h"

const int MATRIX_SEED = 123123;

int askMatrixSize() {
	int size = -1;

	std::cout << "Matrix size: ";

	while (size <= 0) {
		std::cin >> size;
		if (size <= 0) {
			std::cout << std::endl << "Enter positive integer" << std::endl;
		}
	}

	return size;
}

template<typename T>
int findMaxI(int step, Matrix<T>* mat) {
	T maxVal;
	int maxRow = -1;
	for (int i = step; i < mat->size; i++) {
		T val = mat->main[i * mat->size + step];
		if (maxRow == -1 || maxVal < std::abs(val)) {
			maxVal = std::abs(val);
			maxRow = i;
		}
	}

	return maxRow;
}

template<typename T>
void swapRows(int i1, int i2, Matrix<T>* mat) {
	for (int j = 0; j < mat->size; j++) {
		std::swap(mat->main[i1 * mat->size + j], mat->main[i2 * mat->size + j]);
	}
}

template<typename T>
void putMaxRowOnDiag(int step, Matrix<T>* mat, bool* swapped) {
	int maxI = findMaxI(step, mat);
	*swapped = (step != maxI);
	swapRows(step, maxI, mat);
}

template<typename T>
void normalizeLeadingCoef(int step, Matrix<T>* mat) {
	T coefInv = 1 / mat->main[step * mat->size + step];

	for (int j = step; j < mat->size; j++) {
		mat->main[step * mat->size + j] *= coefInv;
	}

	mat->main[step * mat->size + step] = 1;
}

template<typename T>
void firstGausInnerStep(int step, Matrix<T>* mat, T* maxMatRow) {

#pragma omp parallel for

	for (int i = step + 1; i < mat->size; i++) {
		T leadingCoef = mat->main[i * mat->size + step];

		for (int j = step; j < mat->size; j++) {
			mat->main[i * mat->size + j] -= leadingCoef * maxMatRow[j];
		}

		mat->main[i * mat->size + step] = 0;
	}
}

template<typename T>
T evaluateDet(Matrix<T>* mat) {
	T det = 1;
	bool swapped = false;
	T* maxMatRow = new T[mat->size];

	for (int step = 0; step < mat->size; step++) {
		int rowOffset = step * mat->size;
		putMaxRowOnDiag(step, mat, &swapped);

		if (isZero(mat->main[rowOffset + step])) {
			delete[] maxMatRow;
			return 0;
		}

		det *= (mat->main + rowOffset + step)[0];
		det *= (swapped) ? -1 : 1;

		normalizeLeadingCoef(step, mat);
		std::copy(mat->main + rowOffset, mat->main + rowOffset + mat->size, maxMatRow);
		firstGausInnerStep(step, mat, maxMatRow);
	}

	delete[] maxMatRow;
	return det;
}

int main() {
	int size = askMatrixSize();

	double start = omp_get_wtime();

	Matrix<double>* mat = new Matrix<double>(size);
	fillWithRandom(MATRIX_SEED, size, mat->main);
	double det = evaluateDet(mat);

	double end = omp_get_wtime();
	
	std::cout << "Det:  " << det << std::endl;
	std::cout << "Time: " << (end - start) << std::endl;

	delete mat;
	return 0;
}