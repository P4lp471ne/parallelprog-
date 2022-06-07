#include <iostream>
#include "mpi.h"
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>

#include "Matrix.h"
#include "MatrixPiece.h"

int procCount = -1;
int procRank = -1;

const int ROW_SWAP_DIAGONAL_TO_MAX = 1;

const int MATRIX_SEED = 12345;

bool inline isMainProcess() {
	return procRank == 0;
}

int askMatrixSize() {
	int size = -1;

	if (isMainProcess()) {
		std::cout << "Matrix size: ";

		while (size <= 0) {
			std::cin >> size;
			if (size <= 0) {
				std::cout << std::endl << "Enter positive integer" << std::endl;
			}
		}
	}

	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return size;
}

template<typename T>
void initializeMainMatrix(Matrix<T>** mat, int size) {
	if (isMainProcess()) {
		*mat = new Matrix<T>(size);
		fillWithRandom(MATRIX_SEED, size, (*mat)->main);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void fillPiece(Matrix<T>* mat, MatrixPiece<T>* piece) {
	int* sendDispl = nullptr;
	int* sendCount = nullptr;

	if (isMainProcess()) {
		sendDispl = new int[piece->pieceCount];
		sendCount = new int[piece->pieceCount];
	}

	int distribNeeded = (piece->originSize + piece->pieceCount - 1) / piece->pieceCount;
	for (int i = 0; i < distribNeeded; i++) {
		if (isMainProcess()) {
			for (int j = 0; j < piece->pieceCount; j++) {
				sendDispl[j] = piece->originSize * (i * piece->pieceCount + j);
				sendCount[j] = (i * piece->pieceCount + j < piece->originSize) ? piece->originSize : 0;
			}
		}

		T* sendBuff = (mat == nullptr) ? nullptr : mat->main;
		T* receiveBuf = piece->main + i * piece->originSize;
		int receiveCount = (i * piece->pieceCount + piece->pieceIndex < piece->originSize) ? piece->originSize : 0;

		MPI_Scatterv(sendBuff, sendCount, sendDispl, MPI_DOUBLE, receiveBuf, receiveCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	if (isMainProcess()) {
		delete[] sendDispl;
		delete[] sendCount;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void getPieceMaxRow(int step, MatrixPiece<T>* piece, int& maxRow, T& maxValue) {
	int pieceLeadingRow = (step + piece->pieceCount - piece->pieceIndex - 1) / piece->pieceCount;

	maxValue = std::numeric_limits<T>::min();
	int localMaxRow = -1;
	for (int i = pieceLeadingRow; i < piece->calcPieceHeight(); i++) {
		T value = piece->main[piece->originSize * i + step];
		if (localMaxRow < 0 || std::abs(value) > std::abs(maxValue)) {
			maxValue = value;
			localMaxRow = i;
		}
	}

	maxRow = (localMaxRow < 0) ? -1 : piece->getWorldPieceI(localMaxRow);
}

template<typename T>
void getWorldMaxRow(MatrixPiece<T>* piece, int localMaxRow, T localMaxVal, int& worldMaxRow) {
	int* piecesMaxRows = nullptr;
	T* piecesMaxValues = nullptr;

	if (isMainProcess()) {
		piecesMaxRows = new int[piece->pieceCount];
		piecesMaxValues = new T[piece->pieceCount];
	}

	MPI_Gather(&localMaxRow, 1, MPI_INT, piecesMaxRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&localMaxVal, 1, MPI_DOUBLE, piecesMaxValues, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	worldMaxRow = -1;
	T worldMaxElement = std::numeric_limits<T>::min();
	if (isMainProcess()) {
		for (int i = 0; i < piece->pieceCount; i++) {
			if (worldMaxRow < 0 || std::abs(worldMaxElement) < std::abs(piecesMaxValues[i])) {
				worldMaxElement = piecesMaxValues[i];
				worldMaxRow = piecesMaxRows[i];
			}
		}

		delete[] piecesMaxRows;
		delete[] piecesMaxValues;
	}

	MPI_Bcast(&worldMaxRow, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

template<typename T>
void swapRowsFromMaxPiece(MatrixPiece<T>* piece, int worldMaxRow, int destPiece) {
	T* diag = new T[piece->originSize];
	MPI_Recv(diag, piece->originSize, MPI_DOUBLE, destPiece, ROW_SWAP_DIAGONAL_TO_MAX, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

	int offset = piece->getLocalPieceI(worldMaxRow) * piece->originSize;
	std::copy(diag, diag + piece->originSize, piece->main + offset);

	delete[] diag;
}

template<typename T>
void swapRowsFromDestPiece(MatrixPiece<T>* piece, T* maxRow, int step, int maxPiece) {
	int offset = piece->getLocalPieceI(step) * piece->originSize;

	T* diag = new T[piece->originSize];
	std::copy(piece->main + offset, piece->main + offset + piece->originSize, diag);

	MPI_Send(diag, piece->originSize, MPI_DOUBLE, maxPiece, ROW_SWAP_DIAGONAL_TO_MAX, MPI_COMM_WORLD);

	std::copy(maxRow, maxRow + piece->originSize, piece->main + offset);

	delete[] diag;
}

template<typename T>
void selectLeadingRow(int step, MatrixPiece<T>* piece, T* maxRow, bool& swapHappened) {
	int localMaxRow, worldMaxRow;
	T localMaxValue;

	getPieceMaxRow(step, piece, localMaxRow, localMaxValue);
	getWorldMaxRow(piece, localMaxRow, localMaxValue, worldMaxRow);
	swapHappened = (worldMaxRow != step);

	int maxPiece = worldMaxRow % piece->pieceCount;
	int destPiece = step % piece->pieceCount;
	int destLocalRow = step / piece->pieceCount;

	if (piece->pieceIndex == maxPiece) {
		int offset = piece->getLocalPieceI(worldMaxRow) * piece->originSize;
		std::copy(piece->main + offset, piece->main + offset + piece->originSize, maxRow);
	}

	MPI_Bcast(maxRow, piece->originSize, MPI_DOUBLE, maxPiece, MPI_COMM_WORLD);

	if (destPiece == maxPiece) {
		if (piece->pieceIndex == destPiece) {
			piece->swapRows(step, worldMaxRow);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	if (piece->pieceIndex == maxPiece) {
		swapRowsFromMaxPiece(piece, worldMaxRow, destPiece);
	}

	if (piece->pieceIndex == destPiece) {
		swapRowsFromDestPiece(piece, maxRow, step, maxPiece);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void makeIteration(int step, MatrixPiece<T>* piece, T* maxRow) {
	T leadElementInv = 1 / maxRow[step];
	for (int i = step; i < piece->originSize; i++) {
		maxRow[i] *= leadElementInv;
	}
	maxRow[step] = 1;

	int pieceLeadingRow = (step + piece->pieceCount - piece->pieceIndex - 1) / piece->pieceCount;
	for (int i = pieceLeadingRow; i < piece->calcPieceHeight(); i++) {
		int offset = i * piece->originSize;

		if (piece->getWorldPieceI(i) == step) {
			std::copy(maxRow, maxRow + piece->originSize, piece->main + offset);
			continue;
		}

		T rowLead = (piece->main + offset)[step];
		for (int j = step; j < piece->originSize; j++) {
			(piece->main + offset)[j] -= rowLead * maxRow[j];
		}

		(piece->main + offset)[step] = 0;
	}
}

template<typename T>
bool tryMakeFirstGauseStep(MatrixPiece<T>* piece, T& determinant) {
	determinant = 1;
	bool swapHappened;

	T* maxRow = new T[piece->originSize];

	for (int i = 0; i < piece->originSize; i++) {
		selectLeadingRow(i, piece, maxRow, swapHappened);

		determinant *= maxRow[i];
		determinant *= (swapHappened) ? -1 : 1;

		if (isZero(maxRow[i])) {
			delete[] maxRow;
			return false;
		}

		makeIteration(i, piece, maxRow);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	delete[] maxRow;

	MPI_Barrier(MPI_COMM_WORLD);
	return true;
}

template<typename T>
void evaluateDet(int size, Matrix<T>* mat, T& det) {
	MatrixPiece<double>* piece = new MatrixPiece<double>(size, procRank, procCount);
	fillPiece(mat, piece);

	if (!tryMakeFirstGauseStep(piece, det)) {
		det = 0;
		delete piece;
		return;
	}

	delete piece;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

	int size = askMatrixSize();

	double start = MPI_Wtime();

	double det;
	Matrix<double>* mat = nullptr;

	initializeMainMatrix(&mat, size);
	evaluateDet(size, mat, det);

	MPI_Barrier(MPI_COMM_WORLD);

	double end = MPI_Wtime();

	if (isMainProcess()) {
		std::cout << det << std::endl;
		std::cout << end - start << std::endl;
		delete mat;
	}

	MPI_Finalize();
	return 0;
}
