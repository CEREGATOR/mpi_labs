#include <vector>
#include <thread>
#include <stdio.h>
#include <exception>
#include <locale.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

enum class eprocess_type
{
	by_rows = 0,
	by_cols
};

void InitMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols)
{
	ifstream myfile("matrix.txt");
	if (myfile.is_open())
	{
		for (size_t i = 0; i < numb_rows; ++i)
		{
			for (size_t j = 0; j < numb_cols; ++j)
			{
				myfile >> matrix[i][j];
			}
		}
		myfile.close();
	}
}

void LoadTrueValues(const char* filename, double* true_vals, size_t size)
{
	ifstream file(filename);
	if (file.is_open())
	{
		for (size_t i = 0; i < size; ++i)
		{
			file >> true_vals[i];
		}
		file.close();
	}
	else
	{
		throw runtime_error("Cannot open file " + string(filename));
	}
}

void FindAverageValues(eprocess_type proc_type, double **matrix, const size_t numb_rows, const size_t numb_cols, double *average_vals)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	switch (proc_type)
	{
	case eprocess_type::by_rows:
	{
		int rows_per_proc = numb_rows / size;
		int start_row = rank * rows_per_proc;
		int end_row = (rank == size - 1) ? numb_rows : start_row + rows_per_proc;

		for (size_t i = start_row; i < end_row; ++i)
		{
			double sum(0.0);
			for (size_t j = 0; j < numb_cols; ++j)
			{
				sum += matrix[i][j];
			}
			average_vals[i] = sum / numb_cols;
		}

		MPI_Allgather(MPI_IN_PLACE, rows_per_proc, MPI_DOUBLE, average_vals, rows_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
		break;
	}
	case eprocess_type::by_cols:
	{
		int cols_per_proc = numb_cols / size;
		int start_col = rank * cols_per_proc;
		int end_col = (rank == size - 1) ? numb_cols : start_col + cols_per_proc;

		for (size_t j = start_col; j < end_col; ++j)
		{
			double sum(0.0);
			for (size_t i = 0; i < numb_rows; ++i)
			{
				sum += matrix[i][j];
			}
			average_vals[j] = sum / numb_rows;
		}

		MPI_Allgather(MPI_IN_PLACE, cols_per_proc, MPI_DOUBLE, average_vals, cols_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
		break;
	}
	default:
	{
		throw("Incorrect value for parameter 'proc_type' in function FindAverageValues() call!");
	}
	}
}

void CheckValues(double *average_vals, double *true_vals, const size_t counter)
{
	for (size_t i = 0; i < counter; ++i)
	{
		if (std::abs(average_vals[i] - true_vals[i]) > 1e-9)
		{
			printf("average_vals[%zu]: %lf \n", i, average_vals[i]);
			printf("true_vals[%zu]: %lf \n", i, true_vals[i]);
			printf("Error! CheckValues\n");
			return;
		}
	}
	printf("Complete! CheckValues\n");
	return;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const unsigned ERROR_STATUS = -1;
	const unsigned OK_STATUS = 0;
	clock_t start, stop;
	unsigned status = OK_STATUS;

	try
	{
		srand((unsigned)time(0));

		const size_t numb_rows = 1000;
		const size_t numb_cols = 1000;
		start = clock();
		double **matrix = new double *[numb_rows];
		for (size_t i = 0; i < numb_rows; ++i)
		{
			matrix[i] = new double[numb_cols];
		}

		double *average_vals_in_rows = new double[numb_rows];
		double *average_vals_in_cols = new double[numb_cols];
		double *true_vals_in_rows = new double[numb_rows];
		double *true_vals_in_cols = new double[numb_cols];

		InitMatrix(matrix, numb_rows, numb_cols);
		LoadTrueValues("result_rows.txt", true_vals_in_rows, numb_rows);
		LoadTrueValues("result_cols.txt", true_vals_in_cols, numb_cols);

		FindAverageValues(eprocess_type::by_rows, matrix, numb_rows, numb_cols, average_vals_in_rows);
		FindAverageValues(eprocess_type::by_cols, matrix, numb_rows, numb_cols, average_vals_in_cols);

		if (rank == 0)
		{
			CheckValues(average_vals_in_rows, true_vals_in_rows, numb_rows);
			CheckValues(average_vals_in_cols, true_vals_in_cols, numb_cols);
		}

		stop = clock();
		if (rank == 0)
		{
			cout << endl << "Calculations took " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds.\n";
		}

		delete[] matrix;
		delete[] average_vals_in_rows;
		delete[] average_vals_in_cols;
		delete[] true_vals_in_rows;
		delete[] true_vals_in_cols;
	}
	catch (std::exception &except)
	{
		printf("Error occurred!\n");
		std::cout << except.what();
		status = ERROR_STATUS;
	}

	MPI_Finalize();
	return status;
}
