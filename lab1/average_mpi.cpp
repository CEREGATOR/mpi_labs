#include <vector>
#include <thread>
#include <stdio.h>
#include <exception>
#include <locale.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <mpi.h>  // Добавлен заголовочный файл MPI

using namespace std;

enum class eprocess_type {
    by_rows = 0,
    by_cols
};

void InitMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols) {
    ifstream myfile;
    myfile.open("matrix.txt");
    if (myfile.is_open()) {
        for (size_t i = 0; i < numb_rows; ++i) {
            for (size_t j = 0; j < numb_cols; ++j) {
                myfile >> matrix[i][j];
            }
        }
    }
    myfile.close();
}

void PrintMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols) {
    printf("Generated matrix:\n");
    for (size_t i = 0; i < numb_rows; ++i) {
        for (size_t j = 0; j < numb_cols; ++j) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Остальные функции (InitMatrix, PrintMatrix, CheckValues) остаются без изменений

void FindAverageValues(eprocess_type proc_type, double **matrix, const size_t numb_rows, 
                      const size_t numb_cols, double *average_vals, double *true_vals) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (proc_type == eprocess_type::by_rows) {
        int chunk = numb_rows / size;
        int start = rank * chunk;
        int end = (rank == size - 1) ? numb_rows : start + chunk;

        // Локальный буфер для частичных результатов
        double *local_average = new double[chunk]();

        // Вычисление частичных средних
        for (size_t i = start; i < end; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < numb_cols; ++j) {
                sum += matrix[i][j];
            }
            local_average[i - start] = sum / numb_cols;
        }

        // Сбор результатов на процессе 0
        MPI_Gather(
            local_average, chunk, MPI_DOUBLE,    // Отправляемые данные
            average_vals, chunk, MPI_DOUBLE,     // Принимающий буфер (только на root)
            0, MPI_COMM_WORLD
        );

        delete[] local_average;

        // Загрузка эталонных значений (только на root)
        if (rank == 0) {
            ifstream result1("result_rows.txt");
            for (size_t i = 0; i < numb_rows; ++i) {
                result1 >> true_vals[i];
            }
        }
    } 
    else if (proc_type == eprocess_type::by_cols) {
        int chunk = numb_cols / size;
        int start = rank * chunk;
        int end = (rank == size - 1) ? numb_cols : start + chunk;

        double *local_average = new double[chunk]();

        for (size_t j = start; j < end; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < numb_rows; ++i) {
                sum += matrix[i][j];
            }
            local_average[j - start] = sum / numb_rows;
        }

        MPI_Gather(
            local_average, chunk, MPI_DOUBLE,
            average_vals, chunk, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        delete[] local_average;

        if (rank == 0) {
            ifstream result2("result_cols.txt");
            for (size_t j = 0; j < numb_cols; ++j) {
                result2 >> true_vals[j];
            }
        }
    } 
    else {
        throw("Incorrect process type!");
    }
}

// main функция остается без изменений (как в предыдущем ответе)

void CheckValues(double *average_vals, double *true_vals, const size_t counter) {
    for (size_t i = 0; i < counter; ++i) {
        if (std::abs(average_vals[i] - true_vals[i]) > 1e-9) {
            printf("Error! CheckValues\n");
            return;
        }
    }
    printf("Complete! CheckValues\n");
}

int main() {
    MPI_Init(NULL, NULL);  // Инициализация MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	srand((unsigned)time(0));

	clock_t start, stop;

	start = clock();
    const size_t numb_rows = 1000;
    const size_t numb_cols = 1000;
    double **matrix = new double*[numb_rows];
    for (size_t i = 0; i < numb_rows; ++i) {
        matrix[i] = new double[numb_cols];
    }

    // Только процесс 0 читает матрицу и рассылает её остальным
    if (rank == 0) {
        InitMatrix(matrix, numb_rows, numb_cols);
    }
    for (size_t i = 0; i < numb_rows; ++i) {
        MPI_Bcast(matrix[i], numb_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double *average_vals_in_rows = new double[numb_rows]();
    double *average_vals_in_cols = new double[numb_cols]();
    double *true_vals_in_rows = new double[numb_rows];
    double *true_vals_in_cols = new double[numb_cols];

    // Вычисления
    FindAverageValues(eprocess_type::by_rows, matrix, numb_rows, numb_cols, average_vals_in_rows, true_vals_in_rows);
    FindAverageValues(eprocess_type::by_cols, matrix, numb_rows, numb_cols, average_vals_in_cols, true_vals_in_cols);

    // Проверка только на процессе 0
    if (rank == 0) {
        CheckValues(average_vals_in_rows, true_vals_in_rows, numb_rows);
        CheckValues(average_vals_in_cols, true_vals_in_cols, numb_cols);

		stop = clock();
		cout << endl
			 << "Calculations took " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds.\n";
    }

    // Освобождение ресурсов
    for (size_t i = 0; i < numb_rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] average_vals_in_rows;
    delete[] average_vals_in_cols;
    delete[] true_vals_in_rows;
    delete[] true_vals_in_cols;

    MPI_Finalize();  // Завершение MPI

    return 0;
}