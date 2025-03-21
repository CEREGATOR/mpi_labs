/*
 * Программа вычисляет средние значения элементов матрицы по строкам и столбцам
 * с использованием параллельных вычислений через MPI.
 * Проверяет корректность результатов, сравнивая с эталонными значениями из файлов.
 */

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

// Тип обработки: по строкам или по столбцам
enum class eprocess_type
{
    by_rows = 0,
    by_cols
};

// Инициализирует матрицу данными из файла matrix.txt
void InitMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols)
{
    ifstream myfile;
    myfile.open("matrix.txt");
    if (myfile.is_open())
    {
        for (size_t i = 0; i < numb_rows; ++i)
        {
            for (size_t j = 0; j < numb_cols; ++j)
            {
                myfile >> matrix[i][j];
            }
        }
    }
    myfile.close();
}

// Выводит матрицу в консоль (для отладки)
void PrintMatrix(double **matrix, const size_t numb_rows, const size_t numb_cols)
{
    printf("Generated matrix:\n");
    for (size_t i = 0; i < numb_rows; ++i)
    {
        for (size_t j = 0; j < numb_cols; ++j)
        {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

/**
 * Вычисляет средние значения по строкам или столбцам матрицы с использованием MPI.
 *
 * @param proc_type  Способ обработки (строки/столбцы)
 * @param matrix     Исходная матрица
 * @param numb_rows  Количество строк
 * @param numb_cols  Количество столбцов
 * @param average_vals Массив для сохранения вычисленных средних значений
 * @param true_vals  Массив с эталонными значениями для проверки
 */
void FindAverageValues(eprocess_type proc_type, double **matrix, const size_t numb_rows,
                       const size_t numb_cols, double *average_vals, double *true_vals)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (proc_type == eprocess_type::by_rows)
    {
        // Распределение строк между процессами
        int chunk = numb_rows / size;
        int start = rank * chunk;
        int end = (rank == size - 1) ? numb_rows : start + chunk;

        double *local_average = new double[chunk]();

        // Локальное вычисление средних значений для своего блока строк
        for (size_t i = start; i < end; ++i)
        {
            double sum = 0.0;
            for (size_t j = 0; j < numb_cols; ++j)
            {
                sum += matrix[i][j];
            }
            local_average[i - start] = sum / numb_cols;
        }

        // Сбор результатов на процессе 0
        MPI_Gather(
            local_average, chunk, MPI_DOUBLE, // Отправляемые данные
            average_vals, chunk, MPI_DOUBLE,  // Буфер для приема
            0, MPI_COMM_WORLD);

        delete[] local_average;

        // Загрузка эталонных значений для проверки (только на процессе 0)
        if (rank == 0)
        {
            ifstream result1("result_rows.txt");
            for (size_t i = 0; i < numb_rows; ++i)
            {
                result1 >> true_vals[i];
            }
        }
    }
    else if (proc_type == eprocess_type::by_cols)
    {
        // Распределение столбцов между процессами
        int chunk = numb_cols / size;
        int start = rank * chunk;
        int end = (rank == size - 1) ? numb_cols : start + chunk;

        double *local_average = new double[chunk]();

        // Локальное вычисление средних значений для своего блока столбцов
        for (size_t j = start; j < end; ++j)
        {
            double sum = 0.0;
            for (size_t i = 0; i < numb_rows; ++i)
            {
                sum += matrix[i][j];
            }
            local_average[j - start] = sum / numb_rows;
        }

        // Сбор результатов на процессе 0
        MPI_Gather(
            local_average, chunk, MPI_DOUBLE,
            average_vals, chunk, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        delete[] local_average;

        // Загрузка эталонных значений для проверки
        if (rank == 0)
        {
            ifstream result2("result_cols.txt");
            for (size_t j = 0; j < numb_cols; ++j)
            {
                result2 >> true_vals[j];
            }
        }
    }
    else
    {
        throw("Incorrect process type!");
    }
}

// Сравнивает вычисленные значения с эталонными
void CheckValues(double *average_vals, double *true_vals, const size_t counter)
{
    for (size_t i = 0; i < counter; ++i)
    {
        if (std::abs(average_vals[i] - true_vals[i]) > 1e-9)
        {
            printf("Error! CheckValues\n");
            return;
        }
    }
    printf("Complete! CheckValues\n");
}

int main()
{
    MPI_Init(NULL, NULL); // Инициализация MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand((unsigned)time(0));
    clock_t start, stop;
    start = clock();

    // Параметры матрицы
    const size_t numb_rows = 1000;
    const size_t numb_cols = 1000;

    // Выделение памяти под матрицу
    double **matrix = new double *[numb_rows];
    for (size_t i = 0; i < numb_rows; ++i)
    {
        matrix[i] = new double[numb_cols];
    }

    // Процесс 0 загружает матрицу и рассылает её всем процессам
    if (rank == 0)
    {
        InitMatrix(matrix, numb_rows, numb_cols);
    }
    for (size_t i = 0; i < numb_rows; ++i)
    {
        MPI_Bcast(matrix[i], numb_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Выделение памяти для результатов
    double *average_vals_in_rows = new double[numb_rows]();
    double *average_vals_in_cols = new double[numb_cols]();
    double *true_vals_in_rows = new double[numb_rows];
    double *true_vals_in_cols = new double[numb_cols];

    // Вычисление средних значений
    FindAverageValues(eprocess_type::by_rows, matrix, numb_rows, numb_cols, average_vals_in_rows, true_vals_in_rows);
    FindAverageValues(eprocess_type::by_cols, matrix, numb_rows, numb_cols, average_vals_in_cols, true_vals_in_cols);

    // Проверка результатов и вывод времени работы (только на процессе 0)
    if (rank == 0)
    {
        CheckValues(average_vals_in_rows, true_vals_in_rows, numb_rows);
        CheckValues(average_vals_in_cols, true_vals_in_cols, numb_cols);

        stop = clock();
        cout << endl
             << "Calculations took " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds.\n";
    }

    // Освобождение ресурсов
    for (size_t i = 0; i < numb_rows; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] average_vals_in_rows;
    delete[] average_vals_in_cols;
    delete[] true_vals_in_rows;
    delete[] true_vals_in_cols;

    MPI_Finalize(); // Завершение работы с MPI

    return 0;
}