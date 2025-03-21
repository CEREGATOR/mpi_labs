/*
 * Программа вычисляет определенный интеграл функции 4/sqrt(4 - x²) на интервале [0, 1]
 * методом прямоугольников с использованием параллельных вычислений через MPI.
 * Сравнивает результат с эталонным значением и выводит погрешность.
 */

#include <iostream>
#include <math.h>
#include <ctime>
#include <mpi.h>  // Для работы с MPI
#include <chrono> // Для точного замера времени

#define NUM 5 // Количество тестовых случаев

const double refer = 2.094395; // Эталонное значение интеграла

/**
 * Вычисляет интеграл методом средних прямоугольников с распараллеливанием через MPI
 *
 * @param q  Количество интервалов разбиения
 * @param h  Шаг интегрирования
 * @return   Приближенное значение интеграла
 */
double integral(int q, double h)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем ранг текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем общее количество процессов

    double local_sum = 0.0;  // Локальная сумма процесса
    double global_sum = 0.0; // Глобальная сумма после редукции

    // Распределение работы между процессами
    int chunk = q / size;                             // Количество интервалов на процесс
    int start = rank * chunk;                         // Начальный индекс для процесса
    int end = (rank == size - 1) ? q : start + chunk; // Корректировка для последнего процесса

    auto t1 = std::chrono::high_resolution_clock::now(); // Старт замера времени

    // Локальные вычисления интеграла для своего блока интервалов
    for (int i = start; i < end; i++)
    {
        double x = h * i + h / 2;                   // Середина интервала
        local_sum += (4.0 / sqrt(4.0 - x * x)) * h; // Формула метода прямоугольников
    }

    auto t2 = std::chrono::high_resolution_clock::now(); // Окончание замера
    std::chrono::duration<double> duration = t2 - t1;

    // Сбор и суммирование результатов на процессе 0 (MPI_Reduce)
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод времени вычислений только на корневом процессе
    if (rank == 0)
    {
        std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
    }

    return global_sum;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv); // Инициализация MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем ранг процесса

    const double a = 0.0, b = 1.0;                              // Границы интегрирования
    int q[NUM] = {10000, 100000, 1000000, 10000000, 100000000}; // Наборы разбиений
    double h[NUM];                                              // Массив шагов интегрирования

    // Вычисление для разных значений разбиения
    for (int i = 0; i < NUM; i++)
    {
        h[i] = (b - a) / static_cast<double>(q[i]); // Расчет шага

        // Вывод параметров только на корневом процессе
        if (rank == 0)
        {
            std::cout << "q = " << q[i] << std::endl;
            std::cout << "h = " << h[i] << std::endl;
        }

        double res = integral(q[i], h[i]); // Вычисление интеграла

        // Вывод результатов и погрешности только на корневом процессе
        if (rank == 0)
        {
            std::cout << "res = " << res << std::endl;
            std::cout << "err = " << fabs(res - refer) << "\n\n";
        }
    }

    MPI_Finalize(); // Завершение работы с MPI
    return 0;
}