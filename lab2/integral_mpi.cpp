#include <iostream>
#include <math.h>
#include <ctime>
#include <mpi.h>  // Добавляем заголовок MPI
#include <chrono>

#define NUM 5

double refer = 2.094395;

double integral(int q, double h) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_sum = 0.0;
    double global_sum = 0.0;

    // Распределение работы между процессами
    int chunk = q / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? q : start + chunk;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Локальные вычисления
    for (int i = start; i < end; i++) {
        double x = h * i + h / 2;
        local_sum += (4.0 / sqrt(4.0 - x * x)) * h;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t2 - t1;

    // Сбор результатов на процессе 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод времени только на root-процессе
    if (rank == 0) {
        std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
    }

    return global_sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Инициализация MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double a = 0, b = 1, res;
    int q[NUM] = {10000, 100000, 1000000, 10000000, 100000000};
    double h[NUM];

    for (int i = 0; i < NUM; i++) {
        h[i] = (b - a) / static_cast<double>(q[i]);
        if (rank == 0) {  // Вывод параметров только на root-процессе
            std::cout << "q = " << q[i] << std::endl;
            std::cout << "h = " << h[i] << std::endl;
        }
        res = integral(q[i], h[i]);
        if (rank == 0) {  // Вывод результатов только на root-процессе
            std::cout << "res = " << res << std::endl;
            std::cout << "err = " << fabs(res - refer) << "\n\n";
        }
    }

    MPI_Finalize();  // Завершение MPI
    return 0;
}