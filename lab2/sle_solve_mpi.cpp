/*
 * Программа решает системы линейных уравнений методом Гаусса.
 * Поддерживает два режима работы: последовательный и параллельный (с использованием MPI).
 * Особенности:
 * - Генерация диагонально доминирующих матриц для устойчивости решения
 * - Оптимальное распределение вычислений между процессами в параллельном режиме
 * - Замер времени выполнения для сравнения производительности
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

/**
 * Выводит матрицу системы и вектор правых частей
 * @param A - матрица коэффициентов
 * @param b - вектор правых частей
 */
void print_matrix(const vector<vector<double>> &A, const vector<double> &b)
{
    int n = A.size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << "\t";
        cout << "| " << b[i] << endl;
    }
    cout << endl;
}

/**
 * Последовательная реализация метода Гаусса с выбором ведущего элемента
 * @param A - матрица коэффициентов (модифицируется в процессе)
 * @param b - вектор правых частей (модифицируется в процессе)
 */
void sequential_gaussian_elimination(vector<vector<double>> &A, vector<double> &b)
{
    int n = A.size();

    // Прямой ход метода Гаусса
    for (int i = 0; i < n; i++)
    {
        // Поиск максимального элемента в текущем столбце
        int maxRow = i;
        for (int k = i + 1; k < n; k++)
            if (abs(A[k][i]) > abs(A[maxRow][i]))
                maxRow = k;

        // Обмен строк для улучшения устойчивости
        swap(A[i], A[maxRow]);
        swap(b[i], b[maxRow]);

        // Проверка вырожденности матрицы
        if (fabs(A[i][i]) < 1e-9)
        {
            cerr << "Error: Division by zero detected!" << endl;
            return;
        }

        // Исключение переменных в нижних строках
        for (int k = i + 1; k < n; k++)
        {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }

    // Обратная подстановка
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--)
    {
        x[i] = b[i] / A[i][i];
        for (int k = 0; k < i; k++)
            b[k] -= A[k][i] * x[i];
    }

    cout << "Sequential solution: ";
    for (double xi : x)
        cout << xi << " ";
    cout << endl;
}

/**
 * Параллельная реализация метода Гаусса с использованием MPI
 * @param n    - размер системы
 * @param A    - матрица коэффициентов
 * @param b    - вектор правых частей
 * @param rank - ранг текущего процесса
 * @param size - общее количество процессов
 */
void parallel_gaussian_elimination(int n, vector<vector<double>> &A, vector<double> &b, int rank, int size)
{
    for (int i = 0; i < n; i++)
    {
        // Главный процесс определяет ведущую строку
        int maxRow = i;
        if (rank == 0)
        {
            for (int k = i + 1; k < n; k++)
                if (fabs(A[k][i]) > fabs(A[maxRow][i]))
                    maxRow = k;
        }

        // Рассылка номера ведущей строки всем процессам
        MPI_Bcast(&maxRow, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Синхронизация состояния матрицы
        if (maxRow != i)
        {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
        }

        // Проверка на вырожденность системы
        if (fabs(A[i][i]) < 1e-9)
        {
            if (rank == 0)
                cerr << "Error: Singular matrix!" << endl;
            return;
        }

        // Рассылка ведущей строки всем процессам
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&b[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Распределение строк между процессами
        for (int j = i + 1; j < n; j++)
        {
            if (j % size == rank)
            { // Циклическое распределение строк
                double factor = A[j][i] / A[i][i];
                for (int k = i; k < n; k++)
                    A[j][k] -= factor * A[i][k];
                b[j] -= factor * b[i];
            }
        }
    }

    // Обратная подстановка (только на главном процессе)
    if (rank == 0)
    {
        vector<double> x(n);
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = b[i] / A[i][i];
            for (int j = 0; j < i; j++)
                b[j] -= A[j][i] * x[i];
        }

        cout << "Parallel solution: ";
        for (double xi : x)
            cout << xi << " ";
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    int rank = 0, size = 1;
    bool useMPI = false;

    // Инициализация MPI
    if (MPI_Init(&argc, &argv) == MPI_SUCCESS)
    {
        useMPI = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // Определение размера системы
    int n = (argc > 1) ? atoi(argv[1]) : (useMPI ? 500 : 3);

    // Инициализация данных
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Генерация данных на главном процессе
    if (rank == 0)
    {
        srand(time(0));

        // Создание диагонально доминирующей матрицы
        for (int i = 0; i < n; i++)
        {
            double row_sum = 0;
            for (int j = 0; j < n; j++)
            {
                A[i][j] = rand() % 10 + 1;
                row_sum += abs(A[i][j]);
            }
            A[i][i] = row_sum; // Гарантия сходимости
            b[i] = rand() % 20 + 1;
        }

        if (!useMPI)
            print_matrix(A, b);
    }

    // Замер времени выполнения
    double start_time = useMPI ? MPI_Wtime() : clock();

    // Выбор режима вычислений
    if (useMPI)
    {
        parallel_gaussian_elimination(n, A, b, rank, size);
    }
    else
    {
        sequential_gaussian_elimination(A, b);
    }

    // Вывод времени выполнения
    if (rank == 0)
    {
        double end_time = useMPI ? MPI_Wtime() : clock();
        cout << "Execution time: "
             << (end_time - start_time) / (useMPI ? 1.0 : CLOCKS_PER_SEC)
             << " sec" << endl;
    }

    // Завершение работы MPI
    if (useMPI)
        MPI_Finalize();

    return 0;
}