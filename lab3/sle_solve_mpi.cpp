#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

// Функция для вывода матрицы (только для отладки)
void print_matrix(const vector<vector<double>> &A, const vector<double> &b) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << "\t";
        cout << "| " << b[i] << endl;
    }
    cout << endl;
}

// Последовательный метод Гаусса
void sequential_gaussian_elimination(vector<vector<double>> &A, vector<double> &b) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        // Поиск ведущего элемента
        int maxRow = i;
        for (int k = i + 1; k < n; k++)
            if (abs(A[k][i]) > abs(A[maxRow][i]))
                maxRow = k;

        // Обмен строк
        swap(A[i], A[maxRow]);
        swap(b[i], b[maxRow]);

                // Проверка деления на 0
        if (fabs(A[i][i]) < 1e-9) {
            cerr << "Error / 0" << endl;
            return;
        }

        // Прямой ход метода Гаусса
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }

    // Обратный ход
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i] / A[i][i];
        for (int k = 0; k < i; k++)
            b[k] -= A[k][i] * x[i];
    }

    // Вывод решения
    cout << "Solution (sequential): ";
    for (double xi : x)
        cout << xi << " ";
    cout << endl;
}

// Параллельный метод Гаусса с MPI
void parallel_gaussian_elimination(int n, vector<vector<double>> &A, vector<double> &b, int rank, int size) {
    for (int i = 0; i < n; i++) {
        // Главный процесс ищет ведущий элемент
        int maxRow = i;
        if (rank == 0) {
            for (int k = i + 1; k < n; k++) {
                if (fabs(A[k][i]) > fabs(A[maxRow][i])) {
                    maxRow = k;
                }
            }
        }

        // Рассылка индекса ведущей строки
        MPI_Bcast(&maxRow, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Обмен строк (если необходимо)
        if (maxRow != i) {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
        }

        // Проверка деления на 0
        if (fabs(A[i][i]) < 1e-9) {
            if (rank == 0) {
                cerr << "Error" << endl;
            }
            return;
        }

        // Рассылка ведущей строки
        MPI_Bcast(A[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&b[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int j = i + 1; j < n; j++) {
            if (j % size == rank) { // Только часть процессов обновляет строки
                double factor = A[j][i] / A[i][i];
                for (int k = i; k < n; k++) {
                    A[j][k] -= factor * A[i][k];
                }
                b[j] -= factor * b[i];
            }
        }
    }

    // Обратный ход (выполняет только процесс 0)
    if (rank == 0) {
        vector<double> x(n);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = b[i] / A[i][i];
            for (int j = 0; j < i; j++) {
                b[j] -= A[j][i] * x[i];
            }
        }

        // Вывод результата
        cout << "Solution parallel: ";
        for (double xi : x) {
            cout << xi << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {
    int rank = 0, size = 1;
    bool useMPI = false;

    // Инициализация MPI
    if (MPI_Init(&argc, &argv) == MPI_SUCCESS) {
        useMPI = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // Размер системы
    int n;
    if (argc > 1) {
        n = atoi(argv[1]); // Читаем размер из аргумента командной строки
    } else {
        n = useMPI ? 500 : 3; // 500 уравнений для MPI, 3 уравнения для последовательного режима
    }

    // Инициализация матрицы и вектора
    vector<vector<double>> A(n, vector<double>(n, 0));
    vector<double> b(n, 0);

    if (rank == 0) {
        srand(time(0));

        // Генерация случайной матрицы (диагонально доминирующей для устойчивости)
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10 + 1;
                sum += abs(A[i][j]);
            }
            A[i][i] = sum; // Гарантируем диагональное преобладание
            b[i] = rand() % 20 + 1;
        }

        if (!useMPI) {
            cout << "The original system:" << endl;
            print_matrix(A, b);
        }
    }

    // Засекаем время выполнения
    double start_time = useMPI ? MPI_Wtime() : clock();

    if (useMPI) {
        parallel_gaussian_elimination(n, A, b, rank, size);
    } else {
        sequential_gaussian_elimination(A, b);
    }

    // Вычисляем время выполнения
    double end_time = useMPI ? MPI_Wtime() : clock();
    if (rank == 0) {
        cout << "Lead time: " << (end_time - start_time) / (useMPI ? 1.0 : CLOCKS_PER_SEC) << " sec" << endl;
    }

    // Завершаем работу MPI
    if (useMPI) {
        MPI_Finalize();
    }

    return 0;
}
