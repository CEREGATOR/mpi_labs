#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 500; // Размер системы
    vector<vector<double>> local_A;
    vector<double> local_b;
    vector<int> sendcounts(num_procs, 0);
    vector<int> displs(num_procs + 1, 0);

    // Генерация и распределение данных
    if (rank == 0) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = (i == j) ? 1000.0 + rand() % 100 : rand() % 100; // Диагональное преобладание
            }
            b[i] = rand() % 100;
        }

        int rows_per_proc = n / num_procs;
        int remaining = n % num_procs;
        displs[0] = 0;
        for (int r = 0; r < num_procs; ++r) {
            sendcounts[r] = rows_per_proc + (r < remaining ? 1 : 0);
            if (r > 0) displs[r] = displs[r-1] + sendcounts[r-1];
        }
        displs[num_procs] = n;

        for (int r = 1; r < num_procs; ++r) {
            for (int i = displs[r]; i < displs[r+1]; ++i) {
                MPI_Send(A[i].data(), n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                MPI_Send(&b[i], 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            }
        }
        local_A.assign(A.begin(), A.begin() + sendcounts[0]);
        local_b.assign(b.begin(), b.begin() + sendcounts[0]);
    } else {
        MPI_Recv(sendcounts.data(), num_procs, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(displs.data(), num_procs + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int local_size = sendcounts[rank];
        local_A.resize(local_size, vector<double>(n));
        local_b.resize(local_size);
        for (int i = 0; i < local_size; ++i) {
            MPI_Recv(local_A[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&local_b[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Прямой ход
    for (int i = 0; i < n; ++i) {
        int owner_i = lower_bound(displs.begin(), displs.end(), i+1) - displs.begin() - 1;

        double local_max = 0;
        int local_max_row = -1;
        for (int j = 0; j < local_A.size(); ++j) {
            int global_row = displs[rank] + j;
            if (global_row >= i && fabs(local_A[j][i]) > local_max) {
                local_max = fabs(local_A[j][i]);
                local_max_row = global_row;
            }
        }

        struct { double val; int idx; } in{local_max, local_max_row}, out;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        int max_row = out.idx;

        if (max_row != i) {
            int owner_max = lower_bound(displs.begin(), displs.end(), max_row+1) - displs.begin() - 1;
            vector<double> row_buf(n);
            double b_buf;

            if (rank == owner_i || rank == owner_max) {
                if (rank == owner_max) {
                    int local_idx = max_row - displs[rank];
                    copy(local_A[local_idx].begin(), local_A[local_idx].end(), row_buf.begin());
                    b_buf = local_b[local_idx];
                    MPI_Send(row_buf.data(), n, MPI_DOUBLE, owner_i, 0, MPI_COMM_WORLD);
                    MPI_Send(&b_buf, 1, MPI_DOUBLE, owner_i, 0, MPI_COMM_WORLD);
                    MPI_Recv(row_buf.data(), n, MPI_DOUBLE, owner_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&b_buf, 1, MPI_DOUBLE, owner_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    copy(row_buf.begin(), row_buf.end(), local_A[local_idx].begin());
                    local_b[local_idx] = b_buf;
                }
                if (rank == owner_i) {
                    int local_idx = i - displs[rank];
                    MPI_Recv(row_buf.data(), n, MPI_DOUBLE, owner_max, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&b_buf, 1, MPI_DOUBLE, owner_max, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    vector<double> temp_row = local_A[local_idx];
                    double temp_b = local_b[local_idx];
                    MPI_Send(temp_row.data(), n, MPI_DOUBLE, owner_max, 0, MPI_COMM_WORLD);
                    MPI_Send(&temp_b, 1, MPI_DOUBLE, owner_max, 0, MPI_COMM_WORLD);
                    copy(row_buf.begin(), row_buf.end(), local_A[local_idx].begin());
                    local_b[local_idx] = b_buf;
                }
            }
        }

        if (rank == owner_i) {
            int local_idx = i - displs[rank];
            double pivot = local_A[local_idx][i];
            for (int j = i; j < n; ++j) local_A[local_idx][j] /= pivot;
            local_b[local_idx] /= pivot;
        }

        vector<double> pivot_row(n);
        double pivot_b;
        if (rank == owner_i) {
            int local_idx = i - displs[rank];
            pivot_row = local_A[local_idx];
            pivot_b = local_b[local_idx];
        }
        MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);

        for (int j = 0; j < local_A.size(); ++j) {
            int global_row = displs[rank] + j;
            if (global_row > i) {
                double factor = local_A[j][i];
                for (int k = i; k < n; ++k)
                    local_A[j][k] -= factor * pivot_row[k];
                local_b[j] -= factor * pivot_b;
            }
        }
    }

    // Сбор данных и обратный ход
    if (rank == 0) {
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);
        copy(local_A.begin(), local_A.end(), A.begin());
        copy(local_b.begin(), local_b.end(), b.begin());

        for (int r = 1; r < num_procs; ++r) {
            for (int i = displs[r]; i < displs[r+1]; ++i) {
                MPI_Recv(A[i].data(), n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&b[i], 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int i = n-1; i >= 0; --i) {
            for (int j = i-1; j >= 0; --j) {
                double factor = A[j][i];
                b[j] -= factor * b[i];
            }
        }

        cout << "Time: " << MPI_Wtime() - start_time << " sec" << endl;
    } else {
        for (auto& row : local_A)
            MPI_Send(row.data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_b.data(), local_b.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}