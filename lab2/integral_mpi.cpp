#include <iostream>
#include <cmath>
#include <mpi.h>

using namespace std;

#define NUM 5

double refer = 2.094395;

double integral(int q, double h, int rank, int size)
{
    double local_sum = 0.0;
    double global_sum = 0.0;
    int start = (q / size) * rank;
    int end = (q / size) * (rank + 1);
    if (rank == size - 1)
        end = q;

    for (int i = start; i < end; i++)
    {
        local_sum += (4 / sqrt(4 - pow((h * i + h / 2), 2))) * h;
    }
    
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 0, b = 1;
    int q[NUM] = {10000, 100000, 1000000, 10000000, 100000000};
    double h[NUM];
    
    if (rank == 0) printf("Running MPI version\n");
    
    for (int i = 0; i < NUM; i++)
    {
        h[i] = (b - a) / (double)(q[i]);
        if (rank == 0)
        {
            printf("q = %i \n", q[i]);
            printf("h = %f \n", h[i]);
        }

        double start_time = MPI_Wtime();
        double res = integral(q[i], h[i], rank, size);
        double end_time = MPI_Wtime();
        
        if (rank == 0)
        {
            printf("res = %f \n", res);
            printf("err = %e \n", abs(res - refer));
            printf("Duration: %f seconds\n\n", end_time - start_time);
        }
    }
    
    MPI_Finalize();
    return 0;
}
