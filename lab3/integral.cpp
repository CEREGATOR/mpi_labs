// lab1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <math.h>
#include <ctime>
#include <omp.h>

#include <chrono>
using namespace std;
using namespace std::chrono;

#define NUM 5

double refer = 2.094395;

double integral(int q, double h)
{
    double sum = 0;

    // int nt = 6;
    // omp_set_num_threads(nt);

    // double *sums = new double[nt];

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    #pragma omp parallel for reduction (+:sum) num_threads(6)
    // #pragma omp parallel for 
    for (int i = 0; i < q; i++)
    {
        sum = sum + (4 / sqrt(4 - pow((h * i + h / 2), 2))) * h;
        // sums[omp_get_thread_num()] =(4 / sqrt(4 - (h * i + h / 2) * (h * i + h / 2))) * h;
    }
    // #pragma omp simd simdlen(6)   // single instruction + multiple data
    // for (int i = 0; i < nt; i++)
    //     sum += sums[i];
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> duration = (t2 - t1);
    cout << "Duration is: " << duration.count() << " seconds" << endl;
    return sum;
}

int main()
{
    double a = 0;
    double b = 1;
    double res;
    int q[NUM] = {10000, 100000, 1000000, 10000000, 100000000};
    double h[NUM];
    for (int i = 0; i < NUM; i++)
    {
        h[i] = (b - a) / (double)(q[i]);
        printf("q = %i \n", q[i]);
        printf("h = %f \n", h[i]);
        res = integral(q[i], h[i]);
        printf("res = %f \n", res);
        printf("err = %e \n\n\n", abs(res - refer));
    }
}
