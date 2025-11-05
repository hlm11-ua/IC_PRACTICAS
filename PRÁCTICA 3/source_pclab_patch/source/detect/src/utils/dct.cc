#include "dct.h"
#include "image.h"
#include <math.h>

void dct::direct(float **dct, const Block<float> &matrix, int channel)
{
    int m = matrix.size;
    int n = m;

    float ci, cj, dct1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0)
                ci = 1 / sqrt(m);
            else
                ci = sqrt(2) / sqrt(m);
            if (j == 0)
                cj = 1 / sqrt(n);
            else
                cj = sqrt(2) / sqrt(n);

            float sum = 0;
            for (int k = 0; k < m; k++) {
                for (int l = 0; l < n; l++) {
                    sum += matrix.get_pixel(k, l, channel) * 
                           cos((2 * k + 1) * i * M_PI / (2 * m)) * 
                           cos((2 * l + 1) * j * M_PI / (2 * n));
                }
            }
            dct[i][j] = ci * cj * sum;
        }
    }
}
void dct::inverse(Block<float> &idctMatrix, float **dctMatrix, int channel, float min, float max) {
    int size = idctMatrix.size;
    int num_tasks = std::thread::hardware_concurrency();
    if (num_tasks == 0) num_tasks = 4;

    const int rows_per_task = (size + num_tasks - 1) / num_tasks;
    std::vector<std::future<void>> tasks;

    for (int t = 0; t < num_tasks; ++t) {
        int start_row = t * rows_per_task;
        int end_row = std::min(start_row + rows_per_task, size);

        tasks.push_back(std::async(std::launch::async, [&, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < size; ++j) {
                    float sum = 0.0f;
                    for (int u = 0; u < size; ++u) {
                        for (int v = 0; v < size; ++v) {
                            float Cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
                            float Cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;

                            sum += dctMatrix[u][v] *
                                   std::cos((2 * i + 1) * u * M_PI / (2.0f * size)) *
                                   std::cos((2 * j + 1) * v * M_PI / (2.0f * size));
                        }
                    }

                    idctMatrix.set_pixel(i, j, channel, 0.25f * sum);
                }
            }
        }));
    }

    // Esperar a que todas las tareas terminen
    for (auto &task : tasks) {
        task.get();
    }
}
 
void dct::normalize(float **DCTMatrix, int size){
    float max_v=-99999999.0, min_v=999999999.0;
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            if (DCTMatrix[i][j] < min_v) min_v=DCTMatrix[i][j];
            if (DCTMatrix[i][j] > max_v) max_v=DCTMatrix[i][j];
        }
    }
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            DCTMatrix[i][j] = 255.0 * (DCTMatrix[i][j] -min_v)/ (max_v - min_v);
        }
    }
}

void dct::assign(float **DCTMatrix, Block<float> &block, int channel) {
    int size = block.size;
    int num_tasks = std::thread::hardware_concurrency();
    if (num_tasks == 0) num_tasks = 4;

    const int rows_per_task = (size + num_tasks - 1) / num_tasks;
    std::vector<std::future<void>> tasks;

    for (int t = 0; t < num_tasks; ++t) {
        int start_row = t * rows_per_task;
        int end_row = std::min(start_row + rows_per_task, size);

        tasks.push_back(std::async(std::launch::async, [&, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < size; ++j) {
                    block.set_pixel(i, j, channel, DCTMatrix[i][j]);
                }
            }
        }));
    }

    for (auto &task : tasks) {
        task.get();
    }
}

float **dct::create_matrix(int x_size, int y_size){
    float **m = new float*[x_size]; //(float**)calloc(dimX, sizeof(float*));
    float *p = new float[x_size*y_size];//(float*)calloc(dimX*dimY, sizeof(float));
    for(int i=0; i<x_size;i++){
        m[i] = &p[i*y_size];
    }
    return m;
}

void dct::delete_matrix(float **m){
    delete [] m[0];
    delete [] m;
}
