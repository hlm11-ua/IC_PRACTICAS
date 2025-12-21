#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include "mpi.h"


Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3: return get_srm_3x3();
        case 5: return get_srm_5x5();
    }
    return get_srm_3x3();
}

Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size, int rank, int procs) {
    if (rank == 0) std::cout << "Computing SRM " << kernel_size << "x" << kernel_size << "..." << std::endl;
    auto begin = std::chrono::steady_clock::now();

    auto t1_start = std::chrono::steady_clock::now();
    Image<float> srm_input = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);
    
    Image<float> srm_output(srm_input.width, srm_input.height, 1);

    int rows_per_proc = srm_input.height / procs;
    
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    auto t1_end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "  -> Preproc (Gray/Setup): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count() << "ms" << std::endl;

    auto t2_start = std::chrono::steady_clock::now();
    int k_center = kernel.width / 2;
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < srm_input.width; i++) {
            float sum = 0.0;
            for(int u = 0; u < kernel.width; u++) {
                for(int v = 0; v < kernel.width; v++) {
                    int s = (j + u - k_center + srm_input.height) % srm_input.height;
                    int t = (i + v - k_center + srm_input.width) % srm_input.width;
                    sum += srm_input.get(s, t, 0) * kernel.get(u, v, 0);
                }
            }
            srm_output.set(j, i, 0, sum / (kernel.width * kernel.width));
        }
    }
    auto t2_end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "  -> Convolution (Local): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count() << "ms" << std::endl;

    auto t3_start = std::chrono::steady_clock::now();
    long int count = rows_per_proc * srm_input.width;
    float* my_data_ptr = srm_output.matrix.get() + (start_row * srm_input.width);

    MPI_Gather(my_data_ptr, count, MPI_FLOAT, 
               rank == 0 ? srm_output.matrix.get() : NULL, count, MPI_FLOAT, 
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int processed_rows = rows_per_proc * procs;
        
        for (int j = processed_rows; j < srm_input.height; j++) {
            for (int i = 0; i < srm_input.width; i++) {
                float sum = 0.0;
                for(int u = 0; u < kernel.width; u++) {
                    for(int v = 0; v < kernel.width; v++) {
                        int s = (j + u - k_center + srm_input.height) % srm_input.height;
                        int t = (i + v - k_center + srm_input.width) % srm_input.width;
                        sum += srm_input.get(s, t, 0) * kernel.get(u, v, 0);
                    }
                }
                srm_output.set(j, i, 0, sum / (kernel.width * kernel.width));
            }
        }

        srm_output = srm_output.abs().normalized();
        srm_output = srm_output * 255;
        
        auto t3_end = std::chrono::steady_clock::now();
        std::cout << "  -> Postproc (Gather/Remain/Norm): " << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count() << "ms" << std::endl;
        
        return srm_output.convert<unsigned char>();
    }
    
    return Image<unsigned char>();
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert, int rank, int procs) {
    if (rank == 0) {
        std::cout << "Computing";
        if (invert) std::cout << " inverse"; else std::cout << " direct";
        std::cout << " DCT " << block_size << "x" << block_size << "..." << std::endl;
    }
    
    auto t1_start = std::chrono::steady_clock::now();
    int width, height;
    Image<unsigned char> gray_source;

    if (rank == 0) {
        width = image.width;
        height = image.height;
        gray_source = image.to_grayscale();
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = (height / procs) / block_size * block_size;
    
    if (rows_per_proc == 0) rows_per_proc = block_size; 

    long int send_count = rows_per_proc * width;
    
    unsigned char* my_buffer = new unsigned char[send_count];

    MPI_Scatter(rank == 0 ? gray_source.matrix.get() : NULL, send_count, MPI_UNSIGNED_CHAR,
                my_buffer, send_count, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    auto t1_end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "  -> Preproc (Setup/Scatter): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count() << "ms" << std::endl;

    auto t2_start = std::chrono::steady_clock::now();
    Image<unsigned char> my_image_part(width, rows_per_proc, 1);
    for(int i=0; i<send_count; i++) my_image_part.matrix[i] = my_buffer[i];

    Image<float> grayscale_part = my_image_part.convert<float>();
    std::vector<Block<float>> blocks = grayscale_part.get_blocks(block_size);
    for(int i=0; i<blocks.size(); i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
            for(int k=0; k<blocks[i].size/2; k++)
                for(int l=0; l<blocks[i].size/2; l++) dctBlock[k][l] = 0.0;
            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }
    Image<unsigned char> my_result_part = grayscale_part.convert<unsigned char>();
    auto t2_end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "  -> Calculation Loop (Local): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count() << "ms" << std::endl;

    auto t3_start = std::chrono::steady_clock::now();
    Image<unsigned char> final_result;
    if (rank == 0) final_result = Image<unsigned char>(width, height, 1);

    MPI_Gather(my_result_part.matrix.get(), send_count, MPI_UNSIGNED_CHAR,
               rank == 0 ? final_result.matrix.get() : NULL, send_count, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int processed_rows = rows_per_proc * procs;
        int remaining_rows = height - processed_rows;

        if (remaining_rows > 0) {

            Image<unsigned char> bottom_strip(width, remaining_rows, 1);
            
            long int offset = processed_rows * width;
            for (int i=0; i < remaining_rows * width; i++) {
                bottom_strip.matrix[i] = gray_source.matrix[offset + i];
            }

            Image<float> strip_float = bottom_strip.convert<float>();
            std::vector<Block<float>> strip_blocks = strip_float.get_blocks(block_size); 
            
            for(int i=0; i<strip_blocks.size(); i++){
                float **dctBlock = dct::create_matrix(block_size, block_size);
                dct::direct(dctBlock, strip_blocks[i], 0);
                if (invert) {
                    for(int k=0; k<strip_blocks[i].size/2; k++)
                        for(int l=0; l<strip_blocks[i].size/2; l++) dctBlock[k][l] = 0.0;
                    dct::inverse(strip_blocks[i], dctBlock, 0, 0.0, 255.);
                } else {
                    dct::assign(dctBlock, strip_blocks[i], 0);
                }
                dct::delete_matrix(dctBlock);
            }
            Image<unsigned char> strip_result = strip_float.convert<unsigned char>();

            for (int i=0; i < remaining_rows * width; i++) {
                final_result.matrix[offset + i] = strip_result.matrix[i];
            }
        }
        
        auto t3_end = std::chrono::steady_clock::now();
        std::cout << "  -> Postproc (Gather/Remainder): " << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count() << "ms" << std::endl;

        delete[] my_buffer;
        return final_result;
    }

    delete[] my_buffer;
    return Image<unsigned char>();
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality, int rank){
    if (rank == 0) {
        std::cout << "Computing ELA (Sequential)..." << std::endl;
        auto begin = std::chrono::steady_clock::now();
        
        Image<unsigned char> grayscale = image.to_grayscale();
        save_to_file("_temp_ela.jpg", grayscale, quality);
        
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  -> Preproc (Gray/Save): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - begin).count() << "ms" << std::endl;

        Image<float> compressed = load_from_file("_temp_ela.jpg").convert<float>();
        compressed = compressed + (grayscale.convert<float>()*(-1));
        
        auto t2 = std::chrono::steady_clock::now();
        std::cout << "  -> Calculation (Load/Diff): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

        compressed = compressed.abs().normalized() * 255;
        remove("_temp_ela.jpg");
        Image<unsigned char> res = compressed.convert<unsigned char>();
        
        auto end = std::chrono::steady_clock::now();
        std::cout << "  -> Postproc (Norm/Convert): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - t2).count() << "ms" << std::endl;

        return res;
    }
    return Image<unsigned char>();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t_start, t_dist_end, t_compute_start, t_compute_end, t_save_end;
    double t_srm3, t_srm5, t_ela, t_dct_inv, t_dct_dir;

    if(argc == 1) {
        if (rank == 0) std::cerr << "Uso: mpirun -np N ./programa <imagen>" << std::endl;
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (rank == 0) std::cout << "--- Iniciando MPI con " << procs << " procesos ---" << std::endl;

    Image<unsigned char> image;
    int width, height, channels;

    if (rank == 0) {
        image = load_from_file(argv[1]);
        width = image.width;
        height = image.height;
        channels = image.channels;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        image = Image<unsigned char>(width, height, channels);
    }
    MPI_Bcast(image.matrix.get(), width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_dist_end = MPI_Wtime();

    Image<unsigned char> r_srm3, r_srm5, r_ela, r_dct_i, r_dct_d;
    int block_size = 8;

    t_compute_start = MPI_Wtime();
    r_srm3 = compute_srm(image, 3, rank, procs);
    t_srm3 = MPI_Wtime() - t_compute_start;

    t_compute_start = MPI_Wtime();
    r_srm5 = compute_srm(image, 5, rank, procs);
    t_srm5 = MPI_Wtime() - t_compute_start;

    t_compute_start = MPI_Wtime();
    r_ela = compute_ela(image, 90, rank);
    t_ela = MPI_Wtime() - t_compute_start;

    t_compute_start = MPI_Wtime();
    r_dct_i = compute_dct(image, block_size, true, rank, procs);
    t_dct_inv = MPI_Wtime() - t_compute_start;

    t_compute_start = MPI_Wtime();
    r_dct_d = compute_dct(image, block_size, false, rank, procs);
    t_dct_dir = MPI_Wtime() - t_compute_start;

    MPI_Barrier(MPI_COMM_WORLD);
    t_compute_end = MPI_Wtime();

    if (rank == 0) {
        save_to_file("srm_kernel_3x3.png", r_srm3);
        save_to_file("srm_kernel_5x5.png", r_srm5);
        save_to_file("ela.png", r_ela);
        save_to_file("dct_invert.png", r_dct_i);
        save_to_file("dct_direct.png", r_dct_d);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    t_save_end = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== RESULTADOS MPI (%d PROCESOS) ===\n", procs);
        printf("Distribucion datos (Bcast) : %7.2f ms\n", (t_dist_end - t_start) * 1000.0);
        printf("SRM 3x3 (Total)            : %7.2f ms\n", t_srm3 * 1000.0);
        printf("SRM 5x5 (Total)            : %7.2f ms\n", t_srm5 * 1000.0);
        printf("ELA (Secuencial)           : %7.2f ms\n", t_ela * 1000.0);
        printf("DCT Inversa (Total)        : %7.2f ms\n", t_dct_inv * 1000.0);
        printf("DCT Directa (Total)        : %7.2f ms\n", t_dct_dir * 1000.0);
        printf("Guardado disco             : %7.2f ms\n", (t_save_end - t_compute_end) * 1000.0);
        printf("TIEMPO TOTAL GLOBAL        : %7.2f ms\n", (t_save_end - t_start) * 1000.0);
        printf("======================================\n");
    }

    MPI_Finalize();
    return 0;
}
