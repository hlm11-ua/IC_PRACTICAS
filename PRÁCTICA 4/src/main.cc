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
    if (size == 5) return get_srm_5x5();
    return get_srm_3x3();
}



Image<unsigned char> compute_srm_seq(const Image<unsigned char> &image, int kernel_size, int rank) {
    std::cout << "[Rank " << rank << "] Iniciando SRM " << kernel_size << "x" << kernel_size << "..." << std::endl;
    auto begin = std::chrono::steady_clock::now();

    auto t1_start = std::chrono::steady_clock::now();
    Image<float> srm_input = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);
    auto t1_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Preproc (Gray/Conv): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count() << "ms" << std::endl;

    auto t2_start = std::chrono::steady_clock::now();
    Image<float> output = srm_input.convolution(kernel); 
    auto t2_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Convolucion: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count() << "ms" << std::endl;

    auto t3_start = std::chrono::steady_clock::now();
    output = output.abs().normalized() * 255;
    Image<unsigned char> result = output.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Postproc (Norm): " << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count() << "ms" << std::endl;

    auto end = std::chrono::steady_clock::now();
    std::cout << "[Rank " << rank << "] Total SRM " << kernel_size << "x" << kernel_size << ": " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return result;
}

Image<unsigned char> compute_dct_seq(const Image<unsigned char> &image, int block_size, bool invert, int rank) {
    std::cout << "[Rank " << rank << "] Iniciando DCT " << (invert ? "Inversa" : "Directa") << "..." << std::endl;
    auto begin = std::chrono::steady_clock::now();

    auto t1_start = std::chrono::steady_clock::now();
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    auto t1_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Preproc (Blocks): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count() << "ms" << std::endl;

    auto t2_start = std::chrono::steady_clock::now();
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
    auto t2_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Calculo DCT Loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count() << "ms" << std::endl;
    
    auto t3_start = std::chrono::steady_clock::now();
    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank " << rank << "] Postproc (Convert): " << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count() << "ms" << std::endl;

    auto end = std::chrono::steady_clock::now();
    std::cout << "[Rank " << rank << "] Total DCT " << (invert ? "Inv" : "Dir") << ": " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    
    return result;
}

Image<unsigned char> compute_ela_seq(const Image<unsigned char> &image, int quality) {
    std::cout << "[Rank 0] Iniciando ELA..." << std::endl;
    auto begin = std::chrono::steady_clock::now();
    
    auto t1_start = std::chrono::steady_clock::now();
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp_ela_task.jpg", grayscale, quality);
    auto t1_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank 0] Preproc (Save JPEG): " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count() << "ms" << std::endl;

    auto t2_start = std::chrono::steady_clock::now();
    Image<float> compressed = load_from_file("_temp_ela_task.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    auto t2_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank 0] Calc (Load/Diff): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count() << "ms" << std::endl;

    auto t3_start = std::chrono::steady_clock::now();
    compressed = compressed.abs().normalized() * 255;
    remove("_temp_ela_task.jpg");
    Image<unsigned char> result = compressed.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout << "  -> [Rank 0] Postproc (Norm): " << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count() << "ms" << std::endl;
    
    auto end = std::chrono::steady_clock::now();
    std::cout << "[Rank 0] Total ELA: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (procs < 4) {
        if (rank == 0) std::cerr << "ERROR: Se necesitan al menos 4 procesos." << std::endl;
        MPI_Finalize();
        exit(1);
    }

    double t_start_global = MPI_Wtime();
    double t_bcast_end, t_ela_end, t_srm3_rx, t_srm5_rx, t_dcti_rx, t_dctd_rx;
    double t_save_start, t_save_end;

    Image<unsigned char> image;
    int width, height, channels;

    if (rank == 0) {
        if(argc == 1) { std::cerr << "Falta imagen" << std::endl; exit(1); }
        image = load_from_file(argv[1]);
        width = image.width; height = image.height; channels = image.channels;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) image = Image<unsigned char>(width, height, channels);
    MPI_Bcast(image.matrix.get(), width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizar para medir el Bcast limpio
    t_bcast_end = MPI_Wtime();

    long int img_size = width * height;

    if (rank == 0) {
        std::cout << "--- MASTER: Iniciando con 4 procesos ---" << std::endl;
        
        Image<unsigned char> res_ela = compute_ela_seq(image, 90);
        t_ela_end = MPI_Wtime();
        
        t_save_start = MPI_Wtime();
        save_to_file("ela.png", res_ela);
        t_save_end = MPI_Wtime(); 

        Image<unsigned char> buf_img(width, height, 1);
        MPI_Status status;

        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 1, 10, MPI_COMM_WORLD, &status);
        t_srm3_rx = MPI_Wtime(); 
        save_to_file("srm_kernel_3x3.png", buf_img);
        std::cout << "[Master] Recibido SRM 3x3" << std::endl;

        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 1, 11, MPI_COMM_WORLD, &status);
        t_srm5_rx = MPI_Wtime();
        save_to_file("srm_kernel_5x5.png", buf_img);
        std::cout << "[Master] Recibido SRM 5x5" << std::endl;

        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 2, 20, MPI_COMM_WORLD, &status);
        t_dcti_rx = MPI_Wtime();
        save_to_file("dct_invert.png", buf_img);
        std::cout << "[Master] Recibido DCT Inv" << std::endl;

        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 3, 21, MPI_COMM_WORLD, &status);
        t_dctd_rx = MPI_Wtime();
        save_to_file("dct_direct.png", buf_img);
        std::cout << "[Master] Recibido DCT Dir" << std::endl;

    } 
    else if (rank == 1) {
        
        Image<unsigned char> res3 = compute_srm_seq(image, 3, rank);
        MPI_Send(res3.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 10, MPI_COMM_WORLD);

        Image<unsigned char> res5 = compute_srm_seq(image, 5, rank);
        MPI_Send(res5.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 11, MPI_COMM_WORLD);
    }
    else if (rank == 2) {
        Image<unsigned char> res = compute_dct_seq(image, 8, true, rank);
        MPI_Send(res.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 20, MPI_COMM_WORLD);
    }
    else if (rank == 3) {
        Image<unsigned char> res = compute_dct_seq(image, 8, false, rank);
        MPI_Send(res.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 21, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        double t_total = MPI_Wtime() - t_start_global;
        
        printf("\n=== RESUMEN TIEMPOS MPI (TASK PARALLELISM: 4 PROCS) ===\n");
        printf("Difusion inicial (Bcast)     : %7.2f ms\n", (t_bcast_end - t_start_global) * 1000.0);
        printf("---------------------------------------------\n");
        printf("Tarea ELA (Rank 0 Local)     : %7.2f ms\n", (t_ela_end - t_bcast_end) * 1000.0);
        printf("Tarea SRM 3x3 (Rx Rank 1)    : %7.2f ms (Espera desde fin ELA)\n", (t_srm3_rx - t_ela_end) * 1000.0);
        printf("Tarea SRM 5x5 (Rx Rank 1)    : %7.2f ms (Espera desde SRM3)\n", (t_srm5_rx - t_srm3_rx) * 1000.0);
        printf("Tarea DCT Inv (Rx Rank 2)    : %7.2f ms (Espera desde SRM5)\n", (t_dcti_rx - t_srm5_rx) * 1000.0);
        printf("Tarea DCT Dir (Rx Rank 3)    : %7.2f ms (Espera desde DCTI)\n", (t_dctd_rx - t_dcti_rx) * 1000.0);
        printf("---------------------------------------------\n");
        printf("TIEMPO TOTAL EJECUCION       : %7.2f ms\n", t_total * 1000.0);
        printf("=============================================\n");
    }

    MPI_Finalize();
    return 0;
}
