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

// ==========================================
// KERNELS (Sin cambios)
// ==========================================
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

// ==========================================
// FUNCIONES DE CÓMPUTO (SECUENCIALES)
// ==========================================
// Han vuelto a ser simples porque un solo proceso hará todo el trabajo de su tarea asignada.

Image<unsigned char> compute_srm_seq(const Image<unsigned char> &image, int kernel_size) {
    Image<float> srm_input = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);
    Image<float> output = srm_input.convolution(kernel); // Usamos la conv del utils/image.h o manual
    output = output.abs().normalized() * 255;
    return output.convert<unsigned char>();
}

Image<unsigned char> compute_dct_seq(const Image<unsigned char> &image, int block_size, bool invert) {
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

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
    return grayscale.convert<unsigned char>();
}

Image<unsigned char> compute_ela_seq(const Image<unsigned char> &image, int quality) {
    // ELA requiere disco, mejor que lo haga el Master o un proceso con acceso a disco seguro
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp_ela_task.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp_ela_task.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    remove("_temp_ela_task.jpg");
    return compressed.convert<unsigned char>();
}

// ==========================================
// MAIN (PARALELISMO DE TAREAS)
// ==========================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Necesitamos al menos 5 procesos (Rank 0 + 4 trabajadores) para cubrir las 5 tareas
    // Tareas: ELA(Rank0), SRM3(Rank1), SRM5(Rank2), DCT_INV(Rank3), DCT_DIR(Rank4)
    if (procs < 5) {
        if (rank == 0) std::cerr << "ERROR: Para paralelismo de tareas funcional necesitamos al menos 5 procesos." << std::endl;
        MPI_Finalize();
        exit(1);
    }

    double t_start = MPI_Wtime();

    Image<unsigned char> image;
    int width, height, channels;

    // 1. CARGA Y DIFUSIÓN DE IMAGEN (Todos necesitan la entrada)
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

    // 2. DISTRIBUCIÓN DE TAREAS POR RANK
    // Usamos etiquetas (tags) para identificar qué imagen vuelve al maestro
    // Tag 1: SRM3, Tag 2: SRM5, Tag 3: DCT_INV, Tag 4: DCT_DIR

    if (rank == 0) {
        std::cout << "--- MASTER: Iniciando orquestación de tareas ---" << std::endl;
        
        // Tarea del Maestro: ELA (por ser I/O bound)
        double t_ela = MPI_Wtime();
        Image<unsigned char> res_ela = compute_ela_seq(image, 90);
        save_to_file("ela.png", res_ela);
        std::cout << "Master termino ELA en " << (MPI_Wtime() - t_ela)*1000 << "ms" << std::endl;

        // Recolección de resultados de los trabajadores
        // Necesitamos buffers para recibir
        long int img_size = width * height; // SRM y DCT devuelven 1 canal (gris)
        Image<unsigned char> buf_img(width, height, 1);
        
        // Recibir SRM 3x3 de Rank 1
        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        save_to_file("srm_kernel_3x3.png", buf_img);
        std::cout << "Recibido SRM 3x3 de Rank 1" << std::endl;

        // Recibir SRM 5x5 de Rank 2
        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        save_to_file("srm_kernel_5x5.png", buf_img);
        std::cout << "Recibido SRM 5x5 de Rank 2" << std::endl;

        // Recibir DCT INVERT de Rank 3
        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        save_to_file("dct_invert.png", buf_img);
        std::cout << "Recibido DCT Inv de Rank 3" << std::endl;

        // Recibir DCT DIRECT de Rank 4
        MPI_Recv(buf_img.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 4, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        save_to_file("dct_direct.png", buf_img);
        std::cout << "Recibido DCT Dir de Rank 4" << std::endl;

    } 
    else if (rank == 1) {
        // TAREA 1: SRM 3x3
        Image<unsigned char> res = compute_srm_seq(image, 3);
        MPI_Send(res.matrix.get(), res.width * res.height, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    else if (rank == 2) {
        // TAREA 2: SRM 5x5
        Image<unsigned char> res = compute_srm_seq(image, 5);
        MPI_Send(res.matrix.get(), res.width * res.height, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
    }
    else if (rank == 3) {
        // TAREA 3: DCT Inversa
        Image<unsigned char> res = compute_dct_seq(image, 8, true);
        MPI_Send(res.matrix.get(), res.width * res.height, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
    }
    else if (rank == 4) {
        // TAREA 4: DCT Directa
        Image<unsigned char> res = compute_dct_seq(image, 8, false);
        MPI_Send(res.matrix.get(), res.width * res.height, MPI_UNSIGNED_CHAR, 0, 4, MPI_COMM_WORLD);
    }
    else {
        std::cout << "Rank " << rank << " no tiene tarea asignada y se va a dormir." << std::endl;
    }

    // Esperar a que todos acaben
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Tiempo Total Ejecucion: " << (MPI_Wtime() - t_start)*1000 << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}