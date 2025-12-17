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
// FUNCIONES ADAPTADAS A SUB-GRUPOS
// ==========================================

// SRM: Versión simple que usa un comunicador específico (comm)
// Si el grupo tiene 1 proceso, actúa secuencial. Si tiene más, paraleliza.
Image<unsigned char> compute_srm_group(const Image<unsigned char> &image, int kernel_size, MPI_Comm comm) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    Image<float> srm_input = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);
    Image<float> srm_partial(srm_input.width, srm_input.height, 1);

    int rows_per_proc = srm_input.height / procs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == procs - 1) ? srm_input.height : (rank + 1) * rows_per_proc;

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
            srm_partial.set(j, i, 0, sum / (kernel.width * kernel.width));
        }
    }

    long int count = rows_per_proc * srm_input.width;
    float* send_ptr = srm_partial.matrix.get() + (start_row * srm_input.width);

    // OJO: Usamos 'comm', NO 'MPI_COMM_WORLD'
    if (rank == 0) {
        MPI_Gather(MPI_IN_PLACE, count, MPI_FLOAT, 
                   srm_partial.matrix.get(), count, MPI_FLOAT, 
                   0, comm);
    } else {
        MPI_Gather(send_ptr, count, MPI_FLOAT, 
                   NULL, count, MPI_FLOAT, 
                   0, comm);
    }

    if (rank == 0) {
        srm_partial = srm_partial.abs().normalized();
        srm_partial = srm_partial * 255;
        return srm_partial.convert<unsigned char>();
    }
    return Image<unsigned char>();
}

// DCT: Paralelizada dentro del grupo 'comm'
Image<unsigned char> compute_dct_group(const Image<unsigned char> &image, int block_size, bool invert, MPI_Comm comm) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    int total_blocks = blocks.size();
    int blocks_per_proc = total_blocks / procs;
    int start_idx = rank * blocks_per_proc;
    int end_idx = (rank == procs - 1) ? total_blocks : (rank + 1) * blocks_per_proc;

    for(int i = start_idx; i < end_idx; i++){
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

    long int floats_per_proc = (grayscale.width * grayscale.height) / procs;
    float* send_ptr = grayscale.matrix.get() + (rank * floats_per_proc);

    // OJO: Usamos 'comm', NO 'MPI_COMM_WORLD'
    if (rank == 0) {
         MPI_Gather(MPI_IN_PLACE, floats_per_proc, MPI_FLOAT,
                    grayscale.matrix.get(), floats_per_proc, MPI_FLOAT,
                    0, comm);
         return grayscale.convert<unsigned char>();
    } else {
         MPI_Gather(send_ptr, floats_per_proc, MPI_FLOAT,
                    NULL, floats_per_proc, MPI_FLOAT,
                    0, comm);
    }
    return Image<unsigned char>();
}

Image<unsigned char> compute_ela_seq(const Image<unsigned char> &image, int quality) {
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp_ela_hybrid.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp_ela_hybrid.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    remove("_temp_ela_hybrid.jpg");
    return compressed.convert<unsigned char>();
}

// ==========================================
// MAIN HÍBRIDO
// ==========================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &world_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_procs < 3) {
        if(world_rank==0) std::cerr << "Se necesitan al menos 3 procesos para este modo híbrido." << std::endl;
        MPI_Finalize(); exit(1);
    }

    // 1. CARGA DE IMAGEN (Todos la tienen)
    Image<unsigned char> image;
    int width, height, channels;
    if (world_rank == 0) {
        if(argc == 1) { std::cerr << "Falta imagen" << std::endl; exit(1); }
        image = load_from_file(argv[1]);
        width = image.width; height = image.height; channels = image.channels;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) image = Image<unsigned char>(width, height, channels);
    MPI_Bcast(image.matrix.get(), width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------
    // 2. CREACIÓN DE GRUPOS (MPI_Comm_split)
    // -------------------------------------------------------------
    // Definimos el "color" del grupo. Procesos con el mismo color estarán juntos.
    // Color 0: Maestro (Rank 0) -> Hará ELA
    // Color 1: Equipo SRM (Rank 1) -> Hará SRM
    // Color 2: Equipo DCT (Rank 2, 3...) -> Harán DCT en paralelo
    
    int color;
    if (world_rank == 0) color = 0;
    else if (world_rank == 1) color = 1;
    else color = 2; // Ranks 2, 3, etc. van al equipo DCT

    // Creamos el nuevo comunicador "local"
    MPI_Comm local_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);

    int local_rank, local_procs;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_procs);

    // -------------------------------------------------------------
    // 3. EJECUCIÓN POR GRUPOS
    // -------------------------------------------------------------
    
    // VARIABLES PARA RECOLECCIÓN FINAL
    long int img_size = width * height;
    Image<unsigned char> buffer_recv(width, height, 1);

    if (color == 0) {
        // --- GRUPO MAESTRO (Rank 0 global) ---
        std::cout << "[Rank 0] Haciendo ELA y coordinando..." << std::endl;
        
        // 1. Hace su tarea
        auto start = std::chrono::steady_clock::now();
        Image<unsigned char> ela = compute_ela_seq(image, 90);
        save_to_file("ela.png", ela);
        auto end = std::chrono::steady_clock::now();
        std::cout << "ELA completado (" << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms)" << std::endl;

        // 2. Recibe resultados de los jefes de otros grupos
        MPI_Status status;
        
        // Recibir SRM 3x3 del Jefe Grupo 1 (Rank 1 global)
        MPI_Recv(buffer_recv.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 1, 10, MPI_COMM_WORLD, &status);
        save_to_file("srm_kernel_3x3.png", buffer_recv);
        std::cout << "[Rank 0] Recibido SRM 3x3" << std::endl;

        // Recibir SRM 5x5 del Jefe Grupo 1
        MPI_Recv(buffer_recv.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 1, 11, MPI_COMM_WORLD, &status);
        save_to_file("srm_kernel_5x5.png", buffer_recv);
        std::cout << "[Rank 0] Recibido SRM 5x5" << std::endl;

        // Recibir DCT Inversa del Jefe Grupo 2 (Rank 2 global)
        MPI_Recv(buffer_recv.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 2, 20, MPI_COMM_WORLD, &status);
        save_to_file("dct_invert.png", buffer_recv);
        std::cout << "[Rank 0] Recibido DCT Inv" << std::endl;

        // Recibir DCT Directa del Jefe Grupo 2
        MPI_Recv(buffer_recv.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 2, 21, MPI_COMM_WORLD, &status);
        save_to_file("dct_direct.png", buffer_recv);
        std::cout << "[Rank 0] Recibido DCT Dir" << std::endl;

    } 
    else if (color == 1) {
        // --- GRUPO SRM (Rank 1 global) ---
        // Este grupo podría tener más procesos si aumentasemos -np. 
        // compute_srm_group usará todos los procesos de 'local_comm'.
        if(local_rank == 0) std::cout << "[Grupo SRM] Procesando con " << local_procs << " procesos..." << std::endl;

        Image<unsigned char> srm3 = compute_srm_group(image, 3, local_comm);
        // El jefe local (local_rank 0) envía al Rank 0 global
        if (local_rank == 0) {
            MPI_Send(srm3.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 10, MPI_COMM_WORLD);
        }

        Image<unsigned char> srm5 = compute_srm_group(image, 5, local_comm);
        if (local_rank == 0) {
            MPI_Send(srm5.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 11, MPI_COMM_WORLD);
        }
    } 
    else if (color == 2) {
        // --- GRUPO DCT (Rank 2, 3... globales) ---
        if(local_rank == 0) std::cout << "[Grupo DCT] Procesando con " << local_procs << " procesos..." << std::endl;

        Image<unsigned char> dct_i = compute_dct_group(image, 8, true, local_comm);
        if (local_rank == 0) {
            // Rank 2 global es el local_rank 0 de este grupo
            MPI_Send(dct_i.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 20, MPI_COMM_WORLD);
        }

        Image<unsigned char> dct_d = compute_dct_group(image, 8, false, local_comm);
        if (local_rank == 0) {
            MPI_Send(dct_d.matrix.get(), img_size, MPI_UNSIGNED_CHAR, 0, 21, MPI_COMM_WORLD);
        }
    }

    // Liberar el comunicador temporal
    MPI_Comm_free(&local_comm);

    MPI_Finalize();
    return 0;
}