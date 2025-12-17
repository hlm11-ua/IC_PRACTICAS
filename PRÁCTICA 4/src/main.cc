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
// KERNELS Y MATRICES (SRM) - Sin cambios
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
    assert(size == 3 || size == 5);
    switch(size){
        case 3: return get_srm_3x3();
        case 5: return get_srm_5x5();
    }
    return get_srm_3x3();
}

// ==========================================
// FUNCIONES DE CÓMPUTO PARALELIZADAS
// ==========================================

// --- SRM: Usamos Bcast para entrada (fácil halos) y Gather para salida ---
Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size, int rank, int procs) {
    Image<float> srm_input = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);
    
    // Buffer donde escribiremos el resultado (todos reservan memoria)
    // Nota: Aunque solo usaremos un trozo, reservar todo simplifica los índices.
    Image<float> srm_output(srm_input.width, srm_input.height, 1);

    // 1. Calcular división EXACTA por filas
    // Si height=102 y procs=4 -> rows_per_proc = 25. (Sobran 2 filas al final)
    int rows_per_proc = srm_input.height / procs;
    
    // Mis filas asignadas
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc; // EXACTO, sin coger sobrantes

    // 2. Cómputo Local (Parte Paralela)
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

    // 3. Recolección (Gather Estándar)
    // Enviamos exactamente 'rows_per_proc' filas.
    long int count = rows_per_proc * srm_input.width;
    float* my_data_ptr = srm_output.matrix.get() + (start_row * srm_input.width);

    // MPI_Gather estándar (sin V)
    // El root recibe en srm_output.matrix.get().
    // Como Gather escribe secuencialmente, llenará desde la fila 0 hasta (rows_per_proc * procs).
    MPI_Gather(my_data_ptr, count, MPI_FLOAT, 
               rank == 0 ? srm_output.matrix.get() : NULL, count, MPI_FLOAT, 
               0, MPI_COMM_WORLD);

    // 4. Procesar el RESIDUO (Solo el Maestro)
    // Si sobraron filas al final (ej. filas 100 y 101), el maestro las hace ahora a mano.
    if (rank == 0) {
        int processed_rows = rows_per_proc * procs;
        
        // Bucle para lo que falta
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

        // Post-proceso final
        srm_output = srm_output.abs().normalized();
        srm_output = srm_output * 255;
        return srm_output.convert<unsigned char>();
    }
    
    return Image<unsigned char>();
}

// --- DCT: Usamos Scatter para entrada y Gather para salida (ahorro memoria) ---
Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert, int rank, int procs) {
    int width, height;
    Image<unsigned char> gray_source;

    // 1. Preparación en Maestro
    if (rank == 0) {
        width = image.width;
        height = image.height;
        gray_source = image.to_grayscale();
    }

    // 2. Compartir dimensiones
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 3. Calcular división EXACTA y ALINEADA A 8
    // Calculamos cuántas filas caben por proceso que sean múltiplo de 8
    int rows_per_proc = (height / procs) / block_size * block_size;
    
    // Protección: Si la imagen es muy pequeña, forzamos al menos 8 filas (aunque MPI fallaría si procs es grande)
    if (rows_per_proc == 0) rows_per_proc = block_size; 

    long int send_count = rows_per_proc * width;
    
    // Buffer local para recibir MI parte
    unsigned char* my_buffer = new unsigned char[send_count];

    // 4. Repartir (Scatter Estándar)
    // El maestro envía desde el inicio. Lo que sobre al final se ignora por ahora.
    MPI_Scatter(rank == 0 ? gray_source.matrix.get() : NULL, send_count, MPI_UNSIGNED_CHAR,
                my_buffer, send_count, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // 5. Cómputo Local
    // Reconstruimos mini-imagen local
    Image<unsigned char> my_image_part(width, rows_per_proc, 1);
    for(int i=0; i<send_count; i++) my_image_part.matrix[i] = my_buffer[i];

    // Lógica DCT Local (misma lógica de siempre)
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

    // 6. Recolección (Gather Estándar)
    Image<unsigned char> final_result;
    if (rank == 0) final_result = Image<unsigned char>(width, height, 1);

    MPI_Gather(my_result_part.matrix.get(), send_count, MPI_UNSIGNED_CHAR,
               rank == 0 ? final_result.matrix.get() : NULL, send_count, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // 7. Procesar el RESIDUO (Solo el Maestro)
    if (rank == 0) {
        int processed_rows = rows_per_proc * procs;
        int remaining_rows = height - processed_rows;

        // Si sobran filas, el maestro las procesa secuencialmente aquí
        if (remaining_rows > 0) {
            // Creamos una mini imagen temporal con la franja final
            Image<unsigned char> bottom_strip(width, remaining_rows, 1);
            
            // Copiamos los datos del final de la imagen original
            long int offset = processed_rows * width;
            for (int i=0; i < remaining_rows * width; i++) {
                bottom_strip.matrix[i] = gray_source.matrix[offset + i];
            }

            // Aplicamos DCT a la franja final
            Image<float> strip_float = bottom_strip.convert<float>();
            // OJO: get_blocks rellena con negro si no es múltiplo de 8, lo cual es correcto
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

            // Pegamos el resultado en la imagen final
            for (int i=0; i < remaining_rows * width; i++) {
                final_result.matrix[offset + i] = strip_result.matrix[i];
            }
        }
        
        delete[] my_buffer;
        return final_result;
    }

    delete[] my_buffer;
    return Image<unsigned char>();
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality, int rank){
    if (rank == 0) {
        Image<unsigned char> grayscale = image.to_grayscale();
        save_to_file("_temp_ela.jpg", grayscale, quality);
        Image<float> compressed = load_from_file("_temp_ela.jpg").convert<float>();
        compressed = compressed + (grayscale.convert<float>()*(-1));
        compressed = compressed.abs().normalized() * 255;
        remove("_temp_ela.jpg");
        return compressed.convert<unsigned char>();
    }
    return Image<unsigned char>();
}

// ==========================================
// MAIN
// ==========================================

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

    // 1. Maestro carga metadatos y datos
    if (rank == 0) {
        image = load_from_file(argv[1]);
        width = image.width;
        height = image.height;
        channels = image.channels;
    }

    // 2. Comunicar dimensiones y datos (Broadcast)
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        image = Image<unsigned char>(width, height, channels);
    }
    // Broadcast de la imagen completa para SRM
    MPI_Bcast(image.matrix.get(), width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_dist_end = MPI_Wtime();

    // ---------------------------------------------------------
    // ETAPA 2: CÓMPUTO PARALELO
    // ---------------------------------------------------------
    
    Image<unsigned char> r_srm3, r_srm5, r_ela, r_dct_i, r_dct_d;
    int block_size = 8;

    // >> SRM 3x3
    t_compute_start = MPI_Wtime();
    r_srm3 = compute_srm(image, 3, rank, procs);
    t_srm3 = MPI_Wtime() - t_compute_start;

    // >> SRM 5x5
    t_compute_start = MPI_Wtime();
    r_srm5 = compute_srm(image, 5, rank, procs);
    t_srm5 = MPI_Wtime() - t_compute_start;

    // >> ELA
    t_compute_start = MPI_Wtime();
    r_ela = compute_ela(image, 90, rank);
    t_ela = MPI_Wtime() - t_compute_start;

    // >> DCT Inversa
    t_compute_start = MPI_Wtime();
    r_dct_i = compute_dct(image, block_size, true, rank, procs);
    t_dct_inv = MPI_Wtime() - t_compute_start;

    // >> DCT Directa
    t_compute_start = MPI_Wtime();
    r_dct_d = compute_dct(image, block_size, false, rank, procs);
    t_dct_dir = MPI_Wtime() - t_compute_start;

    MPI_Barrier(MPI_COMM_WORLD);
    t_compute_end = MPI_Wtime();

    // ---------------------------------------------------------
    // ETAPA 3: GUARDADO (Solo Maestro)
    // ---------------------------------------------------------
    if (rank == 0) {
        save_to_file("srm_kernel_3x3.png", r_srm3);
        save_to_file("srm_kernel_5x5.png", r_srm5);
        save_to_file("ela.png", r_ela);
        save_to_file("dct_invert.png", r_dct_i);
        save_to_file("dct_direct.png", r_dct_d);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    t_save_end = MPI_Wtime();

    // REPORTE
    if (rank == 0) {
        printf("\n=== RESULTADOS MPI (%d PROCESOS) ===\n", procs);
        printf("Distribucion datos (Bcast) : %7.2f ms\n", (t_dist_end - t_start) * 1000.0);
        printf("SRM 3x3 (Paralelo)         : %7.2f ms\n", t_srm3 * 1000.0);
        printf("SRM 5x5 (Paralelo)         : %7.2f ms\n", t_srm5 * 1000.0);
        printf("ELA (Secuencial)           : %7.2f ms\n", t_ela * 1000.0);
        printf("DCT Inversa (Paralelo)     : %7.2f ms\n", t_dct_inv * 1000.0);
        printf("DCT Directa (Paralelo)     : %7.2f ms\n", t_dct_dir * 1000.0);
        printf("Guardado disco             : %7.2f ms\n", (t_save_end - t_compute_end) * 1000.0);
        printf("TIEMPO TOTAL               : %7.2f ms\n", (t_save_end - t_start) * 1000.0);
        printf("======================================\n");
    }

    MPI_Finalize();
    return 0;
}