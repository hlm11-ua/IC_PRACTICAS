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
#include <thread>
#include <unistd.h>
#include "mpi.h"

// ==========================================
// KERNELS (Matemáticas para filtros)
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
// FUNCIONES DE CÓMPUTO (Procesan la porción local)
// ==========================================

Image<unsigned char> compute_srm(const Image<float> &local_float_img, int kernel_size) {
    Image<float> srm = local_float_img.to_grayscale(); 
    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    return srm.convert<unsigned char>();
}

Image<unsigned char> compute_dct(const Image<float> &local_float_img, int block_size, bool invert) {
    Image<float> grayscale = local_float_img.to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    for(int i=0; i<blocks.size(); i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
          for(int k=0; k<blocks[i].size/2; k++)
            for(int l=0; l<blocks[i].size/2; l++)
              dctBlock[k][l] = 0.0;
          dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
             dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }
    return grayscale.convert<unsigned char>();
}

Image<unsigned char> compute_ela(const Image<float> &local_float_img, int quality, int rank){
    Image<unsigned char> grayscale = local_float_img.to_grayscale().convert<unsigned char>();
   
    std::string temp_filename = "temp_proc_" + std::to_string(rank) + ".jpg";
    save_to_file(temp_filename.c_str(), grayscale, quality);
   
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
   
    Image<float> compressed;
    try {
        Image<unsigned char> loaded = load_from_file(temp_filename.c_str());
        if(loaded.channels == 3) {
            compressed = loaded.to_grayscale().convert<float>();
        } else {
            compressed = loaded.convert<float>();
        }

    } catch (...) {
        return grayscale;
    }
   
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
   
    remove(temp_filename.c_str());
   
    return compressed.convert<unsigned char>();
}

// ==========================================
// MAIN COMPLETO
// ==========================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start, t_part_end, t_compute_end, t_union_end;

    // --- VARIABLES GLOBALES PARA SCATTER/GATHER ---
    Image<float> full_image; 
   
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int local_data_size = 0;
    int full_width = 0, full_height = 0, full_channels = 0;

    if(argc == 1) {
        if (rank == 0) std::cerr << "Uso: ./detect <imagen>" << std::endl;
        MPI_Finalize();
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    // =======================================================
    // ETAPA 1: PARTICIÓN
    // =======================================================
    if (rank == 0) std::cout << "--- [Etapa 1] Particion y Distribucion ---" << std::endl;

    if (rank == 0) {
        try {
            full_image = load_from_file(argv[1]).convert<float>();
            full_width = full_image.width;
            full_height = full_image.height;
            full_channels = full_image.channels;

            int row_size = full_width * full_channels;
            int base_rows = full_height / size;
            int remainder = full_height % size;
            int current_disp = 0;

            for (int i = 0; i < size; ++i) {
                int rows = base_rows + (i < remainder ? 1 : 0);
                sendcounts[i] = rows * row_size; 
                displs[i] = current_disp;
                current_disp += sendcounts[i];
            }
        } catch (...) {
            std::cerr << "Rank 0: Error cargando imagen." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&full_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&full_channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    local_data_size = sendcounts[rank];
    int local_height = local_data_size / (full_width * full_channels);
    Image<float> local_image(full_width, local_height, full_channels);

    // FIX: Usar .matrix.get() en lugar de .data.data()
    MPI_Scatterv(
        rank == 0 ? full_image.matrix.get() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_FLOAT,
        local_image.matrix.get(),
        local_data_size,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    t_part_end = MPI_Wtime();

    // =======================================================
    // ETAPA 2: CÓMPUTO PARALELO
    // =======================================================
    if (rank == 0) std::cout << "--- [Etapa 2] Computo Local ---" << std::endl;

    Image<unsigned char> res_srm3 = compute_srm(local_image, 3);
    Image<unsigned char> res_srm5 = compute_srm(local_image, 5);
    Image<unsigned char> res_ela  = compute_ela(local_image, 90, rank);
    Image<unsigned char> res_dct_inv = compute_dct(local_image, 8, true);
    Image<unsigned char> res_dct_dir = compute_dct(local_image, 8, false);

    MPI_Barrier(MPI_COMM_WORLD);
    t_compute_end = MPI_Wtime();

    // =======================================================
    // ETAPA 3: UNIÓN (Gather)
    // =======================================================
    if (rank == 0) std::cout << "--- [Etapa 3] Recoleccion y Guardado ---" << std::endl;

    Image<unsigned char> final_srm3, final_srm5, final_ela, final_dct_i, final_dct_d;
   
    int result_channels = 1;
    std::vector<int> recvcounts_res(size);
    std::vector<int> displs_res(size);
   
    if (rank == 0) {
        final_srm3 = Image<unsigned char>(full_width, full_height, result_channels);
        final_srm5 = Image<unsigned char>(full_width, full_height, result_channels);
        final_ela  = Image<unsigned char>(full_width, full_height, result_channels);
        final_dct_i= Image<unsigned char>(full_width, full_height, result_channels);
        final_dct_d= Image<unsigned char>(full_width, full_height, result_channels);

        int current_d = 0;
        int rows_base = full_height / size;
        int rem = full_height % size;
        for(int i=0; i<size; i++){
            int r = rows_base + (i < rem ? 1 : 0);
            recvcounts_res[i] = r * full_width * result_channels; 
            displs_res[i] = current_d;
            current_d += recvcounts_res[i];
        }
    }

    // FIX: Usar .matrix.get() y calcular size manualmente
    // 1. Reunir SRM 3x3
    int res_size = res_srm3.width * res_srm3.height * res_srm3.channels;
    MPI_Gatherv(res_srm3.matrix.get(), res_size, MPI_UNSIGNED_CHAR,
                rank == 0 ? final_srm3.matrix.get() : nullptr, recvcounts_res.data(), displs_res.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 2. Reunir SRM 5x5
    res_size = res_srm5.width * res_srm5.height * res_srm5.channels;
    MPI_Gatherv(res_srm5.matrix.get(), res_size, MPI_UNSIGNED_CHAR,
                rank == 0 ? final_srm5.matrix.get() : nullptr, recvcounts_res.data(), displs_res.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
   
    // 3. Reunir ELA
    res_size = res_ela.width * res_ela.height * res_ela.channels;
    MPI_Gatherv(res_ela.matrix.get(), res_size, MPI_UNSIGNED_CHAR,
                rank == 0 ? final_ela.matrix.get() : nullptr, recvcounts_res.data(), displs_res.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 4. Reunir DCT Inv
    res_size = res_dct_inv.width * res_dct_inv.height * res_dct_inv.channels;
    MPI_Gatherv(res_dct_inv.matrix.get(), res_size, MPI_UNSIGNED_CHAR,
                rank == 0 ? final_dct_i.matrix.get() : nullptr, recvcounts_res.data(), displs_res.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 5. Reunir DCT Direct
    res_size = res_dct_dir.width * res_dct_dir.height * res_dct_dir.channels;
    MPI_Gatherv(res_dct_dir.matrix.get(), res_size, MPI_UNSIGNED_CHAR,
                rank == 0 ? final_dct_d.matrix.get() : nullptr, recvcounts_res.data(), displs_res.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        save_to_file("srm_kernel_3x3.png", final_srm3);
        save_to_file("srm_kernel_5x5.png", final_srm5);
        save_to_file("ela.png", final_ela);
        save_to_file("dct_invert.png", final_dct_i);
        save_to_file("dct_direct.png", final_dct_d);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_union_end = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== REPORTE MPI (%d Procesos) ===\n", size);
        printf("1. Particion (Load/Scatter) : %.2f ms\n", (t_part_end - t_start)*1000);
        printf("2. Computo Paralelo         : %.2f ms\n", (t_compute_end - t_part_end)*1000);
        printf("3. Union (Gather/Save)      : %.2f ms\n", (t_union_end - t_compute_end)*1000);
        printf("-----------------------------------\n");
        printf("TIEMPO TOTAL                : %.2f ms\n", (t_union_end - t_start)*1000);
        printf("===================================\n");
    }

    MPI_Finalize();
    return 0;
}
