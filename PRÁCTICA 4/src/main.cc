#include "mpi.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include "utils/image.h" // Asegúrate que la ruta sea correcta según tu carpeta
#include "utils/dct.h"

// --- Configuración por defecto ---
const std::string DEFAULT_FILENAME = "input.png";

// Función SRM (Spatial Rich Models) - Convolución
Image<float> get_srm_kernel_3x3() {
    Image<float> k(3, 3, 1);
    k.set(0,0,0, -1); k.set(0,1,0, 2); k.set(0,2,0, -1);
    k.set(1,0,0, 2);  k.set(1,1,0, -4); k.set(1,2,0, 2);
    k.set(2,0,0, -1); k.set(2,1,0, 2); k.set(2,2,0, -1);
    return k;
}

Image<unsigned char> compute_srm(const Image<unsigned char> &img) {
    Image<float> kernel = get_srm_kernel_3x3();
    // La convolución devuelve imagen del mismo tamaño
    return img.convolution(kernel).convert<unsigned char>();
}

// Función ELA (Secuencial, solo Rank 0)
Image<unsigned char> compute_ela(const Image<unsigned char> &img, int quality=90) {
    std::cout << "[Rank 0] Ejecutando ELA..." << std::endl;
    Image<unsigned char> gray = img.to_grayscale();
    save_to_file("_ela_temp.jpg", gray, quality);
    
    Image<float> compressed = load_from_file("_ela_temp.jpg").convert<float>();
    Image<float> orig = gray.convert<float>();
    
    // Diff + Abs + Normalize
    Image<float> diff = (compressed + (orig * -1.0)).abs();
    
    // Normalización simple x scale
    diff = diff * 10.0; // Amplificar error visualmente
    
    remove("_ela_temp.jpg");
    return diff.convert<unsigned char>();
}

int main(int argc, char **argv) {
    // 1. Inicializar MPI
    MPI_Init(&argc, &argv);
    
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Variables globales
    int width = 0, height = 0, channels = 0;
    Image<unsigned char> global_image;

    // 2. Carga de Imagen (Solo Rank 0)
    if (rank == 0) {
        std::string filename = DEFAULT_FILENAME;
        if (argc > 1) filename = argv[1]; // Opcional: permite argumento si se da

        if (!std::filesystem::exists(filename)) {
            std::cerr << "ERROR: No se encuentra el archivo '" << filename << "'." << std::endl;
            std::cerr << "Por favor, coloca una imagen llamada 'input.png' en el directorio." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        global_image = load_from_file(filename);
        width = global_image.width;
        height = global_image.height;
        channels = global_image.channels;
        
        std::cout << "Imagen cargada: " << width << "x" << height << " (" << channels << " canales)." << std::endl;

        // Comprobación de divisibilidad para Scatter simple
        if (height % procs != 0) {
            std::cerr << "ERROR FATAL: La altura (" << height << ") no es divisible por " << procs << " procesos." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Comprobación para DCT (bloques de 8) en cada trozo local
        int local_h = height / procs;
        if (local_h % 8 != 0) {
            std::cerr << "ERROR FATAL: La altura local (" << local_h << ") no es múltiplo de 8 (requerido para DCT)." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 3. Broadcast de metadatos
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 4. Preparar memoria local
    int local_height = height / procs;
    int data_size = width * local_height * channels;
    Image<unsigned char> local_image(width, local_height, channels);

    // 5. Scatter (Repartir imagen)
    unsigned char* sendptr = (rank == 0) ? global_image.matrix.get() : nullptr;
    MPI_Scatter(sendptr, data_size, MPI_UNSIGNED_CHAR, 
                local_image.matrix.get(), data_size, MPI_UNSIGNED_CHAR, 
                0, MPI_COMM_WORLD);

    // ----------------------------------------------------
    // INICIO ZONA PARALELA
    // ----------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    // A. Calcular SRM Local
    Image<unsigned char> local_srm = compute_srm(local_image);

    Image<unsigned char> local_dct = dct::compute_full_dct(local_image, 8);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    // ----------------------------------------------------

    // 6. Gather (Recolectar resultados)
    Image<unsigned char> final_srm;
    Image<unsigned char> final_dct;

    if (rank == 0) {
        final_srm = Image<unsigned char>(width, height, channels);
        final_dct = Image<unsigned char>(width, height, channels);
    }

    // Recoger SRM
    MPI_Gather(local_srm.matrix.get(), data_size, MPI_UNSIGNED_CHAR,
               (rank==0 ? final_srm.matrix.get() : nullptr), data_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // Recoger DCT
    MPI_Gather(local_dct.matrix.get(), data_size, MPI_UNSIGNED_CHAR,
               (rank==0 ? final_dct.matrix.get() : nullptr), data_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // 7. Guardar y ELA (Solo Rank 0)
    if (rank == 0) {
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "Tiempo de Computo Paralelo (SRM + DCT): " << elapsed << " ms" << std::endl;

        save_to_file("output_srm.png", final_srm);
        std::cout << "Guardado: output_srm.png" << std::endl;

        save_to_file("output_dct.png", final_dct);
        std::cout << "Guardado: output_dct.png" << std::endl;

        // ELA (No paralelizable eficientemente por IO, se hace secuencial aquí)
        auto t_ela_start = std::chrono::high_resolution_clock::now();
        Image<unsigned char> ela = compute_ela(global_image);
        auto t_ela_end = std::chrono::high_resolution_clock::now();
        
        save_to_file("output_ela.png", ela);
        std::cout << "Guardado: output_ela.png (" 
                  << std::chrono::duration<double, std::milli>(t_ela_end - t_ela_start).count() 
                  << " ms)" << std::endl;
    }

    MPI_Finalize();
    return 0;
}