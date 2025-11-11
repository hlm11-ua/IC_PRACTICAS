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
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}


Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"Computing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;

    // --- Medición 1: Preprocesamiento (Grises, Convertir a Float) ---
    auto t1_start = std::chrono::steady_clock::now();
    Image<float> srm = image.to_grayscale().convert<float>();
    auto t1_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Preproc (Grayscale/Convert): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count()<<"ms"<<std::endl;

    // --- Medición 2: Convolución (SRM) ---
    auto t2_start = std::chrono::steady_clock::now();
    srm = srm.convolution(get_srm_kernel(kernel_size));
    auto t2_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Convolution: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count()<<"ms"<<std::endl;
    
    // --- Medición 3: Postprocesamiento (Abs, Normalize, Escalar) ---
    auto t3_start = std::chrono::steady_clock::now();
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Postproc (Abs/Norm/Scale/Convert): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count()<<"ms"<<std::endl;
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"SRM elapsed time (Total): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"Computing"; 
    if (invert) std::cout<<" inverse";
    else std::cout<<" direct";
    std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    
    // --- Medición 1: Preprocesamiento (Grises, Convertir, Obtener Bloques) ---
    auto t1_start = std::chrono::steady_clock::now();
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    auto t1_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Preproc (Grayscale/Blocks): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count()<<"ms"<<std::endl;
    
    // --- Medición 2: Paralelización / Transformación DCT (OpenMP Region) ---
    auto t2_start = std::chrono::steady_clock::now();
    
    // Paraleliza el procesado por bloques (cada iteración es independiente)
    #pragma omp parallel for schedule(static)
    for(int i=0;i<(int)blocks.size();i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
                    for(int k=0;k<blocks[i].size/2;k++)
                        for(int l=0;l<blocks[i].size/2;l++)
                            dctBlock[k][l] = 0.0;
          dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        }else dct::assign(dctBlock, blocks[i], 0);
        dct::delete_matrix(dctBlock);
    }
    
    auto t2_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Parallel DCT/IDCT blocks (OpenMP): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count()<<"ms"<<std::endl;
    
    // --- Medición 3: Postprocesamiento (Convertir a unsigned char) ---
    auto t3_start = std::chrono::steady_clock::now();
    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Postproc (Final Convert): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count()<<"ms"<<std::endl;
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"DCT elapsed time (Total): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality){
    std::cout<<"Computing ELA..."<<std::endl;
    auto begin = std::chrono::steady_clock::now();
    
    // --- Medición 1: Preprocesamiento (Grises y Guardado JPEG) ---
    auto t1_start = std::chrono::steady_clock::now();
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality); // I/O
    auto t1_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Preproc (Grayscale/Save Temp JPEG): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count()<<"ms"<<std::endl;
    
    // --- Medición 2: Carga de imagen comprimida y resta ---
    auto t2_start = std::chrono::steady_clock::now();
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>(); // I/O y conversión
    compressed = compressed + (grayscale.convert<float>()*(-1));
    auto t2_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Load/Convert/Subtract: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count()<<"ms"<<std::endl;
    
    // --- Medición 3: Postprocesamiento (Abs, Normalize, Escalar) ---
    auto t3_start = std::chrono::steady_clock::now();
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    auto t3_end = std::chrono::steady_clock::now();
    std::cout<<"  -> Postproc (Abs/Norm/Scale/Convert): "<<std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count()<<"ms"<<std::endl;
    
    auto end = std::chrono::steady_clock::now();
    std::cout<<"ELA elapsed time (Total): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

int main(int argc, char **argv) {
    auto total_begin = std::chrono::steady_clock::now(); // INICIO DE MEDICIÓN TOTAL DEL MAIN
    if(argc == 1) {
        std::cerr<<"Image filename missing from arguments. Usage ./dct <filename>"<<std::endl;
        exit(1);
    }
    int block_size=8;
    Image<unsigned char> image = load_from_file(argv[1]);
    
    // El código se ejecuta secuencialmente aquí, pero las llamadas internas a compute_* están paralelizadas
    
    Image<unsigned char> srm3x3 = compute_srm(image, 3);
    save_to_file("srm_kernel_3x3.png", srm3x3);
    
    Image<unsigned char> srm5x5 = compute_srm(image, 5);
    save_to_file("srm_kernel_5x5.png", srm5x5);

    Image<unsigned char> ela = compute_ela(image, 90);
    save_to_file("ela.png", ela);
    
    Image<unsigned char> dct_invert = compute_dct(image, block_size, true);
    save_to_file("dct_invert.png", dct_invert);
    
    Image<unsigned char> dct_direct = compute_dct(image, block_size, false);
    save_to_file("dct_direct.png", dct_direct);

    auto total_end = std::chrono::steady_clock::now(); // FIN DE MEDICIÓN TOTAL DEL MAIN
    std::cout << "\nTotal execution time (main body): " << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_begin).count() << "ms" << std::endl;

    return 0;
}