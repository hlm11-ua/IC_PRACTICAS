#include "image.h"
#include "png.h"
#include "jpeglib.h"
#include <filesystem>
#include <cstring>
#include <cmath>

// --- Implementación básica de la clase Image ---
template <typename T>
Image<T>::Image() : width(0), height(0), channels(0), matrix(nullptr) {}

template <typename T>
Image<T>::Image(int w, int h, int c) : width(w), height(h), channels(c) {
    matrix = std::shared_ptr<T[]>(new T[w * h * c], std::default_delete<T[]>());
}

template <typename T>
Image<T>::Image(const Image<T> &a) : width(a.width), height(a.height), channels(a.channels), matrix(a.matrix) {}

template <typename T>
Image<T>::~Image() {}

template <typename T>
Image<T> Image<T>::operator=(const Image<T>& other){
    width = other.width; height = other.height; channels = other.channels;
    matrix = other.matrix;
    return *this;
}

template <typename T>
T Image<T>::get(int row, int col, int channel) const {
    if (row < 0 || row >= height || col < 0 || col >= width) return 0; 
    return matrix[(row * width + col) * channels + channel];
}

template <typename T>
void Image<T>::set(int row, int col, int channel, T value) {
    if (row >= 0 && row < height && col >= 0 && col < width)
        matrix[(row * width + col) * channels + channel] = value;
}

// Conversiones y Operadores
template <typename T> template <typename S> 
Image<S> Image<T>::convert() const {
    Image<S> res(width, height, channels);
    for(int i=0; i<width*height*channels; i++) res.matrix[i] = (S)matrix[i];
    return res;
}

template <typename T>
Image<T> Image<T>::to_grayscale() const {
    if(channels == 1) return *this;
    Image<T> res(width, height, 1);
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            float val = 0.299 * get(y,x,0) + 0.587 * get(y,x,1) + 0.114 * get(y,x,2);
            res.set(y, x, 0, (T)val);
        }
    }
    return res;
}

template <typename T>
Image<T> Image<T>::abs() const {
    Image<T> res(width, height, channels);
    for(int i=0; i<width*height*channels; i++) {
        res.matrix[i] = matrix[i] > 0 ? matrix[i] : -matrix[i];
    }
    return res;
}

template <typename T>
Image<float> Image<T>::normalized() const {
    Image<float> res(width, height, channels);
    float min_v = 1e9, max_v = -1e9;
    for(int i=0; i<width*height*channels; i++) {
        if((float)matrix[i] < min_v) min_v = (float)matrix[i];
        if((float)matrix[i] > max_v) max_v = (float)matrix[i];
    }
    float range = max_v - min_v;
    if(range == 0) range = 1;
    for(int i=0; i<width*height*channels; i++) {
        res.matrix[i] = ((float)matrix[i] - min_v) / range;
    }
    return res;
}

template <typename T>
Image<T> Image<T>::operator+(const Image<T>& other) const {
    Image<T> res(width, height, channels);
    for(int i=0; i<width*height*channels; i++) res.matrix[i] = matrix[i] + other.matrix[i];
    return res;
}

template <typename T>
Image<T> Image<T>::operator*(float scalar) const {
    Image<T> res(width, height, channels);
    for(int i=0; i<width*height*channels; i++) res.matrix[i] = matrix[i] * scalar;
    return res;
}

// Convolución básica
template <typename T>
Image<T> Image<T>::convolution(const Image<float> &kernel) const {
    Image<T> res(width, height, channels);
    int kw = kernel.width;
    int kh = kernel.height;
    int kw2 = kw/2; 
    int kh2 = kh/2;

    for(int c=0; c<channels; c++){
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                float sum = 0;
                for(int ky=0; ky<kh; ky++){
                    for(int kx=0; kx<kw; kx++){
                        // Coordenadas imagen
                        int iy = y + (ky - kh2);
                        int ix = x + (kx - kw2);
                        sum += (float)get(iy, ix, c) * kernel.get(ky, kx, 0);
                    }
                }
                res.set(y, x, c, (T)sum);
            }
        }
    }
    return res;
}

template <typename T>
std::vector<Block<T>> Image<T>::get_blocks(int block_size) {
    std::vector<Block<T>> blocks;
    // Implementación simplificada si fuera necesaria
    return blocks;
}

// --- Lectura/Escritura (PNG/JPEG) ---

Image<unsigned char> read_png(const std::string &filename) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if(!fp) return Image<unsigned char>();

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    int channels = 3; // Forzamos RGB para simplificar lab
    png_byte color_type = png_get_color_type(png, info);
    
    if(color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if(png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);

    png_read_update_info(png, info);
    Image<unsigned char> img(width, height, channels);
    
    std::vector<png_bytep> row_pointers(height);
    for(int y=0; y<height; y++) 
        row_pointers[y] = &img.matrix[y * width * channels];

    png_read_image(png, row_pointers.data());
    fclose(fp);
    return img;
}

Image<unsigned char> read_jpeg(const std::string &filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile = fopen(filename.c_str(), "rb");
    if(!infile) return Image<unsigned char>();

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int w = cinfo.output_width;
    int h = cinfo.output_height;
    int c = cinfo.output_components;
    Image<unsigned char> img(w, h, c);

    while(cinfo.output_scanline < cinfo.output_height) {
        unsigned char* buffer_array[1];
        buffer_array[0] = &img.matrix[cinfo.output_scanline * w * c];
        jpeg_read_scanlines(&cinfo, buffer_array, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return img;
}

Image<unsigned char> load_from_file(const std::string &filename){
    if (filename.find(".png") != std::string::npos) return read_png(filename);
    if (filename.find(".jpg") != std::string::npos || filename.find(".jpeg") != std::string::npos) return read_jpeg(filename);
    return Image<unsigned char>();
}

void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality){
    if(filename.find(".png") != std::string::npos) {
        FILE *fp = fopen(filename.c_str(), "wb");
        if(!fp) return;
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop info = png_create_info_struct(png);
        if(setjmp(png_jmpbuf(png))) return;
        png_init_io(png, fp);
        png_set_IHDR(png, info, image.width, image.height, 8, 
                     (image.channels==3)?PNG_COLOR_TYPE_RGB:PNG_COLOR_TYPE_GRAY, 
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        std::vector<png_bytep> rows(image.height);
        for(int y=0; y<image.height; y++) 
            rows[y] = (png_bytep)&image.matrix[y * image.width * image.channels];
        png_write_image(png, rows.data());
        png_write_end(png, NULL);
        fclose(fp);
    } else {
        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;
        FILE *outfile = fopen(filename.c_str(), "wb");
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, outfile);
        cinfo.image_width = image.width;
        cinfo.image_height = image.height;
        cinfo.input_components = image.channels;
        cinfo.in_color_space = (image.channels==3)?JCS_RGB:JCS_GRAYSCALE;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality, TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        while (cinfo.next_scanline < cinfo.image_height) {
            JSAMPROW row_pointer[1];
            row_pointer[0] = (JSAMPROW)&image.matrix[cinfo.next_scanline * image.width * image.channels];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
    }
}

// ---------------------------------------------------------------------
// INSTANCIACIONES EXPLÍCITAS (Solución al error de Linker)
// ---------------------------------------------------------------------

// 1. Instanciar las Clases completas
template class Image<unsigned char>;
template class Image<float>;

// 2. Instanciar los Métodos Template específicos (convert)
//    Esto es necesario porque 'convert' es un template miembro.
template Image<float> Image<unsigned char>::convert<float>() const;
template Image<unsigned char> Image<float>::convert<unsigned char>() const;
template Image<unsigned char> Image<unsigned char>::convert<unsigned char>() const;