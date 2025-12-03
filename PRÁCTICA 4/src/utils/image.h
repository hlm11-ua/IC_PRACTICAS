#ifndef __IMAGE__H__
#define __IMAGE__H__
#include <vector>
#include <memory>
#include <iostream>
#include "assert.h"
#include <string>

template <typename T> class Block;

template <typename T> class Image{
public:
  int width, height, channels;
  std::shared_ptr<T[]> matrix;
  
  void release();
  Image();
  Image(int width, int height, int channels);
  Image(const Image<T> &a);
  ~Image();
  
  Image<T> operator=(const Image<T>& other);
  Image<T> operator*(const Image<T>& other) const;
  Image<T> operator*(float scalar) const;
  Image<T> operator+(const Image<T>& other) const;
  Image<T> operator+(float scalar) const;
  
  T get(int row, int col, int channel) const;
  void set(int row, int col, int channel, T value);
  
  template <typename S> Image<S> convert() const;
  Image<T> to_grayscale() const;
  Image<T> abs() const;
  Image<float> normalized() const;
  Image<T> convolution(const Image<float> &kernel) const;
  std::vector<Block<T>> get_blocks(int block_size=8);
};

Image<unsigned char> load_from_file(const std::string &filename);
void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality=100);

template <typename T> class Block {
public:
    int row, col, size, channel;
    Image<T> *image; // Referencia a la imagen padre
    Block(Image<T> &img, int r, int c, int s, int ch) : row(r), col(c), size(s), channel(ch), image(&img) {}
    
    T get_pixel(int r, int c, int ch) const { return image->get(row+r, col+c, ch); }
    // Helpers para la DCT que espera funciones tipo matriz
    int get_size() const { return size; }
};

#endif