#pragma once
#include "tensor.h"

template <typename T>
std::ostream &operator<< (std::ostream &os, const Tensor<T> &tensor);

template <typename T>
Tensor<T> operator+ (const Tensor<T> &left, const Tensor<T> &right);
template <typename T>
Tensor<T> operator- (const Tensor<T> &left, const Tensor<T> &right);
template <typename T>

Tensor<T> operator* (const Tensor<T> &left, const Tensor<T> &right);
template <typename T> Tensor<T> operator* (const Tensor<T> &left, T right);
template <typename T> Tensor<T> operator* (T left, const Tensor<T> &right);

template <typename T> Tensor<T> operator/ (const Tensor<T> &left, T right);

template <typename T>
bool operator== (const Tensor<T> &left, const Tensor<T> &right);
