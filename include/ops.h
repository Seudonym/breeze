#pragma once
#include "tensor.h"

template <typename T>
Tensor<T> operator+ (const Tensor<T> &left, const Tensor<T> &right);
template <typename T>
Tensor<T> operator- (const Tensor<T> &left, const Tensor<T> &right);
template <typename T>
Tensor<T> operator* (const Tensor<T> &left, const Tensor<T> &right);

template <typename T>
bool operator== (const Tensor<T> &left, const Tensor<T> &right);
