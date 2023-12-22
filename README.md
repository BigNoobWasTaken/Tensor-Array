# Tensor-Array

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)

A C++ Tensor library that use to work with machine learning or deep learning project.

Build your own neural network models with this library.

## Why this reposity named `Tensor-Array`
I created a template struct that named `TensorArray` is a multi-dimensional array wrapper.

```C++
#include "tensorbase.hh"

int main()
{
  tensor_array::value::TensorArray<float, 4, 4> example_tensor_array =
  {{
    {{ 1, 2, 3, 4 }},
    {{ 5, 6, 7, 8 }},
    {{ 9, 10, 11, 12 }},
    {{ 13, 14, 15, 16 }},
  }};
  return 0;
}

```

## The `Tensor` class.
The `Tensor` class is a storage that store value and calculate the tensor.

```C++
#include "tensor.hh"

int main()
{
  tensor_array::value::TensorArray<float, 4, 4> example_tensor_array =
  {{
    {{ 1, 2, 3, 4 }},
    {{ 5, 6, 7, 8 }},
    {{ 9, 10, 11, 12 }},
    {{ 13, 14, 15, 16 }},
  }};
  tensor_array::value::TensorArray<float> example_tensor_array_scalar = {100};
  tensor_array::value::Tensor example_tensor_1(example_tensor_array);
  tensor_array::value::Tensor example_tensor_2(example_tensor_array_scalar);
  tensor_array::value::Tensor example_tensor_sum = example_tensor_1 + example_tensor_2;
  return 0;
}

```
