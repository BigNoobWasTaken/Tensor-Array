#include "attention.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API TransformerEncoderImpl final : TensorCalculateLayerImpl 
        {
        private:
            MultiHeadAttention multihead_attn;
            Sequential feed_forward;
            Normalization
                layer_norm_1 = Normalization(std::initializer_list<unsigned char>{0}),
                layer_norm_2 = Normalization(std::initializer_list<unsigned char>{0});
        public:
            TransformerEncoderImpl(unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&) override;
        };
    }
}

