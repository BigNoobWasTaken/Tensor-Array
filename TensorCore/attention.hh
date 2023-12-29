#include "linear.hh"
#include "sequential.hh"
#include "normalization.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        value::Tensor CUDA_ML_API scaled_dot_product_attention(const value::Tensor& q, const value::Tensor& k, const value::Tensor& v);

        class CUDA_ML_API MultiHeadAttentionImpl final :
            public LayerImpl,
            public CalculateStruct<value::Tensor, const value::Tensor&, const value::Tensor&, const value::Tensor&>
        {
        private:
            const unsigned int d_model, n_head;
            Linear w_q, w_k, w_v, w_o;
        public:
            MultiHeadAttentionImpl(unsigned int, unsigned int);
            value::Tensor calculate(const value::Tensor&, const value::Tensor&, const value::Tensor&) override;
        };

        using MultiHeadAttention = LayerHolder<MultiHeadAttentionImpl>;
    }
}