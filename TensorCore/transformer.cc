#include "transformer.hh"
#include "layer_utility.hh"

namespace tensor_array
{
	namespace layers
	{
		TransformerEncoderImpl::TransformerEncoderImpl(unsigned int d_model, unsigned int n_head) :
			multihead_attn(d_model, n_head),
			feed_forward
			{
				Linear(4 * d_model),
				Activation(&ReLU),
				Linear(d_model)
			}
		{
		}
		value::Tensor TransformerEncoderImpl::calculate(const value::Tensor& input)
		{
			auto attn_output = this->multihead_attn(input, input, input);
			attn_output = this->layer_norm_1(input + attn_output);
			auto ff_output = this->feed_forward(attn_output);
			return this->layer_norm_2(attn_output + ff_output);
		}
	}
}
