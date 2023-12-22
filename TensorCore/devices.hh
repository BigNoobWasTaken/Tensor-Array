#pragma once

#ifdef CUDA_ML_EXPORTS
#define CUDA_ML_API __declspec(dllexport)
#else
#define CUDA_ML_API __declspec(dllimport)
#endif

namespace tensor_array
{
	namespace devices
	{
		enum DeviceType
		{
			CPU,
			CUDA
		};

		extern thread_local struct Device
		{
			DeviceType dev_t;
			int index;
		} default_dev;

		constexpr Device DEVICE_CPU_0{ CPU,0 };

		void device_memcpy(void*, Device, const void*, Device, size_t);

		void device_memcpy(void*, Device, const void*, Device, size_t, void*);

		void CUDA_ML_API device_CUDA_get_info();
	}
}

void* operator new(size_t, tensor_array::devices::Device);

void* operator new(size_t, tensor_array::devices::Device, void*);

void operator delete(void*, tensor_array::devices::Device);

void operator delete(void*, tensor_array::devices::Device, void*);

#undef CUDA_ML_API