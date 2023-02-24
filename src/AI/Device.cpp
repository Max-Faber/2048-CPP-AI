#include <Device.h>

torch::Device Device::device(torch::kCPU);
torch::TensorOptions Device::tensorDeviceOptions;

void Device::checkDevice()
{
    if (torch::hasCUDA()) device = torch::Device(torch::kCUDA);
    // else if (torch::hasMPS()) device = torch::Device(torch::kMPS);
    tensorDeviceOptions = torch::TensorOptions().device(device);
}
