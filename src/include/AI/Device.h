#include <torch/torch.h>

class Device
{
private:
    static torch::Device device;
    static torch::TensorOptions tensorDeviceOptions;
public:
    static void checkDevice();
    static torch::Device getDevice() { return device; };
    static torch::TensorOptions getTensorDeviceOptions() { return tensorDeviceOptions; }
};