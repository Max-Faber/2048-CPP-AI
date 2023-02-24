#include <torch/torch.h>
#include <Device.h>

#define BATCH_SIZE 128

class Model : torch::nn::Module
{
private:
    torch::nn::Sequential model;
public:
    Model(int numberInputs, int numberOutputs, float learningRate);
    torch::Tensor forward(const torch::Tensor& x);
    std::vector<torch::Tensor> getParameters();
    static torch::Tensor normalizeLog2N(const torch::Tensor& input);
    static torch::Tensor normalizeMinMax(torch::Tensor input);
};
