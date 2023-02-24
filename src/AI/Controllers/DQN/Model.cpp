#include <Model.h>

Model::Model(int numberInputs, int numberOutputs, float learningRate)
{
    int layerParameters = 1024;

    model = torch::nn::Sequential(
            torch::nn::Linear(numberInputs, layerParameters),
            torch::nn::ReLU(),
            torch::nn::Linear(layerParameters, layerParameters),
            torch::nn::ReLU(),
            torch::nn::Linear(layerParameters, numberOutputs)
    );
    model->to(Device::getDevice());
}

torch::Tensor Model::forward(const torch::Tensor& x)
{
    return model->forward(x);
}

std::vector<torch::Tensor> Model::getParameters()
{
    return model->parameters();
}

torch::Tensor Model::normalizeLog2N(const torch::Tensor& input)
{
    return input / torch::log2(torch::tensor(65536));
}

torch::Tensor Model::normalizeMinMax(torch::Tensor input)
{
    return (input - input.min()) / (input.max() - input.min());
}
