#include <Model.h>

Model::Model(int numberInputs, int numberOutputs)
{
    int layerParameters = 1024;



//            torch::nn::Linear(numberInputs, layerParameters),
//            torch::nn::ReLU(),
//            torch::nn::Linear(layerParameters, layerParameters),
//            torch::nn::ReLU(),
//            torch::nn::Linear(layerParameters, numberOutputs)
//    );
    // model->to(Device::getDevice());
}

torch::Tensor Model::forward(torch::Tensor x)
{
    x = this->conv1(x);
    x = this->relu(x);
    x = this->conv2(x);
    x = this->relu(x);
    x = this->conv3(x);
    x = this->relu(x);
    x = torch::reshape(x, {-1, 16 * 1 * 13});
    x = this->lin1(x);
    x = this->relu(x);
    x = this->lin2(x);
    return x;
}

//Model::Model(int numberInputs, int numberOutputs)
//{
//    int layerParameters = 1024;
//
//    model = torch::nn::Sequential(
//            torch::nn::Linear(numberInputs, layerParameters),
//            torch::nn::ReLU(),
//            torch::nn::Linear(layerParameters, layerParameters),
//            torch::nn::ReLU(),
//            torch::nn::Linear(layerParameters, numberOutputs)
//    );
//    model->to(Device::getDevice());
//}
//
//torch::Tensor Model::forward(torch::Tensor x)
//{
//    return model->forward(x);
//}

std::vector<torch::Tensor> Model::getParameters()
{
    return this->parameters();
}
