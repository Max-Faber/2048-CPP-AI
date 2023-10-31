#include <Device.h>

#define BATCH_SIZE 8

class Model : torch::nn::Module
{
private:
//    torch::nn::Sequential model;

//            torch::nn::ReLU(),
//            torch::nn::ReLU(),
//            torch::nn::ReLU(),
//            torch::nn::ReLU(),

    torch::nn::Conv2d conv1 = torch::nn::Conv2d(4, 32, 2);
    torch::nn::Conv2d conv2 = torch::nn::Conv2d(32, 64, 2);
    torch::nn::Conv2d conv3 = torch::nn::Conv2d(64, 16, 2);
    torch::nn::Linear lin1 = torch::nn::Linear(16 * 1 * 13, 8);
    torch::nn::Linear lin2 = torch::nn::Linear(8, 4);//numberOutputs);
    torch::nn::ReLU relu = torch::nn::ReLU();
public:
    Model(int numberInputs, int numberOutputs);
    torch::Tensor forward(torch::Tensor x);
    std::vector<torch::Tensor> getParameters();
};
