#include <GameState.h>
#include <GameRendering.h>
#include <ExperienceReplay.h>
#include <EpsilonAdaptiveGreedy.h>
#include <Model.h>
#include <torch/torch.h>
#include <vector>

#define N_EPISODES 5000
#define LEARNING_RATE 1e-3
#define MODEL_ALIGN_FREQUENCY 500
#define GAMMA .9f

class DQN
{
private:
    Model* qNetwork;
    Model* targetNetwork;
    torch::optim::Adam* optimizer;
    EpsilonAdaptiveGreedy* epsilonAdaptiveGreedy;
    std::mt19937* gen; // Standard mersenne_twister_engine seeded with random_device
    std::uniform_int_distribution<int> randomActionDistributionInt;
    ExperienceReplay experienceReplay;
    std::uniform_real_distribution<> randomActionDistributionReal = std::uniform_real_distribution<>(0, 1);

    void updateModel();
    float calculateReward(const torch::Tensor& state, bool tilesMoved, bool isTerminal, int mergeSum, Move action);
    void initializeRandom();
    static torch::Tensor getState();
    Move getAction(const torch::Tensor& state, float epsilon);
    void alignModels();
public:
    DQN(int numberInputs, int numberOutputs, float learningRate);

    void train();
};
