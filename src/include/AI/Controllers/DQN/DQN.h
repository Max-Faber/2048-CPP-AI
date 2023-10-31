#include <GameState.h>
#include <GameRendering.h>
#include <ExperienceReplay.h>
#include <EpsilonAdaptiveGreedy.h>
#include <Model.h>
#include <torch/torch.h>
#include <vector>

#define N_EPISODES 5000
#define LEARNING_RATE 1e-7
#define MODEL_ALIGN_FREQUENCY 500
#define GAMMA .9f

class DQN
{
private:
    static Model *qNetwork, *targetNetwork;
    static torch::optim::Adam *optimizer;
    static EpsilonAdaptiveGreedy *epsilonAdaptiveGreedy;
    static std::mt19937 *gen; // Standard mersenne_twister_engine seeded with random_device
    static std::uniform_int_distribution<int> randomActionDistributionInt;
    static ExperienceReplay experienceReplay;
    static std::uniform_real_distribution<> randomActionDistributionReal;

    static void updateModel();
    static float calculateReward(bool isTerminal, int mergeSum);
    static void initializeRandom();
    static torch::Tensor getState();
    static std::vector<Move> getAction(const torch::Tensor& state, float epsilon);
    static void alignModels();
public:
    static void train(int numberInputs, int numberOutputs);
};
