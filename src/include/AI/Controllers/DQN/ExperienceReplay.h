#include <torch/torch.h>
#include <GameState.h>

#define EXPERIENCE_REPLAY_LENGTH 50000

struct Experience
{
    torch::Tensor state;
    Move action;
    float reward;
    torch::Tensor newState;
    bool isTerminal;

    Experience(const torch::Tensor& state, Move action, float reward, const torch::Tensor& newState, bool isTerminal)
    {
        this->state = state;
        this->action = action;
        this->reward = reward;
        this->newState = newState;
        this->isTerminal = isTerminal;
    }
};

class ExperienceReplay
{
private:
    std::deque<Experience*> experienceReplay;
public:
    void addExperience(const torch::Tensor& state, Move action, float reward, const torch::Tensor& newState, bool isTerminal);
    std::deque<Experience*> getExperienceReplay();
};
