#include <ExperienceReplay.h>

void ExperienceReplay::addExperience(const torch::Tensor& state, Move action, float reward, const torch::Tensor& newState, bool isTerminal)
{
    experienceReplay.push_back(new Experience(state, action, reward, newState, isTerminal));
    if (experienceReplay.size() > EXPERIENCE_REPLAY_LENGTH)
    {
        delete experienceReplay[0];
        experienceReplay.pop_front();
    }
}

std::deque<Experience*> ExperienceReplay::getExperienceReplay()
{
    return experienceReplay;
}
