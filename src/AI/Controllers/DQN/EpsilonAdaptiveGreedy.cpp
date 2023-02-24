#include <EpsilonAdaptiveGreedy.h>

EpsilonAdaptiveGreedy::EpsilonAdaptiveGreedy(bool adaptive)
{
    epsilon = adaptive ? 1.f : epsilonMin;
}

void EpsilonAdaptiveGreedy::update()
{
    epsilon *= epsilon > epsilonMin ? epsilonDecay : 1.f;
}

float EpsilonAdaptiveGreedy::getEpsilon()
{
    return epsilon;
}
