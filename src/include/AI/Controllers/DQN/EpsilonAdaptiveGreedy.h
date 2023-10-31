#include <cstdio>

class EpsilonAdaptiveGreedy
{
private:
    constexpr static float epsilonMin = .01f;
    constexpr static float epsilonDecay = .9999f;
    float epsilon;
public:
    EpsilonAdaptiveGreedy(bool adaptive = true);
    void update();
    float getEpsilon();
};