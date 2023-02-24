#include <2048-AI.h>

int main(int argc, char** argv)
{
    Device::checkDevice();
    DQN dqn = DQN(GameState::gridDimension * GameState::gridDimension, 4, 0.001);

    Graphics::init(argc, argv, GameRendering::initialWidth, GameRendering::initialHeight);
    GameState::initialize();
    GameRendering::show();
    dqn.train();
    // std::thread thread(&DQN::train, dqn);
    return 0;
}
