#include <2048-AI.h>

int main(int argc, char **argv)
{
    GameState::gameState = new GameState();

    GameState::gameState->initialize();
    Graphics::init(argc, argv, GameRendering::initialWidth, GameRendering::initialHeight);

    GameRendering::show();

    // DQN::train((int)pow(GameState::gridDimension, 4), 4);
    Expectimax::run();
    return 0;
}
