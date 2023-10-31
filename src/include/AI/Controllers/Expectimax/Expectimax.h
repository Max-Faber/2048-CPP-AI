#include <GameState.h>
#include <GameRendering.h>

#define N_EPISODES 100
#define DEPTH 3

class Expectimax {
private:
    static int functionCalls;
    static int possibleTileValues[];

    static Move chooseMove(int depth);
//    static std::vector<GameState*> getChildrenMove(Move move, GameState *gState);
    static std::vector<GameState*> getChildren(GameState *gState);
    static float expectimax(GameState *gState, bool isMax, int depth);
    static float calculateReward(GameState *gState, bool isTerminal);
public:
    static void run();
};
