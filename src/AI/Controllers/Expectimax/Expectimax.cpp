#include <Expectimax.h>

int Expectimax::functionCalls = 0;
int Expectimax::possibleTileValues[] = { 2, 4 };

void Expectimax::run()
{
    int maxScore = 0, maxTile = 0;
    int lastDepth = 0;
    std::vector<int> scores;

    for (int episode = 1; episode <= N_EPISODES; episode++)
    {
        GameState::gameState->reset();
        // if (episode == 2) exit(1);
        GameRendering::display();
        while (true)
        {
            int mergeSum;
            bool isTerminal, tilesMoved;
            std::pair<bool, int> moveResult;
            float depth = (float)((GameState::gridDimension * GameState::gridDimension) + 1) - (float)GameState::gameState->emptyFieldPositions.size();
            depth /= 2.f;
            if (depth < 1) depth = 1;
            depth = 5.f;

            if (GameState::gameState->emptyFieldPositions.size() < 4) depth = 6.f;
            else depth = 5.f;
            depth = 4.f;
            if ((int)depth != lastDepth)
            {
                lastDepth = (int)depth;
                // printf("Depth: %d\n", (int)depth);
            }

            Move move = chooseMove((int)depth);

            moveResult = GameState::gameState->makeMove(move, true);
            tilesMoved = moveResult.first;
            mergeSum = moveResult.second;
            Keyboard::tilesMoved = tilesMoved;
            isTerminal = GameState::gameState->isTerminal();
            if (isTerminal) { GameState::gameState->cleanTransitionInfo(); break; }
            GameRendering::display();
            Keyboard::redrawRequired = true;
        }
        maxScore = std::max(maxScore, GameState::gameState->score);
        scores.push_back(GameState::gameState->score);
        for (int x = 0; x < GameState::gridDimension; x++)
            for (int y = 0; y < GameState::gridDimension; y++)
                maxTile = std::max(maxTile, GameState::gameState->getTileValue(x, y));
        printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d\n", episode, GameState::gameState->score, maxScore, maxTile);
        // printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d, moves/sec: %d\n", episode, GameState::gameState->score, maxScore, maxTile, (int)(moves / episodeTime));
    }
    int scoreSum = 0;
    for (int i = 0; i < scores.size(); i++) scoreSum += scores[i];
    printf("Average score: %d\n", (int)(scoreSum / scores.size()));
}

Move Expectimax::chooseMove(int depth)
{
    float highestReward = -INFINITY;
    Move bestMove;

    for (Move move : GameState::moves)
    {
        float reward;
        GameState *gameStateCopy = new GameState(GameState::gameState);
        std::pair<bool, int> mergeResult = gameStateCopy->makeMove(move, false);
        bool tilesMoved = mergeResult.first;

        if (!tilesMoved) continue;
        reward = expectimax(gameStateCopy, false, depth - 1);
        if (reward > highestReward)
        {
            highestReward = reward;
            bestMove = move;
        }
        delete gameStateCopy;
    }
    return bestMove;
}

float Expectimax::expectimax(GameState *gState, bool isMax, int depth)
{
    std::vector<GameState*> possibleStates;
    bool isTerminal = gState->isTerminal();
    float summedReward = 0.f;

    functionCalls += 1;
    // printf("expectimax(), depth: %d, functionCalls: %d\n", depth, functionCalls);
    if (depth == 0 || isTerminal) return calculateReward(gState, isTerminal);
    if (isMax)
    {
        float bestReward = -INFINITY;

        for (Move move : GameState::moves)
        {
            float reward;
            GameState *gameStateCopy = new GameState(gState);

            gameStateCopy->makeMove(move, false);
            reward = expectimax(gameStateCopy, false, depth - 1);
            bestReward = std::max(bestReward, reward);
            delete gameStateCopy;
        }
        return bestReward;
    }
    for (FieldPos *fPos : gState->emptyFieldPositions)
    {
        for (int possibleTileValue : possibleTileValues)
        {
            float reward;
            float probability = possibleTileValue == 2 ? 0.9f : 0.1f;

            fPos->tile = new Tile(possibleTileValue);
            reward = expectimax(gState, true, depth - 1);
            delete fPos->tile;
            fPos->tile = nullptr;
            summedReward += probability * reward;
        }
    }
    return summedReward / (gState->emptyFieldPositions.size() * 2);
}

float Expectimax::calculateReward(GameState *gState, bool isTerminal)
{
    std::vector<std::vector<int>> priorityGrid = {
        // { 15, 14, 13, 12 },
        // { 8, 9, 10, 11 },
        // { 4, 5, 6, 7 },
        // { 0, 1, 2, 3 }
        { 6, 5, 4, 1 },
        { 5, 4, 1, 0 },
        { 4, 1, 0, -1 },
        { 1, 0, -1, -2 }
    };
    float reward = 0.f;
    float penalty = 0.f;
    // int cornerValue = gState->getTileValue(0, 0);
    // std::vector<int> stateFlattened = gState->getStateFlattened();
    // int maxValue = *std::max_element(stateFlattened.begin(), stateFlattened.end());
    // if (cornerValue == maxValue) reward += 100.f;

    for (int row = 0; row < GameState::gridDimension; row++)
    {
        for (int column = 0; column < GameState::gridDimension; column++)
        {
            int tileValue = gState->getTileValue(row, column);
            std::vector<std::vector<int>> directions = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

            reward += tileValue * tileValue * priorityGrid[row][column];
            for (int i = 0; i < directions.size(); i++)
            {
                int x = row + directions[i][0], y = column + directions[i][1];
                int tileValueNeighbour;

                if (x < 0 | x > GameState::gridDimension - 1) continue;
                if (y < 0 | y > GameState::gridDimension - 1) continue;
                tileValueNeighbour = gState->getTileValue(x, y);
                penalty += abs(tileValue - tileValueNeighbour);
            }
        }
    }
    // reward += (float)gState->emptyFieldPositions.size() * 10.f;
    if (isTerminal) reward = abs(reward) * -10;
    return reward - penalty;
}

// float Expectimax::calculateReward(GameState *gState, bool isTerminal)
// {
//     float reward = 0.f;

// //    GameState::printGrid();
// //    int cornerValue = gState->getTileValue(0, 0);
//     int cornerValue = std::max(
//             std::max(
//                     gState->getTileValue(GameState::gridDimension - 1, 0),
//                     gState->getTileValue(0, GameState::gridDimension - 1)
//             ),
//             std::max(
//                     gState->getTileValue(GameState::gridDimension - 1, GameState::gridDimension - 1),
//                     gState->getTileValue(0, 0)
//             )
//     );
// //    if (cornerValue > 0) reward += (float)log2(cornerValue);
//     for (int row = 0; row < GameState::gridDimension; row++)
//     {
//         for (int column = 0; column < GameState::gridDimension - 1; column++)
//         {
//             int a = gState->getTileValue(row, column);
//             int b = gState->getTileValue(row, column + 1);

// //            if (a == 0 || b == 0) continue;
// //            if (a > b) reward += .1f;
//             if (a == b) reward += 2.f;
//         }
//     }
//     std::vector<int> stateFlattened = gState->getStateFlattened();
//     int maxValue = *std::max_element(stateFlattened.begin(), stateFlattened.end());
//     if (cornerValue == maxValue) reward += 100.f;
//     for (int column = 0; column < GameState::gridDimension; column++)
//     {
//         for (int row = 0; row < GameState::gridDimension - 1; row++)
//         {
//             int a = gState->getTileValue(row, column);
//             int b = gState->getTileValue(row + 1, column);

// //            if (a == 0 || b == 0) continue;
// //            if (column % 2 == 1)
// //            {
// //                int temp = a;
// //
// //                a = b;
// //                b = temp;
// //            }
// //            float factor = (float)(column + 1) / GameState::gridDimension;
//             if (a >= b) reward += 1.f;// * factor;
//             else reward -= 5.f;// * factor;
//         }
//     }
// //    reward += (float)gState->emptyFieldPositions.size() * 10.f;
// //    reward += (float)log2(mergeSum);
//     if (isTerminal) reward = abs(reward) * -10;
//     return reward;
// }
