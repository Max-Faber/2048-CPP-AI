#include <DQN.h>

Model *DQN::qNetwork, *DQN::targetNetwork;
torch::optim::Adam *DQN::optimizer;
EpsilonAdaptiveGreedy *DQN::epsilonAdaptiveGreedy;
std::mt19937 *DQN::gen; // Standard mersenne_twister_engine seeded with random_device
std::uniform_int_distribution<int> DQN::randomActionDistributionInt;
ExperienceReplay DQN::experienceReplay;
std::uniform_real_distribution<> DQN::randomActionDistributionReal = std::uniform_real_distribution<>(0, 1);

void DQN::train(int numberInputs, int numberOutputs)
{
    int maxScore = 0, maxTile = 0, timestepsSinceAlignment = 0;
    bool adaptiveEpsilon = true;
    std::vector<int> endScores, endMaxTiles;

    Device::checkDevice();
    qNetwork = new Model(numberInputs, numberOutputs);
    targetNetwork = new Model(numberInputs, numberOutputs);
    optimizer = new torch::optim::Adam(qNetwork->getParameters(), LEARNING_RATE);
    randomActionDistributionInt = std::uniform_int_distribution<>(0, numberOutputs - 1);
    alignModels();
    initializeRandom();

    epsilonAdaptiveGreedy = new EpsilonAdaptiveGreedy(adaptiveEpsilon);
    for (int episode = 1; episode <= N_EPISODES; episode++)
    {
//        printf("Episode %d/%d\n", episode, N_EPISODES);
        GameState::gameState->reset();
        torch::Tensor stateOneHot = getState();
        int moves = 0;
        const clock_t beginTime = clock();

        while (true)
        {
            float reward;
            int mergeSum;
            bool isTerminal, tilesMoved;
            torch::Tensor newState;
            std::pair<bool, int> moveResult;
            std::vector<Move> actions = getAction(stateOneHot, epsilonAdaptiveGreedy->getEpsilon());

            for (Move action : actions)
            {
                moveResult = GameState::gameState->makeMove(action, true);
                tilesMoved = moveResult.first;
                mergeSum = moveResult.second;
                if (!tilesMoved) continue;
                Keyboard::tilesMoved = tilesMoved;
                moves++;
                GameRendering::display();
                Keyboard::redrawRequired = true;
                newState = getState();
                isTerminal = GameState::gameState->isTerminal();
                reward = calculateReward(isTerminal, mergeSum);
                epsilonAdaptiveGreedy->update();
                experienceReplay.addExperience(stateOneHot, action, reward, newState, isTerminal);
                updateModel();
                stateOneHot = newState;
                maxScore = std::max(maxScore, GameState::gameState->score);
                std::vector<int> state = GameState::gameState->getStateFlattened();
                maxTile = std::max(maxTile, *std::max_element(state.begin(), state.end()));
                if ((timestepsSinceAlignment++) == MODEL_ALIGN_FREQUENCY)
                {
                    alignModels();
                    timestepsSinceAlignment = 0;
                }

                std::string moveString;
                if (action == Move::up) moveString = "up";
                else if (action == Move::right) moveString = "right";
                else if (action == Move::down) moveString = "down";
                else moveString = "left";
                // printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d, move: %s, reward: %f\n", episode, GameState::score, maxScore, maxTile, moveString.c_str(), reward);
                break;
            }
            if (isTerminal)
            {
                endScores.push_back(GameState::gameState->score);
                endMaxTiles.push_back(stateOneHot.max().item<int>());
                break;
            }
        }
        float episodeTime = (float)(clock() - beginTime) / CLOCKS_PER_SEC;
        printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d, moves/sec: %d\n", episode, GameState::gameState->score, maxScore, maxTile, (int)(moves / episodeTime));
    }
}

void DQN::updateModel()
{
    std::vector<Experience*> miniBatch;
    torch::Tensor qPredictions, targetPredictions;
    std::vector<torch::Tensor> states, newStates;
    std::deque<Experience*> experiences = experienceReplay.getExperienceReplay();
    torch::nn::MSELoss criterion = torch::nn::MSELoss();

    if (experiences.size() < BATCH_SIZE) return;
    std::sample(experiences.begin(), experiences.end(), std::back_inserter(miniBatch), BATCH_SIZE, *gen);
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        states.push_back(miniBatch[i]->state);
        newStates.push_back(miniBatch[i]->newState);
    }
    qPredictions = qNetwork->forward(torch::stack(states)).to(torch::kCPU);
    targetPredictions = targetNetwork->forward(torch::stack(newStates));
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        float target;
        float reward = miniBatch[i]->reward;

        if (miniBatch[i]->isTerminal) target = reward;
        else target = reward + GAMMA * targetPredictions[i].max().item<float>();
        qPredictions[i][miniBatch[i]->action] = target;
    }
    optimizer->zero_grad();
    qPredictions = qPredictions.to(Device::getDevice());
    torch::Tensor loss = criterion(qNetwork->forward(torch::stack(states)), qPredictions);
    loss.backward();
    optimizer->step();
}

float DQN::calculateReward(bool isTerminal, int mergeSum)
{
    float reward = 0.f;

//    GameState::printGrid();
    int cornerValue = std::max(
            std::max(
                    GameState::gameState->getTileValue(GameState::gridDimension - 1, 0),
                    GameState::gameState->getTileValue(0, GameState::gridDimension - 1)
            ),
            std::max(
                    GameState::gameState->getTileValue(GameState::gridDimension - 1, GameState::gridDimension - 1),
                    GameState::gameState->getTileValue(0, 0)
            )
    );
//     cornerValue = GameState::getTileValue(0, 0);
//    if (cornerValue > 0) reward += (float)log2(cornerValue);
    for (int row = 0; row < GameState::gridDimension; row++)
    {
        for (int column = 0; column < GameState::gridDimension - 1; column++)
        {
            int a = GameState::gameState->getTileValue(row, column);
            int b = GameState::gameState->getTileValue(row, column + 1);

            if (a == 0 || b == 0) continue;
//            if (a > b) reward += .1f;
            if (a == b) reward += 2.f;
        }
    }
    std::vector<int> stateFlattened = GameState::gameState->getStateFlattened();
    int maxValue = *std::max_element(stateFlattened.begin(), stateFlattened.end());
    if (cornerValue == maxValue) reward += 100.f;
    for (int column = 0; column < GameState::gridDimension; column++)
    {
        for (int row = 0; row < GameState::gridDimension - 1; row++)
        {
            int a = GameState::gameState->getTileValue(row, column);
            int b = GameState::gameState->getTileValue(row + 1, column);

            if (a == 0 || b == 0) continue;
            if (column % 2 == 1)
            {
                int temp = a;

                a = b;
                b = temp;
            }
//            float factor = (float)(column + 1) / GameState::gridDimension;
            if (a >= b) reward += 1.f;// * factor;
            else reward -= 5.f;// * factor;
        }
    }
    reward += (float)GameState::gameState->emptyFieldPositions.size();
    reward += (float)log2(mergeSum);
    if (isTerminal) reward = abs(reward) * -10;
    return reward;
}

void DQN::initializeRandom()
{
    std::random_device rd; // Will be used to obtain a seed for the random number engine

    gen = new std::mt19937(rd());
}

//torch::Tensor DQN::getState()
//{
//    std::vector<int> stateFlattened = GameState::getStateFlattened();
//    std::vector<int> stateFlattenedOneHot;
//
//    for (int i = 0; i < stateFlattened.size(); i++)
//    {
//        std::vector<int> tileOneHot = std::vector<int>(stateFlattened.size());
//        int val = stateFlattened[i];
//
//        if (val > 0) tileOneHot[(int)log2(val) - 1] = 1;
//        stateFlattenedOneHot.insert(std::end(stateFlattenedOneHot), tileOneHot.begin(), tileOneHot.end());
//    }
//    return torch::tensor(stateFlattenedOneHot, Device::getTensorDeviceOptions().dtype(torch::kFloat));
////    return torch::tensor(GameState::getStateFlattened(), Device::getTensorDeviceOptions().dtype(torch::kFloat));
//}

torch::Tensor DQN::getState()
{
    std::vector<int> stateFlattened = GameState::gameState->getStateFlattened();
    torch::Tensor stateOneHot = torch::zeros({GameState::gridDimension, GameState::gridDimension, (int)pow(GameState::gridDimension, 2)}, Device::getTensorDeviceOptions().dtype(torch::kFloat));

    for (int row = 0; row < GameState::gridDimension; row++)
    {
        for (int column = 0; column < GameState::gridDimension; column++)
        {
            int val = GameState::gameState->getTileValue(row, column);

            if (val == 0) continue;
            stateOneHot[row][column][(int)log2(val) - 1] = 1.f;
        }
    }
    return stateOneHot;
}

std::vector<Move> DQN::getAction(const torch::Tensor& state, float epsilon)
{
    if (randomActionDistributionReal(*gen) < epsilon)
    {
        int action = randomActionDistributionInt(*gen);

        return std::vector<Move> { static_cast<Move>(action) };
    }
    torch::Tensor qValues = qNetwork->forward(state);
//    torch::Tensor qValueIndex = torch::argmax(qValues);
    torch::Tensor qValuesSorted = torch::argsort(qValues, -1, true);
    if (qValuesSorted.is_contiguous())
    {
        qValuesSorted = qValuesSorted.contiguous();
    }
    std::vector<Move> moves(qValuesSorted.numel());
    for (int i = 0; i < qValuesSorted.numel(); i++)
    {
        moves[i] = (Move)qValuesSorted[0][i].item<int>();
    }
    // return static_cast<std::vector<Move>>(qValueIndex.item<int>());
    return moves;
}

void DQN::alignModels()
{
    std::vector<torch::Tensor> qNetworkParameters = qNetwork->getParameters();
    std::vector<torch::Tensor> targetNetworkParameters = targetNetwork->getParameters();
    size_t numberParameters = qNetworkParameters.size();

    for (int i = 0; i < numberParameters; i++) targetNetworkParameters[i].data().copy_(qNetworkParameters[i].data());
}
