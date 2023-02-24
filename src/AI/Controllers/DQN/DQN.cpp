#include <DQN.h>

DQN::DQN(int numberInputs, int numberOutputs, float learningRate)
{
    qNetwork = new Model(numberInputs, numberOutputs, learningRate);
    targetNetwork = new Model(numberInputs, numberOutputs, learningRate);
    optimizer = new torch::optim::Adam(qNetwork->getParameters(), LEARNING_RATE);
    randomActionDistributionInt = std::uniform_int_distribution<>(0, numberOutputs - 1);
    alignModels();
    initializeRandom();
}

void DQN::train()
{
    int maxScore = 0, maxTile = 0, timestepsSinceAlignment = 0;
    bool adaptiveEpsilon = true;
    std::vector<int> endScores, endMaxTiles;

    epsilonAdaptiveGreedy = new EpsilonAdaptiveGreedy(adaptiveEpsilon);
    for (int episode = 1; episode <= N_EPISODES; episode++)
    {
        printf("Episode %d/%d\n", episode, N_EPISODES);
        GameState::reset();
        torch::Tensor state = getState();

        while (true)
        {
            float reward;
            int mergeSum;
            bool isTerminal, tilesMoved;
            torch::Tensor newState;
            std::pair<bool, int> moveResult;
            Move action = getAction(state, epsilonAdaptiveGreedy->getEpsilon());

            moveResult = GameState::makeMove(action);
            tilesMoved = moveResult.first;
            mergeSum = moveResult.second;
            if (tilesMoved)
            {
                GameState::spawnTileRandom();
                GameRendering::display();
            }
            newState = getState();
            isTerminal = GameState::isTerminal();
            reward = calculateReward(state, tilesMoved, isTerminal, mergeSum, action);
            epsilonAdaptiveGreedy->update();
            experienceReplay.addExperience(state, action, reward, newState, isTerminal);
            updateModel();
            state = newState;
            maxScore = std::max(maxScore, GameState::score);
            maxTile = std::max(maxTile, state.max().item<int>());
            if ((timestepsSinceAlignment++) == 500)
            {
                alignModels();
                timestepsSinceAlignment = 0;
            }

            std::string moveString;
            if (action == Move::up) moveString = "up";
            else if (action == Move::right) moveString = "right";
            else if (action == Move::down) moveString = "down";
            else moveString = "left";
            printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d, move: %s, reward: %f\n", episode, GameState::score, maxScore, maxTile, moveString.c_str(), reward);
            if (isTerminal)
            {
                endScores.push_back(GameState::score);
                endMaxTiles.push_back(maxTile);
                break;
            }
            // usleep(500000);
        }
        // printf("Episode: %d, score: %d, maxScore: %d, maxTile: %d\n", episode, GameState::score, maxScore, maxTile);
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
        states.push_back(qNetwork->normalizeMinMax(miniBatch[i]->state));
        newStates.push_back(qNetwork->normalizeMinMax(miniBatch[i]->newState));
    }
    qPredictions = qNetwork->forward(torch::stack(states)).to(torch::kCPU);
    targetPredictions = targetNetwork->forward(torch::stack(newStates));
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        float target = 0;
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

float DQN::calculateReward(const torch::Tensor& state, bool tilesMoved, bool isTerminal, int mergeSum, Move action)
{
    float reward = 0.f;
//    int maxTileValue = state.max().item<int>();
//    int topLeft = GameState::fieldTileRows[0][0]->tile ? GameState::fieldTileRows[0][0]->tile->val : -1;
//    int topRight = GameState::fieldTileRows[0][GameState::gridDimension - 1]->tile ? GameState::fieldTileRows[0][GameState::gridDimension - 1]->tile->val : -1;
//    int bottomLeft = GameState::fieldTileRows[GameState::gridDimension - 1][0]->tile ? GameState::fieldTileRows[GameState::gridDimension - 1][0]->tile->val : -1;
//    int bottomRight = GameState::fieldTileRows[GameState::gridDimension - 1][GameState::gridDimension - 1]->tile ? GameState::fieldTileRows[GameState::gridDimension - 1][GameState::gridDimension - 1]->tile->val : -1;
//    int maxCornerValue = std::max(
//            std::max(topLeft, topRight),
//            std::max(bottomLeft, bottomRight)
//    );
//
//    if (!tilesMoved) reward -= 25.f;
//    if (isTerminal) reward -= 50.f;
//    if (maxCornerValue == 1) return reward;
//    reward += (float)maxCornerValue / (float)maxTileValue;
//    mergeSum = mergeSum > maxTileValue ? maxTileValue : mergeSum;
//    reward += (float)mergeSum / (float)maxTileValue;
//    return reward;

    for (int row = 0; row < GameState::gridDimension; row++)
    {
        bool interruptionFound = false;
        int selectedTileValue = GameState::getTileValue(row, 0);

        if (selectedTileValue == 1) continue;
        for (int column = 1; column < GameState::gridDimension - 1; column++)
        {
            int currentTileValue = GameState::getTileValue(row, column);

            if (currentTileValue == 1) continue;
            if (currentTileValue > selectedTileValue)
            {
                interruptionFound = true;
                break;
            }
            reward += 1.f;
            // if (currentTileValue <= GameState::getTileValue(row, column + 1))
            // reward += 1.f;
        }
        if (interruptionFound) break;
        // if (row == GameState::gridDimension - 1) reward += 5.f;
    }
    for (int column = 0; column < GameState::gridDimension; column++)
    {
        bool interruptionFound = false;
        int selectedTileValue = GameState::getTileValue(0, column);

        if (selectedTileValue == 1) continue;
        for (int row = 1; row < GameState::gridDimension - 1; row++)
        {
            int currentTileValue = GameState::getTileValue(row, column);

            if (currentTileValue == 1) continue;
            if (currentTileValue > selectedTileValue)
            {
                interruptionFound = true;
                break;
            }
            reward += 1.f;
            // if (currentTileValue <= GameState::getTileValue(row + 1, column))
            //     reward += 1.f;
        }
        if (interruptionFound) break;
        // if (column == GameState::gridDimension - 1) reward += 5.f;
    }
    if (!tilesMoved) reward *= 0.8f;
    if (isTerminal) reward = -50.f;
    reward += mergeSum > 0 ? 5.f : 0.f;
    return reward;
}

void DQN::initializeRandom()
{
    std::random_device rd; // Will be used to obtain a seed for the random number engine

    gen = new std::mt19937(rd());
}

torch::Tensor DQN::getState()
{
    std::vector<int> stateFlattened = GameState::getStateFlattened();

    return torch::tensor(GameState::getStateFlattened(), Device::getTensorDeviceOptions().dtype(torch::kFloat));
}

Move DQN::getAction(const torch::Tensor& state, float epsilon)
{
    if (randomActionDistributionReal(*gen) < epsilon)
    {
        int action = randomActionDistributionInt(*gen);

        return static_cast<Move>(action);
    }
    torch::Tensor qValues = qNetwork->forward(state);
    torch::Tensor qValueIndex = torch::argmax(qValues);
    return static_cast<Move>(qValueIndex.item<int>());
}

void DQN::alignModels()
{
    std::vector<torch::Tensor> qNetworkParameters = qNetwork->getParameters();
    std::vector<torch::Tensor> targetNetworkParameters = targetNetwork->getParameters();
    size_t numberParameters = qNetworkParameters.size();

    for (int i = 0; i < numberParameters; i++) targetNetworkParameters[i].data().copy_(qNetworkParameters[i].data());
}
