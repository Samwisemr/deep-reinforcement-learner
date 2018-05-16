import copy
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Deep Q Learning Agent Class
# ========================================================================
class DeepQLearningAgent:

    def __init__(self, env):
        self.env = env                      # scenario
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=10000)   # stores agent's experiences
        self.gamma = 0.95                   # decay rate
        self.epsilon = 1.0                  # exploration
        self.epsilon_decay = .995           # decrease exploration rate
        self.epsilon_min = 0.1              # explore at least this much
        self.learning_rate = 0.001          # rate of learning per iteration
        self._build_model()                 # build the neural network model

    # build the neural network model
    def _build_model(self):
        model = Sequential([
            Dense(48, activation='relu', input_dim=self.state_size),
            Dense(48, activation='relu'),
            Dense(self.action_size, activation='linear'),
        ])

        # loss='mse' means "minimize the mean_squared_error
        model.compile(loss='mse',
                optimizer=Adam(lr=self.learning_rate))

        self.model = model

    # store experiences learned from training in memory
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # retrain network with experiences from memory
    def replay(self, batch_size):
        # make batches a random sampling of memories from memory
        #   here batches is a list of indeces in memory
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)

        for i in batches:
            # get info from the i-th memory in memory
            state, action, reward, next_state = self.memory[i]

            # if we are not looking at the latest experience in memory
            if i != (len(self.memory) - 1):
                # predict the future discounted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward

            # approximately map the current state to future discounted reward
            #   call it target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # train the neural network with the state and target_f
            #   feeds state and target_f info to model ==> makes neural net to predict
            #   the reward value (target_f) from a certain state
            self.model.fit(state, target_f, epochs=1, verbose=0)

            # decay epsilon if we've reached our minimum explorations
            if self.epsilon > self.epsilon_min:
                self.epsilon += self.epsilon_decay

    # decide how to act
    def act(self, state):
        # act randomly at first to test different approaches
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()

        # predict reward value based on the given state
        act_reward_values = self.model.predict(state)

        # pick action that correlates to maximum reward
        return np.argmax(act_reward_values[0])


if __name__ == "__main__":

    EPISODES = 100
    np.random.seed(7)
    done = False

    # main loops
    # ========================================================================

    # initialize gym environment and the agent
    env = gym.make('MountainCar-v0')
    print("State size = {}".format(env.observation_space.shape[0]))
    print("Action size = {}".format(env.action_space.n))
    agent = DeepQLearningAgent(env)

    # per number of times we want to run the scenario
    for i_episode in range(EPISODES):
        # observation stores the state of the scenario
        state = env.reset()
        state = np.reshape(state, [1, 2])
        maxPosition = state[0][0]

        for t in range(10000):
            env.render()

            # decide next action
            action = agent.act(state)

            # advance scenario to next frame
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 2])

            position = next_state[0][0]

            # reward agent for how far up the right slope it got
            reward = 100 if done else position

            # remember the previous experience
            agent.remember(state, action, reward, next_state)

            # update current state for next frame
            state = copy.deepcopy(next_state)

            # update max position
            if (position > maxPosition):
                maxPosition = position

            # when the game ends
            if done:
                print("Episode: {}/{}, Score: {}".format(i_episode, EPISODES, maxPosition))
                break

        # train the agent with the experience of the episode
        agent.replay(32)

