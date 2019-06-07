# -*- coding: utf-8 -*-
import random, os
import numpy as np
from DeepCars import GridWorld as envObj
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import time, sys
import gym, gym_deepcars

MAX_EPISODE = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self.huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model


    def act(self, state):
        act_values = self.model.predict(state)          # Q values for each action
        return np.argmax(act_values[0])  # returns action

    def load(self, name):
        self.model.load_weights(name)


if __name__ == "__main__":

    # for visualization disable next two lines
    # os.environ['SDL_AUDIODRIVER'] = "dummy"  # Create a AUDIO DRIVER to not produce the pygame sound
    # os.environ["SDL_VIDEODRIVER"] = "dummy"  # Create a dummy window to not show the pygame window

    env = gym.make('DeepCars-v0')
    state_size = env.ObservationSpace()
    action_size = env.ActionSpace()
    agent = DQNAgent(state_size, action_size)
    agent.load("./Save/ARC_AVL_DQN.h5")
    batch_size = 32

    state = env.reset()

    episode_rew = [0.0]
    while True:
        action = agent.act(state)
        next_state, reward, done, HitCarsCount, PassedCarsCount = env.step(action, True)
        # time.sleep(.2)
        env.render()
        episode_rew[-1] += reward
        if done:
            print(f'episode_rew={episode_rew[-1]}')
            episode_rew.append(0.0)
            next_state = env.reset()
        state = next_state
        if len(episode_rew) >= MAX_EPISODE:
            break

    print(f'episode mean rewards{np.mean(episode_rew)}')
    import pandas as pd

    dict = {'eps_rew': episode_rew}
    df = pd.DataFrame(dict)
    df.to_csv('Test_log/Test_Log_DQN.csv')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(episode_rew, zorder=1)  # on top
    plt.title('Episode rewards')
    plt.show()
    plt.clf()  # Here is another path