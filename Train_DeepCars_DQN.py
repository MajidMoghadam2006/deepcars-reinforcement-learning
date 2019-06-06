# -*- coding: utf-8 -*-
import os, random, time
import numpy as np
from DeepCars import GridWorld as envObj
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
import gym, gym_deepcars

MAX_STEPS = 100000
SAVE_FREQ = 5000
PRINT_FREQ = 1

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.steps = 0
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
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self.huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)          # Q values for each action
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                Q_target = reward
            else:
                Q_target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            Q_f = self.model.predict(state)
            Q_f[0][action] = Q_target
            self.model.fit(state, Q_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":

    # for visualization disable next two lines
    # os.environ['SDL_AUDIODRIVER'] = "dummy"  # Create a AUDIO DRIVER to not produce the pygame sound
    # os.environ["SDL_VIDEODRIVER"] = "dummy"  # Create a dummy window to not show the pygame window

    # open text file to save information
    # f = open("Save/Data_DQN.dat", 'w')
    # f.write(str("Time   "))
    # f.write(str("Accuracy   "))
    # f.write(str("LastHitFrame   "))
    # f.write(str("\n"))

    env = gym.make('DeepCars-v0')
    state_size = env.ObservationSpace()
    action_size = env.ActionSpace()
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    batch_size = 32

    state = env.reset()
    t0 = time.time()

    SaveCounter = 1
    episode_rewards = [0.0]
    n_eps_mean = [0.0]
    totalHitCars = 0
    totalPassedCars = 0
    dict = {'step': [], 'episode reward': [], 'accuracy': [], '100 eps mean': []}
    df = pd.DataFrame(dict)
    for agent.steps in range(MAX_STEPS):

        action = agent.act(state)
        next_state, reward, done, HitCarsCount, PassedCarsCount = env.step(action, True)
        # env.render()
        agent.remember(state, action, reward, next_state, done) # save to memory
        state = next_state
        episode_rewards[-1] += reward

        if done:
            state = env.reset()
            episode_rewards.append(0.0)
            totalHitCars += HitCarsCount
            totalPassedCars += PassedCarsCount
            accuracy = round(totalPassedCars / (totalPassedCars + totalHitCars) * 100, 2)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if agent.steps % SAVE_FREQ == 0:
                agent.save("./Save/ARC_AVL_DQN_{}.h5".format(SaveCounter))
                print('********************* model is saved: ./Save/ARC_AVL_DQN_{}.h5*****************'.format(SaveCounter))
                SaveCounter += 1

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)

        if done and num_episodes % PRINT_FREQ == 0:
            t1 = round(time.time() - t0, 2)     # time (s) spend since the start of training
            print("eps: ", num_episodes,
                  "   step: ", agent.steps,
                  "   time (s): ", "%.2f" % t1,
                  "   accuracy ", "%.2f" % accuracy, "%"
                  "   eps rew: ", "%d" % episode_rewards[-2],
                  "   mean 100 eps rew: ", mean_100ep_reward)
            # Save log file:
            df = df.append({'step': agent.steps, 'episode reward': episode_rewards[-2], \
                            'accuracy': accuracy, '100 eps mean': mean_100ep_reward, \
                            'time': "%2f" % t1}, ignore_index=True)
            df.to_csv('./Save/Training_Log_DQN.csv')
            # f.write(str(t1))
            # f.write(str("     "))
            # f.write(str(accuracy))
            # f.write(str("     "))
            # f.write(str(totalHitCars))
            # f.write(str("\n"))

    agent.save("./Save/ARC_AVL_DQN.h5")
    print("The training is finished. Last model is saved in /Save/ARC_AVL_DQN.h5")
    print("Hit cars: ", totalHitCars)
    print("Passed cars: ", totalPassedCars)
    print("Accuracy ", accuracy, "%")
    print("Use python Test_DeepCars_DQN.py to test the agent")
