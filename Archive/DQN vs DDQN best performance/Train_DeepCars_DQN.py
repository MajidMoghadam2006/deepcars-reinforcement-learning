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

MAX_STEPS = 100000
SAVE_FREQ = 5000

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
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self.huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)          # Q values for each action
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
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
    os.environ['SDL_AUDIODRIVER'] = "dummy"  # Create a AUDIO DRIVER to not produce the pygame sound
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Create a dummy window to not show the pygame window

    # open text file to save information
    f = open("Save/Data_DQN.dat", 'w')
    f.write(str("Time   "))
    f.write(str("Accuracy   "))
    f.write(str("LastHitFrame   "))
    f.write(str("\n"))

    env = envObj()
    env.PygameInitialize()
    state_size = env.ObservationSpace()
    action_size = env.ActionSpace()
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    batch_size = 32

    state = env.Reset()
    t0 = time.time()
    HitCarsCount_ = 0
    HitFrmCounter = 0

    SaveCounter = 1
    episode_rewards = [0.0, 0.0]
    episode_steps = []
    Accuracies = []
    for agent.steps in range(MAX_STEPS):

        action = agent.act(state)
        next_state, reward, IsTerminated, HitCarsCount, PassedCarsCount, done = env.update(action, True)
        agent.remember(state, action, reward, next_state)

        episode_rewards[-1] += reward
        if HitCarsCount > HitCarsCount_:
            HitFrmCounter = 0
        else:
            HitFrmCounter += 1
        HitCarsCount_ = HitCarsCount
        Accuracy = round(PassedCarsCount / (PassedCarsCount + HitCarsCount) * 100, 2)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if agent.steps % SAVE_FREQ == 0:
                agent.save("./Save/ARC_AVL_DQN_{}.h5".format(SaveCounter))
                print('********************* model is saved: ./Save/ARC_AVL_DQN_{}.h5*****************'.format(SaveCounter))
                SaveCounter += 1

        t1 = round(time.time() - t0, 2)
        print("Step: ", agent.steps,
              "   Time (s): ", "%.2f" % t1,
              "   Accuracy ", "%.2f" % Accuracy, "%",
              "   No hit frames: ", HitFrmCounter,
              "   Episode reward: ", episode_rewards[-2])
        f.write(str(t1))
        f.write(str("     "))
        f.write(str(Accuracy))
        f.write(str("     "))
        f.write(str(HitFrmCounter))
        f.write(str("\n"))

        if done:
            episode_steps.append(agent.steps)
            episode_rewards.append(0.0)
            Accuracies.append(Accuracy)
        state = next_state

        if IsTerminated:
            print("Training is terminated manually")
            break

    for _ in range(len(episode_rewards)-len(Accuracies)):
        del episode_rewards[0]  # Remove first elements of reward vector as initialized to zero

    agent.save("./Save/ARC_AVL_DQN.h5")
    print("The training is finished. Last model is saved in /Save/ARC_AVL_DQN.h5")
    print("Hit cars: ", HitCarsCount)
    print("Passed cars: ", PassedCarsCount)
    print("Accuracy ", round(PassedCarsCount / (PassedCarsCount + HitCarsCount) * 100, 2), "%")
    print("Use python Test_DeepCars_DQN.py to test the agent")
    # Save log file:
    dict = {'episode steps': episode_steps, 'episode reward': episode_rewards, 'accuracy': Accuracies}
    df = pd.DataFrame(dict)
    df.to_csv('./Save/Training_Log_DQN.csv')