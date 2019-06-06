
import random, os, time
import numpy as np
from DeepCars import GridWorld as envObj
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
import gym, gym_deepcars

MAX_STEPS = 1000000
SAVE_FREQ = 10000
TARGET_UPDATE_FREQUENCY = 100
EPOCHES = 1
PRINT_FREQ = 1

import datetime
temp_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir('./Save/{}'.format(temp_file_name))
logger_dir = './Save/' + temp_file_name

# Huber loss is used for for the error clipping, as discussed in DeepMind human-level paper
# The idea taken from https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
# And codes from https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

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
        # Save model summary to file
        from contextlib import redirect_stdout
        with open(logger_dir + '/modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.steps % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_model()
            # print("*******************************Target Model Updated*******************************")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)  # Q values for each action
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        LossSum = 0
        ValueSum = 0
        for state, action, reward, next_state, done in minibatch:
            Q_target = self.model.predict(state)
            ValueSum += Q_target.max()
            a = self.model.predict(next_state)[0]
            t = self.target_model.predict(next_state)[0]
            if done:
                Q_Est = reward
            else:
                Q_Est = reward + self.gamma * t[np.argmax(a)]
            LossSum += (Q_target[0][action] - Q_Est) ** 2
            Q_target[0][action] = Q_Est
            self.model.fit(state, Q_target, epochs=EPOCHES, verbose=0)
        LossAvr = LossSum / batch_size
        ValueAvr = ValueSum / batch_size
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return LossAvr, ValueAvr

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
    agent = DDQNAgent(state_size, action_size)
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
                agent.save(logger_dir + "/ARC_AVL_DDQN_{}.h5".format(SaveCounter))
                print('********************* model is saved: .../ARC_AVL_DDQN_{}.h5*****************'.format(SaveCounter))
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
            df.to_csv(logger_dir + '/Training_Log_DDQN.csv')
            # f.write(str(t1))
            # f.write(str("     "))
            # f.write(str(accuracy))
            # f.write(str("     "))
            # f.write(str(totalHitCars))
            # f.write(str("\n"))

    agent.save(logger_dir + "/ARC_AVL_DDQN.h5")
    print("The training is finished. Last model is saved in {}/ARC_AVL_DDQN.h5".format(logger_dir))
    print("Hit cars: ", totalHitCars)
    print("Passed cars: ", totalPassedCars)
    print("Accuracy ", accuracy, "%")
    print("Use python Test_DeepCars_DDQN.py to test the agent")