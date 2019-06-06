from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


def huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)


learning_rate = 0.001

# Neural Net for Deep-Q learning Model
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss=huber_loss,
              optimizer=Adam(lr=learning_rate))

model.save_weights('./test_model.h5')
print('model is saved')




