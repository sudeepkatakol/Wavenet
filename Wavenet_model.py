from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv1D, Activation, Add, Multiply
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
import numpy as np

X = np.load('Piano_half_hour_16kHz_10s_samples.npy')

dilation_channels = 64
causal_channels = 32
dilation_kernel = 2

def residual_block(X, dilation_rate):
    F = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    G = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    F = Activation('tanh')(F)
    G = Activation('sigmoid')(G)
    Y = Multiply()([F, G])
    return Y

causal_kernel = 2
causal_channels = 32
dilation_rate = 2
quantization_channels = 256
def model(input_shape):
    X_input = Input(input_shape)
    X = Conv1D(causal_channels, causal_kernel, padding='causal', dilation_rate=1)(X_input)
    Y = residual_block(X, 1)
    S0 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 2)
    S1 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 4)
    S2 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 8)
    S3 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 16)
    S4 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 32)
    S5 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 64)
    S6 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 128)
    S7 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 256)
    S8 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 512)
    S9 = Conv1D(dilation_channels, 1, padding="same")(Y)
    S = Add()([S0, S1, S2, S3, S4, S5, S6, S7, S8, S9])
    X = Activation('relu')(S)
    X = Conv1D(128, 1, padding="same")(X)
    X = Activation('relu')(X)
    X = Conv1D(256, 1, padding="same")(X)
    X = Activation('softmax')(X)
    return Model(inputs = X_input, outputs = X)
    
net = model((160124, 256))
net.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

X_ = X[:2]
X_ = to_categorical(X_)
Y_ = np.concatenate([X_[:, 1:, :], X_[:, -1, :].reshape(X_.shape[0], 1, X_.shape[2])], axis = 1)

net.fit(x = X_, y = Y_, epochs = 1, batch_size = 2)

model.save('test_model_0.h5')
