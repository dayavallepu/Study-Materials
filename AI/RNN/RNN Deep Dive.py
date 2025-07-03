#word level one-hot encoding
import numpy as np # for numerical calculations
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}
for sample in samples: # for each sample in the list of samples
    for word in sample.split(): # for each word in the sample
        if word not in token_index: # if the word is not already in the token index
            token_index[word] = len(token_index) + 1 # add the word to the token index with a unique index
max_length = 10 # maximum length of the sequences
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1)) # it ensures 
for i, sample in enumerate(samples): # for each sample in the list of samples
    for j, word in list(enumerate(sample.split()))[:max_length]: # for each word in the sample split, up to the maximum length of the sequences
        index = token_index.get(word) # get the index of the word from the token index
        results[i, j, index] = 1 # set the value at the index to 1 in the results 
        
#Character level one-hot encoding

import string # for string operations
samples = ['The cat sat on the mat.', 'The dog ate my homework.'] # list of samples
characters = string.printable # get all printable characters
token_index = dict(zip(range(1, len(characters) + 1), characters)) # create a dictionary of characters with their corresponding indexes
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1)) # create a numpy array of zeros with shape (number of samples, maximum length, maximum index + 1)
for i, sample in enumerate(samples): # for each sample in the list of samples
    for j, character in enumerate(sample): # for each character in the sample
        index = token_index.get(character) # get the index of the character from the token index
        results[i, j, index] = 1 # set the value at the index to 1 in the results
        
#In-built Keras code for word level one-hot encoding
from tensorflow.keras.preprocessing.text import Tokenizer # for text preprocessing
samples = ['The cat sat on the mat.', 'The dog ate my homework.'] # list of samples
tokenizer = Tokenizer(num_words=1000) # create a tokenizer object with a maximum of 1000 words
tokenizer.fit_on_texts(samples) # fit the tokenizer on the samples
sequences = tokenizer.texts_to_sequences(samples) # converting the samples to sequences of integers
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') # converting the samples to one-hot encoded which os in the range of 0 to 1
word_index = tokenizer.word_index # get the word index from the tokenizer
print('Found %s unique tokens.' % len(word_index)) # print the number of unique tokens found in the samples

#Work-level one-hot encoding with hashing trick
#mapping words to intergers effectively, it is mapped into fixed size vector
import numpy as np # for numerical calculation
samples = ['The cat sat on the mat.', 'The dog ate my homework.'] # list of samples
dimensionality = 1000 # fixed size of the vector
max_length = 10 # maximum length of the sequences
results = np.zeros((len(samples), max_length, dimensionality)) # it gives the zeros array of shape (number of samples, maximum length, dimensionality)
for i, sample in enumerate(samples): # for each sample in the list of samples, Enumerate gives the index and the sample
    for j, word in list(enumerate(sample.split()))[:max_length]: # for each word in the sample split, up to the maximum length of the sequences
        index = abs(hash(word)) % dimensionality # get the index of the word from the token index
        results[i, j, index] = 1

# word embedding
from keras.layers import Embedding # importing the embedding layer from keras
embedding_layer = Embedding(1000, 64) # create an embedding layer with 1000 words and 64 dimensions

from keras.datasets import imdb # importing the IMDB dataset from keras
from keras import preprocessing # for preprocessing the data
max_features = 10000 # maximum number of words to consider
maxlen = 20 # maximum length of the sequences
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # load the IMDB dataset and split it into training and testing sets
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) # pad the sequences to the maximum length
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen) # pad the sequences to the maximum length

from keras.models import Sequential # importing the sequential model from keras
from keras.layers import Flatten, Dense # importing the flatten and dense layers from keras
model = Sequential() # create a sequential model
model.add(Embedding(10000, 8, input_length=maxlen)) # add an embedding layer with 10000 words, 8 dimensions and input length of maxlen dimensions means the input shape of the data
model.add(Flatten()) # add a flatten layer to flatten the input data. it converts the 2D input data into 1D data
model.add(Dense(1, activation='sigmoid')) # add a dense layer with 1 output and sigmoid activation function
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # compile the model with rmsprop optimizer, binary crossentropy loss and accuracy metric
model.summary() # print the summary of the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32,validation_split=0.2) # fit the model on the training data with 10 epochs, batch size of 32 and validation split of 0.2



#Processing the labels of the raw IMDB data
import os 
imdb_dir = 'C:/Users/Bharani Kumar/Desktop/AI/DL&AI_codes/Day03/Datasets/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
        

# Tokenizing the text of the raw IMDB data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


glove_dir = 'C:/Users/Bharani Kumar/Desktop/AI/DL&AI_codes/Day03/Datasets/glove'
embeddings_index = {}   
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

# Pseudocode RNN
state_t = 0
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    
    state_t = output_t
    
# Detailed Pseudocode RNN    
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
    
import numpy as np
timesteps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.concatenate(successive_outputs, axis=0)

from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()

from keras.datasets import imdb # importing the IMDB dataset from keras
from keras.preprocessing import sequence # for preprocessing the data
max_features = 10000 # maximum number of words to consider(any word more than 10000 times only consider)
maxlen = 500 # maximum length of the sequences to be considered
batch_size = 32 # batch size for training
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features) # load the IMDB dataset and split it into training and testing sets
print(len(input_train), 'train sequences') # print the number of training sequences
print(len(input_test), 'test sequences') # print the number of testing sequences
print('Pad sequences (samples x time)') # pad the sequences to the maximum length
input_train = sequence.pad_sequences(input_train, maxlen=maxlen) # pad the training sequences to the maximum length
input_test = sequence.pad_sequences(input_test, maxlen=maxlen) # pad the testing sequences to the maximum length
print('input_train shape:', input_train.shape) # print the shape of the training sequences
print('input_test shape:', input_test.shape) # print the shape of the testing sequences

from keras.layers import Dense # for dense layers
from keras.models import Sequential
model = Sequential() # create a sequential model
model.add(Embedding(max_features, 32)) # add an embedding layer with 10000 words and 32 dimensions, by using embedding layer converts dense representation
model.add(SimpleRNN(32)) # add a simple RNN layer with 32 units
model.add(Dense(1, activation='sigmoid')) # add a dense layer with 1 output and sigmoid activation function
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # compile the model with rmsprop optimizer, binary crossentropy loss and accuracy metric
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2) # train the model

import matplotlib.pyplot as plt # for plotting the graphs
acc = history.history['acc'] # get the accuracy of the model
val_acc = history.history['val_acc'] # get the validation accuracy of the model
loss = history.history['loss'] # get the loss of the model
val_loss = history.history['val_loss'] # get the validation loss of the model
epochs = range(1, len(acc) + 1) # create a range of epochs
plt.plot(epochs, acc, 'bo', label='Training acc') # plot the training accuracy
plt.plot(epochs, val_acc, 'b', label='Validation acc') # plot the validation accuracy
plt.title('Training and validation accuracy') # set the title of the plot
plt.legend() # add the legend to the plot
plt.figure() # create a new figure
plt.plot(epochs, loss, 'bo', label='Training loss') # plot the training loss
plt.plot(epochs, val_loss, 'b', label='Validation loss') # plot the validation loss
plt.title('Training and validation loss') # set the title of the plot
plt.legend() # add the legend to the plot
plt.show() # show the plot

from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense
max_features = 10000 # maximum number of words to consider(any word more than 10000 times only consider)
maxlen = 500 # maximum length of the sequences to be considered
batch_size = 32 # batch size for training

from keras.datasets import imdb # importing the IMDB dataset from keras
from keras.preprocessing import sequence# for preprocessing the data
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features) # load the IMDB dataset and split it into training and testing sets
print(len(input_train), 'train sequences') # print the number of training sequences
print(len(input_test), 'test sequences') # print the number of testing sequences
print('Pad sequences (samples x time)') # pad the sequences to the maximum length
input_train = sequence.pad_sequences(input_train, maxlen=maxlen) # pad the training sequences to the maximum length
input_test = sequence.pad_sequences(input_test, maxlen=maxlen) # pad the testing sequences to the maximum length
print('input_train shape:', input_train.shape) # print the shape of the training sequences
print('input_test shape:', input_test.shape) # print the shape of the testing sequences
model = Sequential() # intializing the sequential model
model.add(Embedding(max_features, 32)) # add an embedding layer with 10000 words and 32 dimensions, by using embedding layer converts dense representation
model.add(LSTM(32)) # adding 32 LSTM units
model.add(Dense(1, activation='sigmoid')) # add a dense layer with 1 output and sigmoid activation function
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # compile the model with rmsprop optimizer, binary crossentropy loss and accuracy metric
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2) # train the model with 10 epochs, batch size of 128 and validation split of 0.2

import matplotlib.pyplot as plt # for plotting the graphs
acc = history.history['acc'] # get the accuracy of the model
val_acc = history.history['val_acc'] # get the validation accuracy of the model
loss = history.history['loss'] # get the loss of the model
val_loss = history.history['val_loss'] # get the validation loss of the model
epochs = range(1, len(acc) + 1) # create a range of epochs
plt.plot(epochs, acc, 'bo', label='Training acc') # plot the training accuracy
plt.plot(epochs, val_acc, 'b', label='Validation acc') # plot the validation accuracy
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#Forecasting using LSTM
#Inspecting data
import os
import pandas as pd
data_dir = r'C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN\Datasets_RNN'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname) 
data = f.read() 
f.close()
data = pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN\Datasets_RNN\jena_climate_2009_2016.csv\jena_climate_2009_2016.csv")
lines = data.split('\n') 
header = lines[0].split(',') 
lines = lines[1:]
print(header)
print(len(lines))

import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]] 
    float_data[i, :] = values
        
names = ['Gopi','jagan', 'sid','sri']
list(enumerate(names))


from matplotlib import pyplot as plt
temp = float_data[:, 1] #temperature #(in degrees Celsius) 
plt.plot(range(len(temp)), temp)
plt.plot(range(1440), temp[:1440])

#Normalizing the data
mean = float_data[:200000].mean(axis=0) 
float_data -= mean
std = float_data[:200000].std(axis=0) 
float_data /= std

#Generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay
    i = min_index + lookback 
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index)) 
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step) 
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
# Preparing the training, validation and test generators
lookback = 1440 
step = 6
delay = 144 
batch_size = 128
train_gen = generator(float_data, lookback=lookback,delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback,delay=delay, min_index=200001, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback,delay=delay, min_index=300001, max_index=None, step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)
#Computing MAE & this takes time

def evaluate_naive_method(): 
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets)) 
        batch_maes.append(mae)
    print(np.mean(batch_maes)) 
evaluate_naive_method()   # getting error
# converting MAE to Celsius error
celsius_mae = 0.29 * std[1]
np.mean(np.abs(preds - targets))

# training & Evaluating a densly connected model
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1]))) 
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss'] 
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1) 
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.legend()
plt.show()

#GRU model
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1]))) 
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

#Dropout regularized GRU model
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential() 
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,
steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)


# STacking Recurrent Layers
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential() 
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5, return_sequences=True,input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1,recurrent_dropout=0.5)) 
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

# Training & Evaluating LSTM using reversed sequences
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
max_features = 10000 
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data( num_words=max_features)
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen) 
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential() 
model.add(layers.Embedding(max_features, 128)) 
model.add(layers.LSTM(32)) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10,batch_size=128, validation_split=0.2)

#Bidirectional LSTM
model = Sequential() 
model.add(layers.Embedding(max_features, 32)) 
model.add(layers.Bidirectional(layers.LSTM(32))) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
history = model.fit(x_train, y_train,epochs=10, batch_size=128, validation_split=0.2)

#Bidirectional GRU
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential() 
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1]))) 
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

#1D Convnet- Preparing IMDB data
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000 
max_len = 500
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) 
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len) 
x_test = sequence.pad_sequences(x_test, maxlen=max_len) 
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#Training and evaluating a simple 1D convnet on the IMDB data
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len)) 
model.add(layers.Conv1D(32, 7, activation='relu')) 
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu')) 
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy',metrics=['acc']) 
history = model.fit(x_train, y_train,epochs=10, batch_size=128, validation_split=0.2)

# CNN + RNN model
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',input_shape=(None, float_data.shape[-1]))) 
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu')) 
model.add(layers.MaxPooling1D(3)) 
model.add(layers.Conv1D(32, 5, activation='relu')) 
model.add(layers.GlobalMaxPooling1D()) 
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae') 
history = model.fit_generator(train_gen,steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)
#preparing higher-resolution data generators
step = 3 
lookback = 720 
delay = 144
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step)
val_steps = (300000 - 200001 - lookback) // 128 
test_steps = (len(float_data) - 300001 - lookback) // 128

#Combine 1D convolutional base and GRU layer
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',input_shape=(None, float_data.shape[-1]))) 
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu')) 
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5)) 
model.add(layers.Dense(1))
model.summary() 
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,epochs=20, validation_data=val_gen, validation_steps=val_steps)

