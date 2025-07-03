# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 05:09:16 2025

@author: lenny
"""

# =======================
# 1. Word-Level One-Hot Encoding (Manual)
# =======================
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

# =======================
# 2. Character-Level One-Hot Encoding (Manual)
# =======================
import string

characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1

# =======================
# 3. Word-Level One-Hot Encoding using Keras Tokenizer
# =======================
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# =======================
# 4. Word-Level One-Hot Encoding using Hashing Trick
# =======================
dimensionality = 1000
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1

# =======================
# 5. Word Embedding Example using Keras
# =======================
from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)

# =======================
# 6. IMDB Dataset Preprocessing with Word Embedding (Trainable Embeddings)
# =======================
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# =======================
# 7. GloVe Embedding Preparation and Integration with IMDB Raw Text
# =======================

"""
The folder structure of the files 

D:\Trainer\aclImdb\test\
   ├── neg\   --> contains negative reviews as `.txt` files
   └── pos\   --> contains positive reviews as `.txt` files
"""

import os  # Import the os module to work with file and folder paths

# Set the base directory where the IMDB dataset is located
imdb_dir = r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN\Datasets_RNN\aclImdb\aclImdb"

# Join the base path with the 'test' subfolder (since you are using test data)
test_dir = os.path.join(imdb_dir, 'test')  # This becomes D:\Trainer\aclImdb\test

# Initialize empty lists to store review texts and their corresponding labels
labels = []  # 0 for negative, 1 for positive
texts = []   # raw text content of each review

# Loop over the two sentiment folders: 'neg' and 'pos'
for label_type in ['neg', 'pos']:
    # Build the full path to the current label folder (e.g., D:\Trainer\aclImdb\test\neg)
    dir_name = os.path.join(test_dir, label_type)

    # Loop through each file name in the current directory (neg or pos)
    for fname in os.listdir(dir_name):
        # Skip hidden files like .DS_Store
        if not fname.startswith('.'):
            # Open the current text file using UTF-8 encoding
            with open(os.path.join(dir_name, fname), encoding='utf8') as f:
                # Read the content of the file and append it to the texts list
                texts.append(f.read())

            # Append the corresponding label: 0 for 'neg', 1 for 'pos'
            labels.append(0 if label_type == 'neg' else 1)



# =======================
# 8. Tokenize Texts and Create Embedding Matrix from GloVe
# =======================

# Import Tokenizer and padding tools from Keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Set the maximum length of padded sequences
maxlen = 100

# Number of training samples to use
training_samples = 200

# Number of validation samples to use
validation_samples = 10000

# Maximum number of words to consider based on word frequency
max_words = 10000

# Create a tokenizer that will only keep the top 'max_words' most frequent words
tokenizer = Tokenizer(num_words=max_words)

# Fit the tokenizer on the texts (tokenizes and builds word index)
tokenizer.fit_on_texts(texts)

# Convert texts into sequences of integers based on word index
sequences = tokenizer.texts_to_sequences(texts)

# Dictionary mapping words to their index (based on frequency)
word_index = tokenizer.word_index

# Pad sequences so that all sequences have the same length (maxlen)
data = pad_sequences(sequences, maxlen=maxlen)

# Convert labels to a NumPy array (ensures compatibility with Keras)
labels = np.asarray(labels)

# Generate an array of indices equal to the number of data samples
indices = np.arange(data.shape[0])

# Shuffle the indices randomly
np.random.shuffle(indices)

# Shuffle the data and labels using the shuffled indices
data = data[indices]
labels = labels[indices]

# Prepare training data: first 'training_samples' samples
x_train = data[:training_samples]
y_train = labels[:training_samples]

# Prepare validation data: next 'validation_samples' samples
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]



import numpy as np  # Import NumPy for numerical operations
import os
# Set the directory where your GloVe embedding file is located
glove_dir = r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN\Datasets_RNN\glove.6B"

# Initialize an empty dictionary to store word-to-vector mappings
embeddings_index = {}

# Open the GloVe embeddings file (100-dimensional vectors)
with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8') as f:
    # Read each line in the file (each line corresponds to one word and its embedding)
    for line in f:
        values = line.split()              # Split the line into word and numbers
        word = values[0]                   # First token is the actual word
        coefs = np.asarray(values[1:], dtype='float32')  # Convert the rest into a NumPy array (the vector)
        embeddings_index[word] = coefs     # Store word -> vector mapping in the dictionary

# Set the embedding dimension (should match the one from GloVe file, which is 100 here)
embedding_dim = 100

# Initialize a matrix to hold the embedding vectors for each word in your tokenizer's word_index
# Shape: (max_words, 100), initialized with zeros
embedding_matrix = np.zeros((max_words, embedding_dim))

# Loop through every word and its index in your tokenizer's vocabulary
for word, i in word_index.items():
    if i < max_words:  # Only include words within the max_words limit
        embedding_vector = embeddings_index.get(word)  # Get the GloVe vector for this word
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Assign the vector to the corresponding row in the matrix


# =======================
# 9. Build and Train Model using Pre-trained GloVe Embeddings
# =======================
from keras.layers import Embedding # Import Embedding layer
from keras import preprocessing # Import preprocessing tools
from keras.models import Sequential # Import Sequential model
from keras.layers import Flatten, Dense # Import layers for the model
import os 
os.chdir(r'C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN')
model = Sequential() # Initialize a Sequential model
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [embedding_matrix])) # Add an Embedding layer with pre-trained weights
model.add(Flatten()) # Flatten the output from the Embedding layer
model.add(Dense(32, activation='relu')) # Add a Dense layer with ReLU activation
model.add(Dense(1, activation='sigmoid')) # Add a Dense output layer with sigmoid activation for binary classification
model.layers[0].set_weights([embedding_matrix]) # Set the weights of the Embedding layer to the pre-trained GloVe embeddings
model.layers[0].trainable = False # Freeze the Embedding layer to prevent it from being updated during training

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) # Compile the model with RMSprop optimizer and binary crossentropy loss
model.summary() # Print the model summary to see the architecture
# Train the model on the training data for 10 epochs with a batch size of 32 and validate on the validation data
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.weights.h5') # Save the trained model weights to a file

# =======================
# 10. Plotting Accuracy and Loss
# =======================
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

# =======================
# 11. Simple RNN from Scratch (Manual Implementation)
# =======================
import numpy as np  # Import NumPy for numerical operations

timesteps = 100 # Number of time steps in the input sequence
input_features = 32  # Number of features in the input data
output_features = 64 # Number of features in the output data

inputs = np.random.random((timesteps, input_features)) # Generate random input data
state_t = np.zeros((output_features,)) # Initialize the hidden state to zeros

W = np.random.random((output_features, input_features)) # Random weight matrix for input to hidden state transformation
U = np.random.random((output_features, output_features)) # Random weight matrix for hidden state to hidden state transformation
b = np.random.random((output_features,)) # Random bias vector

successive_outputs = [] # List to store the outputs at each time step
# Iterate over each time step in the input sequence
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) # Compute the output at the current time step using the tanh activation function
    successive_outputs.append(output_t) # Append the output to the list
    state_t = output_t # Update the hidden state for the next time step
# Stack the outputs into a 3D array
final_output_sequence = np.stack(successive_outputs, axis=0)

# =======================
# 12. Simple RNN using Keras
# =======================
from keras.models import Sequential # Import Sequential model from Keras
from keras.layers import Embedding, SimpleRNN # Import Embedding and SimpleRNN layers

model = Sequential() # Initialize a Sequential model
model.add(Embedding(10000, 32)) # Add an Embedding layer with 10,000 words and 32-dimensional embeddings
model.add(SimpleRNN(32)) # Add a SimpleRNN layer with 32 units
model.summary() # Print the model summary to see the architecture

# =======================
# 13. Deep Stacked RNN
# =======================
model = Sequential() # Initialize a Sequential model
model.add(Embedding(10000, 32)) # Add an Embedding layer with 10,000 words and 32-dimensional embeddings
model.add(SimpleRNN(32, return_sequences=True)) # Add a SimpleRNN layer with 32 units and return sequences for the next RNN layer
model.add(SimpleRNN(32, return_sequences=True)) # Add another SimpleRNN layer with 32 units and return sequences for the next RNN layer
model.add(SimpleRNN(32, return_sequences=True)) # Add another SimpleRNN layer with 32 units and return sequences for the next RNN layer
model.add(SimpleRNN(32)) # Add another SimpleRNN layer with 32 units and do not return sequences
model.summary() # Print the model summary to see the architecture

# =======================
# 14. Simple RNN on IMDB Dataset
# =======================
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000 # size of the vocabulary
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(f'{len(input_train)} train sequences')
print(f'{len(input_test)} test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# Build & train the RNN model
from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# =======================
# 15. Plot RNN Performance
# =======================
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

# =======================
# 16. LSTM on IMDB Dataset
# =======================
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# =======================
# 17. Plot LSTM Performance
# =======================
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
# =======================
# 18. GRU on IMDB Dataset
# =======================
from keras.layers import GRU

model = Sequential()
model.add(Embedding(max_features, 32)) #32 = the dimension of the dense embedding vectors 
model.add(GRU(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# =======================
# 19. Dropout Regularized GRU
# =======================
model = Sequential()
model.add(Embedding(max_features, 32))
# dropout=0.2 = During training, it randomly drops 20% of input connections (helps prevent overfitting).
# recurrent_dropout=0.2 = It also randomly drops 20% of recurrent (memory) connections inside the GRU (again to prevent overfitting).
model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# =======================
# 20. Bidirectional LSTM
# =======================
from keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# =======================
# 21. Bidirectional GRU
# =======================
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(Bidirectional(GRU(32)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# =======================
# 22. Reversed Sequence LSTM
# =======================
'''
basically we give the reverse input to the lstm, so we get better performance we can get underline patterns well.
'''
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = [x[::-1] for x in x_train] # reverse the inputs 
x_test = [x[::-1] for x in x_test] # reverse the inputs

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# =======================
# 23. Plot Any Recent Model (Generic Template)
# =======================
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

# =======================
# 24. Load and Explore Jena Climate Dataset
# =======================
import os # Import os module for file path operations
# Set the directory where your dataset is located
data_dir = r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\study materials@360\AI\6. RNN\Datasets_RNN\jena_climate_2009_2016.csv"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv') # Join the directory and file name to create the full path
# Read the CSV file and store its content in a variable
with open(fname) as f:
    data = f.read()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print('Header:', header)
print('Total lines:', len(lines))

# =======================
# 25. Convert to Numpy Array
# =======================
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# =======================
# 26. Normalize the Data
# =======================
# for equal importance we do standardization
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# =======================
# 27. Generator Function for Time Series Data
# =======================
# it is going to create sample of data
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    '''
    - data: NumPy array of time series data.
    - lookback: Number of previous time steps to use as input features.
    - delay: Number of time steps to predict into the future.
    - min_index: Minimum index in the data to consider.
    - max_index: Maximum index in the data to consider.
    - shuffle: Whether to shuffle the samples or draw them sequentially.
    - batch_size: Number of samples per batch.
    - step: Period between successive data points in the input sequences. every 6 we take the data 
    '''
    # If max_index is not specified, set it to the end of the data minus delay
    if max_index is None:
        max_index = len(data) - delay
    # Initialize the current index pointer
    i = min_index + lookback
    # Infinite loop: generator keeps producing batches
    while True:
        if shuffle:
            # Randomly pick 'batch_size' number of row indices between min and max 
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # Sequential mode: if we reach the end, restart from beginning
            if i + batch_size >= max_index:
                i = min_index + lookback
            # Create a sequence of row indices for the batch
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows) # Move the index pointer forward by batch size
        # Initialize empty arrays for input sequences (samples) and target values (targets) will shows 
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        # Loop through each row index to build samples and targets
        for j, row in enumerate(rows):
            # Select indices for the input sequence: going back 'lookback' steps, taking every 'step'-th point
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices] # Fill the j-th sample with the selected data points
            targets[j] = data[rows[j] + delay][1] # Set the j-th target: value at 'row + delay' for feature index 1
        yield samples, targets # Yield the batch of samples and their corresponding targets

# =======================
# 28. Define Training, Validation, and Test Generators
# =======================
lookback = 1440  # 10 days
step = 6         # Hourly data
delay = 144      # Predict 1 day ahead
batch_size = 128

train_gen = generator(float_data, lookback, delay, 0, 200000, True, batch_size, step)
val_gen = generator(float_data, lookback, delay, 200001, 300000, False, batch_size, step)
test_gen = generator(float_data, lookback, delay, 300001, None, False, batch_size, step)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

# =======================
# 29. Evaluate Naive Baseline (last observed temp as prediction)
# =======================
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)

naive_mae = evaluate_naive_method()
print("Naive baseline MAE:", naive_mae)
celsius_mae = naive_mae * std[1]
print("MAE in Celsius:", celsius_mae)

# =======================
# 30. Dense Model
# =======================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

# =======================
# 31. Plot Dense Model Loss
# =======================
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Dense Model: Training and Validation Loss')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Evaluate the model on validation data
val_predictions = []
val_targets = []

for _ in range(val_steps):
    x_batch, y_batch = next(val_gen)
    preds = model.predict(x_batch)
    val_predictions.extend(preds.flatten())
    val_targets.extend(y_batch.flatten())

val_predictions = np.array(val_predictions)
val_targets = np.array(val_targets)

# Calculate additional metrics
mae = np.mean(np.abs(val_predictions - val_targets))
mse = mean_squared_error(val_targets, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(val_targets, val_predictions)

print(f"Validation MAE:  {mae:.4f}")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²:   {r2:.4f}")


# =======================
# 32. GRU Model for Time Series Forecasting
# =======================
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

# =======================
# 33. Dropout Regularized GRU Model
# =======================
model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

# =======================
# 34. Stacked GRU Layers
# =======================
model = Sequential()
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
                     return_sequences=True, input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)


# =======================
# 35. 1D Convolutional Model for Time Series Forecasting
# =======================
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

# =======================
# 36. CNN + RNN Hybrid Model (Conv1D + GRU)
# =======================
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

# =======================
# 37. Higher-Resolution Data Preparation for Time Series
# =======================
step = 3
lookback = 720
delay = 144

train_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=0, max_index=200000, shuffle=True, step=step)
val_gen = generator(float_data, lookback=lookback, delay=delay,
                    min_index=200001, max_index=300000, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay,
                     min_index=300001, max_index=None, step=step)

val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

# =======================
# 38. Final CNN + GRU Model with High-Resolution Data
# =======================
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen, steps_per_epoch=500, epochs=10,
                    validation_data=val_gen, validation_steps=val_steps)

