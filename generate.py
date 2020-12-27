import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import os
import datetime
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
path = os.getcwd()

text = open(path+'/data/input.txt', 'rb').read().decode(encoding='utf-8')
print("Text is {} characters long".format(len(text)))
words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print("Text is {} words long".format(len(words)))
vocab = sorted(set(text))
#print ('There are {} unique characters'.format(len(vocab)))
char2int = {c:i for i, c in enumerate(vocab)}
int2char = np.array(vocab)
#print('Vector:\n')
text_as_int = np.array([char2int[ch] for ch in text], dtype=np.int32)
#print ('{}\n mapped to integers:\n {}'.format(repr(text[:100]), text_as_int[:100]))
tr_text = text_as_int[:704000] #text separated for training, divisible by the batch size (64)
val_text = text_as_int[704000:] #text separated for validation
#print(text_as_int.shape, tr_text.shape, val_text.shape)


# Populate the library of tunables - I like keeping them centralized in case I need to change things around:
batch_size = 64
buffer_size = 10000
embedding_dim = 256
epochs = 50
seq_length = 200
examples_per_epoch = len(text)//seq_length
#lr = 0.001 #will use default for Adam optimizer
rnn_units = 1024
vocab_size = len(vocab)
tr_char_dataset = tf.data.Dataset.from_tensor_slices(tr_text)
val_char_dataset = tf.data.Dataset.from_tensor_slices(val_text)
print(tr_char_dataset, val_char_dataset)
tr_sequences = tr_char_dataset.batch(seq_length+1, drop_remainder=True)
val_sequences = val_char_dataset.batch(seq_length+1, drop_remainder=True)
print(tr_sequences, val_sequences)

for item in tr_sequences.take(1):
    print(repr(''.join(int2char[item.numpy()])))
    print(item)
for item in val_sequences.take(1):
    print(repr(''.join(int2char[item.numpy()])))
    print(item)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

tr_dataset = tr_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
val_dataset = val_sequences.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
print(tr_dataset, val_dataset)
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)

for input_example_batch, target_example_batch in tr_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "respectively: batch_size, sequence_length, vocab_size")
model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print("Input: \n", repr("".join(int2char[input_example_batch[0]])))
print()
print("Predictions: \n", repr("".join(int2char[sampled_indices ])))
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
def accuracy(labels, logits):
    return tf.keras.metrics.sparse_categorical_accuracy(labels, logits)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
example_batch_acc  = accuracy(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Loss:      ", example_batch_loss.numpy().mean())
print("Accuracy:      ", example_batch_acc.numpy().mean())


optimizer = tf.keras.optimizers.Adam() 
model.compile(optimizer=optimizer, loss=loss)
patience = 10
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
checkpoint_dir = (path+'/checkpoints')+ datetime.datetime.now().strftime("_%Y.%m.%d-%H:%M:%S")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


history = model.fit(tr_dataset, epochs=epochs , validation_data=val_dataset)
print ("Training stopped as there was no improvement after {} epochs".format(patience))
import matplotlib.pyplot as plt

plt.figure(figsize=(12,9))
plt.plot(history.history['loss'], 'g')
plt.plot(history.history['val_loss'], 'rx') #use if have val data
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.legend(['Train', 'Validation'], loc='upper right') #use if have val date
plt.show()
