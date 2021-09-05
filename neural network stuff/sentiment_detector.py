import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import random

batch_size = 1000
embedding_size = 256
rnn_units = 1024
learning_rate = 1e-3

data = tfds.load("sentiment140", batch_size=-1, as_supervised=True)

x, y = data['train']
xtest, ytest = data['test']
y /= 4
ytest /= 4

tf.print(x.shape,xtest.shape)


vectorizer = TextVectorization(max_tokens=5000)
vectorizer.adapt(x[:10000])

model = Sequential([
    vectorizer,
    Embedding(batch_size, embedding_size),
    LSTM(rnn_units),
    Dense(1, activation="sigmoid")
])


def mod_mse(y_true, y_pred):
    return tf.reduce_mean((y_true-y_pred)**2/(1-(y_true-y_pred)**2))

def mod_acc(y_true,y_pred):
    return tf.reduce_mean(1-(y_true-y_pred)**2)


model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=mod_mse, metrics=[mod_acc])

model.fit(x, y, batch_size=batch_size, epochs=5, shuffle=True, callbacks=[
          tf.keras.callbacks.TensorBoard(log_dir="tf_logs/fit/"+datetime.now().strftime("%Y%m%d-%H%M%S"), update_freq=10)])

model.save("sentiment_model")

model.evaluate(xtest, ytest, batch_size=batch_size)

for i in range(10):
    ind = random.randint(0, len(xtest)-1)
    tf.print(xtest[ind], ytest[ind], model(tf.convert_to_tensor([xtest[ind]])))
