import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

file_path = 'poems.csv'

# Paso 1: Preprocesamiento de Datos
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.index[100:5133])
    nRow, nCol = data.shape
    print(f'There are {nRow} rows and {nCol} columns')
    text = data['content'].str.cat(sep='\n')
    return text

def preprocess_text(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Crear secuencias de n-gramas
    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Crear predictores y etiquetas
    xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return xs, ys, max_sequence_len, total_words, tokenizer

# Paso 2: Modelo de RNN con LSTM
def create_model(max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 200))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation='softmax'))

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.002), metrics=['accuracy'])
    model.summary()

    return model

text = load_data(file_path)
xs, ys, max_sequence_len, total_words, tokenizer = preprocess_text(text)

# Ejecución del modelo
""" 
model = create_model(max_sequence_len, total_words)
model.fit(xs, ys, epochs=250, batch_size=128, verbose=1)

model.save('poem_generator.keras') """

model = load_model('poem_generator.keras')

# Paso 3: Generación de Texto
def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + output_word
    return seed_text

# Generar un poema
seed_text = input("\nIngrese frase semilla para el poema: ")
next_words = 10
generated_poem = generate_text(seed_text, next_words, max_sequence_len, model, tokenizer)
print("\nPoema generado: \n", generated_poem)