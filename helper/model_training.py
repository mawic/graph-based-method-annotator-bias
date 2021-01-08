import pandas as pd
import numpy as numpy
from tqdm import tqdm
import os, re, csv, math, codecs


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
#from tf.keras.optimizers import adam
#from tf.keras import backend as K
#from tf.keras import regularizers
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense, Activation, Dropout, Flatten
#from tf.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
#from tf.keras.utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
#from tf.keras.callbacks import EarlyStopping

from helper.scoring import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def train_model(df_train,df_test_multiple,params):
    # get data as arrays
    train_text = np.asarray(df_train['comment'].tolist())
    train_label = np.asarray(df_train['attack'].tolist())

    test_text_multiple = []
    test_label_multiple = []
    for test_set in df_test_multiple:
        test_text_multiple.append(np.asarray(test_set['comment'].tolist()))
        test_label_multiple.append(np.asarray(test_set['attack'].tolist()))

    # define vocab size
    vocab_size = 0
    vocab = {}
    for text in train_text:
        for item in text:
            vocab[item] = 1
    if len(vocab) > params['max_vocab_size']:
        vocab_size = params['max_vocab_size']
    else:
        vocab_size = len(vocab)

    # tokenize
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=params['oov_tok'])
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = sequence.pad_sequences(train_sequences, maxlen=params['max_seq_len'], padding=params['padding_type'], truncating=params['trunc_type'])

    test_padded_multiple = []
    for test_set in test_text_multiple:
        test_sequences = tokenizer.texts_to_sequences(test_set)
        test_padded_multiple.append(sequence.pad_sequences(test_sequences, maxlen=params['max_seq_len'], padding=params['padding_type'], truncating=params['trunc_type']))

    # setup model
    model = keras.Sequential([
        layers.Embedding(vocab_size, params['embed_dim'], mask_zero=True),
        layers.Bidirectional(layers.LSTM(params['lstm_dim'])),
        layers.Dense(16, activation=keras.activations.relu),
        layers.Dense(1, activation=keras.activations.sigmoid)
    ])
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

    # early stopping callback for training
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
    
    # train
    hist = model.fit(train_padded, train_label, batch_size=params['batch_size'], epochs=params['num_epochs'], validation_split=0.1, shuffle=True, verbose=1,callbacks=[earlystopping])

    # calculate scores
    evaluation_scores = []
    class_scores = []
    for i in range(0,len(test_text_multiple)):
        eval_text_classes,eval_text_overall,eval_text_confmatrix = getEvaluationResults(test_label_multiple[i],model.predict(test_padded_multiple[i]), labels=['0','1'])
        evaluation_scores.append(eval_text_overall)
        class_scores.append(eval_text_classes)
    
    return (model,evaluation_scores,class_scores,tokenizer)

def load_embeddings(path_embedding):
    # load embedding
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(path_embedding, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))    

def train_model_w_embedding(df_train,df_test_multiple,params,embeddings_index):
   
    # get data as arrays
    train_text = np.asarray(df_train['comment'].tolist())
    train_label = np.asarray(df_train['attack'].tolist())

    test_text_multiple = []
    test_label_multiple = []
    for test_set in df_test_multiple:
        test_text_multiple.append(np.asarray(test_set['comment'].tolist()))
        test_label_multiple.append(np.asarray(test_set['attack'].tolist()))

    # define vocab size
    vocab_size = 0
    vocab = {}
    for text in train_text:
        for item in text:
            vocab[item] = 1
    if len(vocab) > params['max_vocab_size']:
        vocab_size = params['max_vocab_size']
    else:
        vocab_size = len(vocab)

    # tokenize
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=params['oov_tok'])
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = sequence.pad_sequences(train_sequences, maxlen=params['max_seq_len'], padding=params['padding_type'], truncating=params['trunc_type'])

    test_padded_multiple = []
    for test_set in test_text_multiple:
        test_sequences = tokenizer.texts_to_sequences(test_set)
        test_padded_multiple.append(sequence.pad_sequences(test_sequences, maxlen=params['max_seq_len'], padding=params['padding_type'], truncating=params['trunc_type']))

        
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(['max_vocab_size'], len(word_index))
    embedding_matrix = np.zeros((nb_words, params['embed_dim']))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))    
        
        
        
    # setup model
    model = keras.Sequential([
        layers.Embedding(nb_words, params['embed_dim'],
          weights=[embedding_matrix], input_length=params['max_seq_len'], trainable=False),
        layers.Bidirectional(layers.LSTM(params['lstm_dim'])),
        layers.Dense(16, activation=keras.activations.relu),
        layers.Dense(1, activation=keras.activations.sigmoid)
    ])
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

    # train
    hist = model.fit(train_padded, train_label, batch_size=params['batch_size'], epochs=params['num_epochs'], validation_split=0.1, shuffle=True, verbose=2)

    # calculate scores
    evaluation_scores = []
    class_scores = []
    for i in range(0,len(test_text_multiple)):
        eval_text_classes,eval_text_overall,eval_text_confmatrix = getEvaluationResults(test_label_multiple[i],model.predict(test_padded_multiple[i]), labels=['0','1'])
        evaluation_scores.append(eval_text_overall)
        class_scores.append(eval_text_classes)
    
    return (model,evaluation_scores,class_scores,tokenizer)
