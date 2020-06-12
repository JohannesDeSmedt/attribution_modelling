import numpy as np

from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation, Conv1D
from keras.layers import TimeDistributed, concatenate, Flatten
from keras.layers import Embedding, Add, Multiply, Dropout, Reshape
from keras.optimizers import Nadam
from datetime import datetime


def nn_hits_model(hit_cutoff):
    # Input embeddings
    input_layer_e = Input(shape=(cutoff,), name='event_input')
    embedding_event = Embedding(input_dim=no_channels, output_dim=embed_dim, input_length=cutoff, name='sequence_embedding')(input_layer_e)

    input_layer_d = Input(shape=(cutoff,), name='device_input')
    embedding_device = Embedding(input_dim=no_devices, output_dim=embed_dim, input_length=cutoff, name='device_embedding')(input_layer_d)

    input_layer_vn = Input(shape=(cutoff,), name='visitno_input')
    visitno_transformed = Reshape([cutoff, 1])(input_layer_vn)

    input_layer_tos = Input(shape=(cutoff,), name='tos_input')
    tos_transformed = Reshape([cutoff, 1])(input_layer_tos)

    # Convolution for web visit information
    hits_input_layer = Input(shape=(cutoff, hit_cutoff,), name='hit_input')
    hit_convolution = Conv1D(filters=16, kernel_size=64, padding="same", name='hit_conv')(hits_input_layer)
   
    emb_vector = [embedding_event]
    if use_device_city:
        emb_vector.append(embedding_device)
    if use_hits:
        emb_vector.append(hit_convolution)
    if use_tos:
        emb_vector.append(tos_transformed)
    emb_vector.append(visitno_transformed)

    input_embeddings = concatenate(emb_vector)

    # Add time attention layer for time between sessions
    embedding_phi = Embedding(input_dim=no_channels, output_dim=1, input_length=cutoff, name='phi_embedding')(input_layer_e)
    embedding_mu = Embedding(input_dim=no_channels, output_dim=1, input_length=cutoff, name='mu_embedding')(input_layer_e)
        
    time_input_layer = Input(shape=(cutoff, 1), name='time_input')

    # Scale other inputs according to time attention
    multiply = Multiply(name='multiplication')([embedding_mu,time_input_layer])
    added = Add(name='addition')([embedding_phi, multiply])
    time_attention = Activation(activation='sigmoid', name='attention_activation')(added)

    product = Multiply(name='embeddings_product')([input_embeddings, time_attention])

    if use_time:
        lstm_1 = LSTM(lstm_dim, return_sequences=True, name='lstm1')(product)
        drop_1 = Dropout(rate=dropout, name='dropout1')(lstm_1)
        lstm_2 = LSTM(lstm_dim, return_sequences=True, name='lstm2')(drop_1)
    else:
        lstm_1 = LSTM(lstm_dim, return_sequences=True)(product)
        drop_1 = Dropout(rate=dropout)(lstm_1)
        lstm_2 = LSTM(lstm_dim, return_sequences=True)(drop_1)
    
    drop_2 = Dropout(rate=dropout, name='dropout2')(lstm_2)

    # Attention output layer
    input_attention_layer = TimeDistributed(Dense(1), name='input_attention_layer')(drop_2)
    attention_output_layer = TimeDistributed(Dense(1, activation='softmax'),
                                             name='attention_output_layer')(input_attention_layer)

    attention_product = Multiply(name='attention_product')([drop_2, attention_output_layer])

    flattened_output = Flatten()(attention_product)
    output_layer = Dense(1, activation='sigmoid', name='final_output')(flattened_output)

    model = Model(inputs=[input_layer_e, input_layer_d, input_layer_vn,
                          input_layer_tos, hits_input_layer, time_input_layer], outputs=output_layer)

    opt = Nadam(lr=learn)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def launch_model(hit_cutoff):
    no_epochs = 100

    padded_labels_time = padded_labels[::, cutoff-1]
    padded_labels_time = np.reshape(padded_labels_time, (len(padded_labels_time), 1))

    model = nn_hits_model(hit_cutoff)

    print(model.summary())

    inputs = [padded_docs[::, ::, 0], padded_docs[::, ::, 1], padded_docs[::, ::, 2],
              padded_docs[::, ::, 3], padded_hits_inner, padded_time]


    start_time = datetime.now()
    batch_size = 512

    print('Fitting model')
    model.fit(inputs, padded_labels_time, batch_size=batch_size, epochs=no_epochs, validation_split=0.2, verbose=2)

    current_time = datetime.now()
    duration = current_time - start_time

    loss, accuracy = model.evaluate(inputs, padded_labels_time, verbose=0)
    prediction = model.predict(inputs)

    auc = 0
    try:
        auc = roc_auc_score(np.reshape(padded_labels_time, (len(padded_labels_time) * 1)),
                            np.reshape(prediction, (len(prediction) * 1)))
    except:
        print('AUC not retrievable')

    print('Accuracy: %f' % (accuracy * 100))
    print('AUC: %f' % (auc * 100))
    acc = accuracy
    print(
        f'Accuracy - {embed_dim},{lstm_dim},{learn},{hit_cutoff},{no_layers},{dropout}.{use_time}'
        f',{use_device_city},{use_hits},{use_tos},{cutoff},{lookahead},{no_epochs}, time {time} = {acc},{auc}')
    result_file = open(filename + '.csv', 'a')
    result_file.write(
        f'{prefix},{sample_size},{min_vis},{embed_dim},{lstm_dim},{learn},{hit_cutoff},{no_layers},{dropout}'
        f',{batch_size},{use_time},{use_device_city},{use_hits},{use_tos},{cutoff},{lookahead},{no_epochs},'
        f'{time},{acc},{auc},{duration}\n')
    result_file.close()


#######
# START
#######

prefix = 'ctr_'
# prefix = ''

filename = prefix + 'results_attribution_new'
result_file = open(filename + '.csv', 'a')
result_file.write('prefix,sample_size,min_vis,embed_dim,lstm_dim,learn,hit_cutoff,no_layers,dropout'
                  ',batch_size,use_time,use_device,use_hits,use_tos,cutoff,lookahead,'
                  'no_epochs,time,accuracy,auc,duration\n')
result_file.close()


###########
# Load data
###########

for min_vis in [3]:
    for cutoff in [3]:
        parameter_string = str(min_vis)
        padded_docs = np.load('./data/' + prefix + 'channels_' + parameter_string + '_' + str(cutoff)+'.npy')
        padded_time = np.load('./data/' + prefix + 'time_' + parameter_string + '_' + str(cutoff)+'.npy')
        padded_labels = np.load('./data/' + prefix + 'labels_bin_' + parameter_string + '_' + str(cutoff)+'.npy')
        padded_hits_inner = np.load('./data/' + prefix + 'hits_' + parameter_string + '_' + str(cutoff)+'.npy')
        padded_time = np.reshape(padded_time, (len(padded_time), cutoff, 1))

        print('Channels:', padded_docs.shape)
        print('Time difference:', padded_time.shape)
        print('Labels:', padded_labels.shape)
        print('Hits:', padded_hits_inner.shape)

        no_channels = np.max(padded_docs[::, ::, 0]) + 1
        hit_dim = padded_hits_inner.shape[2]
        no_devices = np.max(padded_docs[::, ::, 1]) + 1
        no_cities = np.max(padded_docs[::, ::, 3]) + 1

        print('Channels:', no_channels)
        print('Hit dim:', hit_dim)
        print('Device enc:', no_devices)

        embed_dim = 64
        lstm_dim = 128
        learn = 0.001
        dropout = 0

        for use_time in [True, False]:
            for use_device_city in [True, False]:
                for use_hits in [True]:
                    for use_tos in [True]:
                        for time in range(0, 1):
                            launch_model(hit_dim)