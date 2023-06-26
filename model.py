import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Lambda, Bidirectional, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, MeanAbsoluteError
from tensorflow.keras.metrics import mean_squared_error, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def vae_sampling(args):
    z_mean, z_log_sigma = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size,latent_dim), mean=0, stddev=1)
    
    return z_mean + z_log_sigma * epsilon

def vae_encoder(input_dim, n_features, latent_space_dim, n_units):
    input_x = Input(shape=(input_dim, n_features,))
    encoder_LSTM = Bidirectional(LSTM(n_units, kernel_initializer="random_uniform",
                                                   input_shape = (input_dim, n_features,)), 
                                                   merge_mode = "ave")(input_x)
    z_mean = Dense(latent_space_dim)(encoder_LSTM)
    z_log_sigma = Dense(latent_space_dim)(encoder_LSTM)
    z = Lambda(vae_sampling, output_shape = (latent_space_dim,))([z_mean, z_log_sigma])
    encoder = Model(input_x, [z_mean, z_log_sigma, z], name="encoder")
    encoder.summary()
    return encoder

def vae_decoder(input_dim, n_features, latent_space_dim, n_units):
    decoder_input = Input(shape=(latent_space_dim))
    repeat_decoded = RepeatVector(input_dim)(decoder_input)
    decoder_LSTM = Bidirectional(LSTM(n_units, kernel_initializer="random_uniform",
                                            input_shape=(input_dim, latent_space_dim,), return_sequences=True),
                                            merge_mode = "ave")(repeat_decoded)
    decoder_output = TimeDistributed(Dense((n_features)))(decoder_LSTM)
    decoder = Model(decoder_input, decoder_output, name="decoder")
    decoder.summary()
    return decoder

def beta_vae(input_dim, n_features, encoder, decoder, beta):
    # VAE
    input_x = Input(shape=(input_dim, n_features,))
    z_mean = encoder(input_x)[0]
    z_log_sigma = encoder(input_x)[1]
    z = encoder(input_x)[2]
    
    output = decoder(z)
    vae = Model(input_x, output, name = 'beta_vae')
    vae.summary()

    # VAE Loss
    reconstruction_loss = mse(input_x, output)
    kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    total_loss =  reconstruction_loss + beta * kl_loss
    vae.add_loss(total_loss)
    return vae

def bilstm(emb_train_X, n_units, maximum_norm, dropout_rate, n_steps_out) -> tf.keras.models.Model:
    # BiLSTM backend  
    model = Sequential()
    model.add(Bidirectional(LSTM(n_units, activation='tanh', return_sequences=True, 
                                kernel_constraint= max_norm(maximum_norm), recurrent_constraint=max_norm(maximum_norm), 
                                bias_constraint=max_norm(maximum_norm),
                                input_shape=(emb_train_X.shape[1],emb_train_X.shape[2]))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(int(n_units/2), activation='tanh', return_sequences=True, 
                                kernel_constraint=max_norm(maximum_norm), recurrent_constraint=max_norm(maximum_norm), 
                                bias_constraint=max_norm(maximum_norm))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(int(n_units/4), activation='tanh', return_sequences=False, 
                                kernel_constraint=max_norm(maximum_norm), recurrent_constraint=max_norm(maximum_norm), 
                                bias_constraint=max_norm(maximum_norm))))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_steps_out))
    return model