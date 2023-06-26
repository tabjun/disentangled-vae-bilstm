import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
from keras.models import load_model

from time_series_generation import time_series_generator
from data_labeling import data_labeling_algorithm
from model import vae_sampling, vae_encoder, vae_decoder, beta_vae, bilstm 
from utils import *


if __name__ == '__main__':
    with open("config.yml") as file:
            config = yaml.safe_load(file)

    # Time series generation
    data = list(time_series_generator(median=config['median'], outlier_err=config['outlier_err'], 
                                      size=config['inliers_size'], outlier_size=config['outliers_size']))
    plt.plot(data)
    plt.savefig('./outcome/generated_time_series.png')

    # Data labeling
    data_df = pd.DataFrame(data).rename(columns={0:"HR"})
    data_df, total_anomalies, time = data_labeling_algorithm(data_df, data, if_estimators=config['if_estimators'], 
                                                       if_random_state=config['if_random_state'], 
                                                       if_contamination=config['if_contamination'], kd_algo=config['kd_algo'], 
                                                       kd_kernel=config['kd_kernel'], kd_metric=config['kd_metric'], 
                                                       kd_score=config['kd_score'], svm_nu=config['svm_nu'], 
                                                       svm_kernel=config['svm_kernel'], svm_gamma=config['svm_gamma'],
                                                       sw_window_percentage=config['sw_window_percentage'], sw_std=config['sw_std'])

    # Plot labeled anomalies
    plt.figure(figsize=(20,6))
    plt.plot(data_df.index, data_df[["HR"]], label="HR")
    sns.scatterplot(x=data_df[data_df["total_anomalies"] == True].index, 
                    y=data_df[data_df["total_anomalies"] == True][["HR"]].values.reshape(-1,), 
                    s=30, color="red", label="Anomaly")
    plt.title("Total Anomalies", size=20)
    plt.legend(fontsize=15)
    plt.savefig('./outcome/labeled_anomalies.png')

    # Data split, train cleaning, data scaling and preparation
    train, validation, test = data_split(data_df, 
                                         train_delimiter=config['train_delimiter'], 
                                         validation_delimiter=config['validation_delimiter'])
    anomalies = sorted(total_anomalies)
    cleaned_train = train_cleaning(train, anomalies)
    index_anomalies_validation, index_anomalies_test = anomalies_validation_test(cleaned_train, validation, test, time, anomalies)

    train_df = pd.DataFrame(cleaned_train).rename(columns={0:"HR"})
    validation_df = pd.DataFrame(validation)
    test_df = pd.DataFrame(test)
    scaler = StandardScaler()
    train_df, validation_df, test_df = data_scaling(train_df,validation_df,test_df, scaler)

    # Split into sequences of length n
    train_X = np.array_split(train_df.values.reshape(-1), int(len(train_df)/config['sequence_length']))
    train_X = np.array(train_X).reshape(np.array(train_X).shape[0], np.array(train_X).shape[1], 1)

    validation_X = np.array_split(validation_df.values.reshape(-1), int(len(validation_df)/config['sequence_length']))
    validation_X = np.array(validation_X).reshape(np.array(validation_X).shape[0], np.array(validation_X).shape[1], 1)

    test_X = np.array_split(test_df.values.reshape(-1), int(len(test_df)/config['sequence_length']))
    test_X = np.array(test_X).reshape(np.array(test_X).shape[0], np.array(test_X).shape[1], 1)

    # Beta-VAE
    n_features = config['n_features']
    input_dim = len(train_X[0])
    latent_space_dim = config['latent_space_dimension']
    optimizer = config['optimizer']

    encoder = vae_encoder(input_dim, n_features, latent_space_dim, n_units=config['vae_n_units'])
    decoder = vae_decoder(input_dim, n_features, latent_space_dim, n_units=config['vae_n_units'])
    vae = beta_vae(input_dim, n_features, encoder, decoder, beta=config['beta'])
    vae.compile(optimizer = optimizer)

    es = EarlyStopping(monitor = "val_loss", patience = config['es_patience'], mode = "min")
    vae.fit(train_X, batch_size=config['batch_size'], validation_split=config['validation_split'], 
                     epochs=config['n_epochs'], shuffle=False, callbacks = [es])
    
    # Plot VAE Reconstruction Loss
    train_mae_loss = reconstruction_loss(vae, train_X)
    plt.figure(figsize=(20,6))
    plt.title("Reconstruction Loss on Training Set", size=20)
    plt.xlabel("Training Loss (MAE)", fontsize=15)
    plt.ylabel("N. of samples", fontsize=15)
    sns.histplot(train_mae_loss, bins=50, kde=True)
    plt.savefig('./outcome/train_reconstruction_loss.png')

    # Use the encoder to encode samples in latent space
    latents_train = encoder.predict(train_X)[0]
    latents_validation = encoder.predict(validation_X)[0]
    latents_test = encoder.predict(test_X)[0]

    flattened_latents_train = np.concatenate(latents_train).ravel()
    flattened_latents_validation = np.concatenate(latents_validation).ravel()
    flattened_latents_test = np.concatenate(latents_test).ravel()

    # BiLSTM backend
    n_embeddings_in = config['n_embeddings_in']
    n_steps_in = latent_space_dim * n_embeddings_in
    n_steps_out = latent_space_dim

    emb_train_X, emb_train_y = split_sequence(flattened_latents_train,n_steps_in,n_steps_out)
    emb_validation_X, emb_validation_y = split_sequence(flattened_latents_validation,n_steps_in,n_steps_out)
    emb_test_X, emb_test_y = split_sequence(flattened_latents_test,n_steps_in,n_steps_out)
    emb_train_X = emb_train_X.reshape(emb_train_X.shape[0], emb_train_X.shape[1], 1)
    emb_validation_X = emb_validation_X.reshape(emb_validation_X.shape[0], emb_validation_X.shape[1], 1)
    emb_test_X = emb_test_X.reshape(emb_test_X.shape[0], emb_test_X.shape[1], 1)

    bilstm_model = bilstm(emb_train_X, n_steps_out=n_steps_out, n_units=config['bilstm_n_units'], maximum_norm=config['max_norm'], 
                          dropout_rate=config['dropout_rate'])
    bilstm_model.compile(optimizer = optimizer, loss = MeanAbsoluteError(), metrics = [RootMeanSquaredError()])

    mc = ModelCheckpoint('vae_lstm_anomalydetection.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    history = bilstm_model.fit(emb_train_X, emb_train_y, validation_data=(emb_validation_X,emb_validation_y), 
                               batch_size=config['batch_size'], epochs=config['n_epochs'],verbose=1,callbacks=[es,mc])
    # Load saved model
    saved_model = load_model('vae_lstm_anomalydetection.h5', custom_objects={'rmse':RootMeanSquaredError()})

    # Validation set for threshold defining
    for_index = pd.concat([cleaned_train,validation])
    index_train, index_validation, = predict_index(for_index, train_df, n_steps_in, n_steps_out)
    val_predict_result, val_real_value = mean_across_steps(saved_model, emb_validation_X, emb_validation_y, index_validation, n_steps_out)
    val_predicted_embeddings, val_true_embeddings = model_predictions(val_predict_result, val_real_value, emb_train_y, latent_space_dim, n_embeddings_in)

    flattened_rescaled_val_decoded_yhat, flattened_rescaled_val_X = decoder_predicted_embeddings(decoder, val_predicted_embeddings, 
                                                                                                   scaler, validation_X)
    vae_lstm_validation_squared_errors = (flattened_rescaled_val_decoded_yhat - flattened_rescaled_val_X) ** 2

    train_predict_result, train_real_value = mean_across_steps(saved_model, emb_train_X, emb_train_y, index_train, n_steps_out)
    train_predicted_embeddings, train_true_embeddings = model_predictions(train_predict_result, train_real_value, emb_train_y, latent_space_dim, n_embeddings_in)
    
    flattened_rescaled_train_decoded_yhat, flattened_rescaled_train_X = decoder_predicted_embeddings(decoder, train_predicted_embeddings, scaler, train_X)
    vae_lstm_train_squared_errors = (flattened_rescaled_train_decoded_yhat - flattened_rescaled_train_X) ** 2

    # Threshold selection
    selected_margin = margin_comparison_and_selection(vae_lstm_train_squared_errors, vae_lstm_validation_squared_errors,
                                                      validation_df, index_anomalies_validation, flattened_rescaled_val_X)

    # Test performance
    index_test = list(range(len(cleaned_train + validation), len(cleaned_train + validation) + len(test)))
    test_predict_result, test_real_value = mean_across_steps(saved_model, emb_test_X, emb_test_y, index_test, n_steps_out)
    test_predicted_embeddings, test_true_embeddings = model_predictions(test_predict_result, test_real_value, 
                                                                        emb_validation_y, latent_space_dim, n_embeddings_in)

    flattened_rescaled_test_decoded_yhat, flattened_rescaled_test_X = decoder_predicted_embeddings(decoder, test_predicted_embeddings, scaler, test_X)
    vae_lstm_test_squared_errors = (flattened_rescaled_test_decoded_yhat - flattened_rescaled_test_X) ** 2

    threshold = find_threshold(vae_lstm_train_squared_errors, selected_margin)
    test_anomalies = find_anomalies(vae_lstm_test_squared_errors, threshold)

    # Plot anomaly detection on test set
    vae_lstm_test_df = pd.DataFrame(index=test_df.index)
    vae_lstm_test_df["HR"] = flattened_rescaled_test_X
    vae_lstm_test_df["anomalies_vae_lstm"] = test_anomalies
    anomalies_vae_lstm = vae_lstm_test_df[vae_lstm_test_df.anomalies_vae_lstm == True]
    
    plt.figure(figsize=(20,6))
    plt.plot(vae_lstm_test_df.index, vae_lstm_test_df[["HR"]], label="HR")
    sns.scatterplot(x=anomalies_vae_lstm.index, y=anomalies_vae_lstm[["HR"]].values.reshape(-1,), 
                    s=30, color="red", label="Anomaly")
    plt.title(r"$\beta$-VAE-BiLSTM Anomaly Detection (Test set)", size=20)
    plt.legend(fontsize=15)
    plt.savefig('./outcome/test_anomaly_detection.png')

    # Calculate metrics
    augmented_anomalies_vae_lstm = augmented_detection(anomalies_vae_lstm, index_anomalies_test)
    precision, recall, f1_score = metrics(index_anomalies_test, augmented_anomalies_vae_lstm)

    print("Performance on test set")
    print("Precision: ", round(precision,3))
    print("Recall: ", round(recall,3))
    print("F1-Score: ", round(f1_score,3))
