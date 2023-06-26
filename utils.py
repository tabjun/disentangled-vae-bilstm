import numpy as np
import pandas as pd

def data_split(data_df, train_delimiter, validation_delimiter):
    train, validation, test = data_df[:train_delimiter]['HR'], data_df[train_delimiter:validation_delimiter]['HR'], data_df[validation_delimiter:]['HR']
    print("Train size: ", train.shape)
    print("Validation size: ",validation.shape)
    print("Test size: ", test.shape)
    return train, validation, test

def train_cleaning(train, anomalies):
    train_df = pd.DataFrame(train.values)
    # Make all anomalies NaN values, then fill them with previous non-anomalous value 
    # This is to construct train data with only normal HR behaviour
    train_df[train_df.index.isin(anomalies)] = np.nan
    cleaned_train = train_df.fillna(method="ffill").fillna(method="bfill")
    return cleaned_train

def anomalies_validation_test(cleaned_train, validation, test, time, anomalies):
    index_validation = list(range(len(cleaned_train), len(cleaned_train) + len(validation)))
    index_test = list(range(len(cleaned_train + validation), len(cleaned_train + validation) + len(test)))
    # Existing anomalies in validation and test
    index_anomalies_validation = list(validation.reindex(time[anomalies]).dropna().index)
    index_anomalies_test = list(test.reindex(time[anomalies]).dropna().index)
    return index_anomalies_validation, index_anomalies_test

def find_threshold(squared_errors, margin):
    threshold = np.mean(squared_errors) + margin * np.std(squared_errors)
    return threshold

def find_anomalies(squared_errors, threshold):
    anomalies = (squared_errors >= threshold)
    return anomalies

def predict_index(df, X_train, n_steps_in, n_steps_out):
    train_predict_index = df.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out -1, :].index
    test_predict_index = df.iloc[X_train.shape[0] + n_steps_in:, :].index
    return train_predict_index, test_predict_index

def data_scaling(train_df,validation_df,test_df, scaler):
    scaler = scaler.fit(train_df[["HR"]])

    train_df["HR"] = scaler.transform(train_df[["HR"]])
    validation_df["HR"] = scaler.transform(validation_df[["HR"]])
    test_df["HR"] = scaler.transform(test_df[["HR"]])
    return train_df, validation_df, test_df

def split_sequence(sequence, n_steps_in, n_steps_out):
    """split a univariate sequence into samples"""
    X,y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def reconstruction_loss(vae, train_X):
    # Reconstruction loss on training set
    x_train_pred = vae.predict(train_X)
    train_mae_loss = np.mean(np.abs(x_train_pred - train_X), axis=1)
    return train_mae_loss

def mean_across_steps(saved_model, embeddings_X, embeddings_y, index, n_steps_out):
    # Model predictions
    vae_lstm_pred = saved_model.predict(embeddings_X,verbose=0)
    # Flatten the predictions by taking the mean across steps
    predict_result = pd.DataFrame()
    for i in range(vae_lstm_pred.shape[0]):
        y_predict = pd.DataFrame(vae_lstm_pred[i], columns=["predicted"], index = index[i: i + n_steps_out])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    
    real_value = pd.DataFrame()
    for i in range(embeddings_y.shape[0]):
        y_value = pd.DataFrame(embeddings_y[i], columns=["real_value"], index = index[i: i + n_steps_out])
        real_value = pd.concat([real_value, y_value], axis=1, sort=False)
    
    predict_result["predicted_mean"] = predict_result.mean(axis=1)
    real_value["real_mean"] = real_value.mean(axis=1)
    return predict_result, real_value

def model_predictions(predict_result, real_value, emb_data_y, latent_space_dim, n_embeddings_in):
    predicted_embeddings = np.array(predict_result["predicted_mean"]).reshape(int(predict_result.shape[0]/ latent_space_dim), latent_space_dim)
    predicted_embeddings = np.concatenate([emb_data_y[-n_embeddings_in:], predicted_embeddings])
    true_embeddings = np.array(real_value["real_mean"]).reshape(int(predict_result.shape[0]/ latent_space_dim), latent_space_dim)
    return predicted_embeddings, true_embeddings

def decoder_predicted_embeddings(decoder, predicted_embeddings, scaler, data_X):
    decoded_yhat = decoder.predict(predicted_embeddings).squeeze()
    rescaled_decoded_yhat = scaler.inverse_transform(decoded_yhat)
    rescaled_data_X = scaler.inverse_transform(data_X.squeeze())
    # Flatten for visualization
    flattened_rescaled_decoded_yhat = np.concatenate(rescaled_decoded_yhat).ravel()
    flattened_rescaled_data_X = np.concatenate(rescaled_data_X).ravel()
    return flattened_rescaled_decoded_yhat, flattened_rescaled_data_X

def augmented_detection(anomalies_vae_lstm, index_anomalies):
    """Augmented anomaly detection (see Xu et al., 2018)"""
    n_anomaly = len(index_anomalies)
    augmented_anomalies_detection = list(anomalies_vae_lstm.index)
    for i in range(n_anomaly):
        for j in anomalies_vae_lstm.index:
            if j in [index_anomalies[i]]:
                original_detection = set(augmented_anomalies_detection)
                for_anomaly_window = set([index_anomalies[i]])
                to_add = list(for_anomaly_window - original_detection)
                augmented_anomalies_detection = augmented_anomalies_detection + to_add
                break
    return list(np.sort(augmented_anomalies_detection))

def intersection(list1, list2):
    list3 = [value for value in list1 if value in list2]
    return list3

def list_diff(list1, list2):
    a = set(list1)
    b = set(list2)
    list3 = [value for value in a if value not in b]
    return list3

def metrics(index_anomalies, model_anomalies):
    tp = len(intersection(index_anomalies, model_anomalies))
    fp = len(list_diff(model_anomalies, index_anomalies))
    fn = len(list_diff(index_anomalies, model_anomalies))
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score

def margin_comparison_and_selection(vae_lstm_train_squared_errors, vae_lstm_validation_squared_errors,
                                    validation_df, index_anomalies_validation, flattened_rescaled_validation_X):
    f1_scores = []
    for i in np.arange(start=1, stop=9, step=1):
        threshold = find_threshold(vae_lstm_train_squared_errors, margin=i)
        validation_anomalies = find_anomalies(vae_lstm_validation_squared_errors, threshold)
        
        vae_lstm_validation_df = pd.DataFrame(index=validation_df.index)
        vae_lstm_validation_df["HR"] = flattened_rescaled_validation_X
        vae_lstm_validation_df["anomalies_vae_lstm"] = validation_anomalies
        anomalies_vae_lstm = vae_lstm_validation_df[vae_lstm_validation_df.anomalies_vae_lstm == True]

        augmented_anomalies_vae_lstm = augmented_detection(anomalies_vae_lstm, index_anomalies_validation)

        precision, recall, f1_score = metrics(index_anomalies_validation, augmented_anomalies_vae_lstm)

        print("Margin: ",i)
        print("Precision: ", round(precision,3))
        print("Recall: ", round(recall,3))
        print("F1-Score: ", round(f1_score,3))

        f1_scores.append(f1_score)

    d = dict()
    for i,j in zip(range(1,len(f1_scores)+1),f1_scores):
        d[i]=j
    # Choose the margin with the associated highest F1-score
    selected_margin = max(d, key=d.get)
    print('Selected threshold: ', selected_margin)
    return selected_margin