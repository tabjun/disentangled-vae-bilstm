import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM

def data_labeling_algorithm(data_df, data,
                            if_estimators, if_random_state, if_contamination, 
                            kd_algo, kd_kernel, kd_metric, kd_score,
                            svm_nu, svm_kernel, svm_gamma,
                            sw_window_percentage, sw_std):
    # Isolation Forest
    model_if = IsolationForest(n_estimators=if_estimators, random_state=if_random_state, contamination=if_contamination)
    model_if.fit(data_df.values)
    anomaly_dict = {-1: True, 1: False}
    data_df["anomaly_if"] = model_if.predict(data_df.values)
    data_df["anomaly_if"] = data_df["anomaly_if"].map(anomaly_dict)
    anomalies_if = data_df[data_df.anomaly_if == True]

    # Kernel Density Estimation
    kern_dens = KernelDensity(algorithm=kd_algo, kernel=kd_kernel, metric=kd_metric)
    kern_dens.fit(data_df.values)
    scores = kern_dens.score_samples(data_df.values)
    threshold = np.quantile(scores, kd_score)
    data_df["anomaly_kde"] = np.where(scores <= threshold, -1, 1)
    data_df["anomaly_kde"] = data_df["anomaly_kde"].map(anomaly_dict)
    anomalies_kde = data_df[data_df.anomaly_kde == True]

    # Fit One Class SVM
    one_class_svm = OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma).fit(data_df.values)
    data_df["anomaly_svm"] = one_class_svm.predict(data_df.values)
    data_df["anomaly_svm"] = data_df["anomaly_svm"].map(anomaly_dict)
    anomalies_svm = data_df[data_df.anomaly_svm == True]

    # Sliding Window
    column = data_df[["HR"]].to_numpy().reshape(-1,)
    N = len(column)
    time = np.arange(0,N)
    window_percentage = sw_window_percentage
    k = int(len(column) * (window_percentage/100))
    for_bands = pd.DataFrame(data).reset_index()
    get_bands = lambda for_bands : (np.mean(for_bands) + sw_std*np.std(for_bands),
                                    np.mean(for_bands) - sw_std*np.std(for_bands))
    bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)]) for i in range(0,N)]
    upper, lower = zip(*bands)
    # compute anomalies via sliding window
    data_df["anomaly_sw"] = (column > upper) | (column < lower)
    anomalies_sw = data_df[data_df.anomaly_sw == True]

    # Identify anomalies to be intersected and to be added
    index_anomalies_if = set(anomalies_if.index)
    index_anomalies_kde = set(anomalies_kde.index)
    index_anomalies_svm = set(anomalies_svm.index)
    index_anomalies_sw = set(anomalies_sw.index)

    anomalies_1st_group = set.intersection(index_anomalies_if, index_anomalies_kde, index_anomalies_svm)
    total_anomalies = anomalies_1st_group.union(index_anomalies_sw)
    data_df["total_anomalies"] = data_df.index.isin(total_anomalies)
    
    return data_df, total_anomalies, time

