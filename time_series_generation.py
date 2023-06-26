import numpy as np
import pandas as pd

def time_series_generator(median, outlier_err, size, outlier_size):
    errs = np.random.rand(size) * np.random.choice((-1, 1), size)
    data = median + errs

    lower_errs = outlier_err * np.random.rand(int(outlier_size/2))
    lower_outliers = median - lower_errs

    upper_errs = outlier_err * np.random.rand(int(outlier_size/2))
    upper_outliers = median + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)
    return data

