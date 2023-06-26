# A Disentangled VAE-BiLSTM Model for Heart Rate Anomaly Detection
Source code for "A Disentangled VAE-BiLSTM Model for Heart Rate Anomaly Detection".

## References
Staffini A., Svensson T., Chung U.-I., Svensson A.K., [**A Disentangled VAE-BiLSTM Model for Heart Rate Anomaly Detection**](https://www.mdpi.com/2306-5354/10/6/683), Bioengineering 10(6):683, 2023.

## Setup
Clone the repo:
```
git clone https://github.com/staale92/disentangled-vae-bilstm.git
cd disentangled-vae-bilstm
```

Set up the virtual environment with Conda:
```
conda create --name disentangled_vae_bilstm python=3.9.16
conda activate disentangled_vae_bilstm
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Datasets
The authors cannot publicly provide access to original data due to participant privacy in accordance with ethical guidelines. Additionally, the written informed consent obtained from study participants does not include a provision for publicly sharing data.

The script `time_series_generation.py` randomly generates a time series with anomalies to test the performance of the model.

## Run the model
To run the model:
```
python main.py
```

Please refer to `config.yml` to change parameters from the default configuration.




