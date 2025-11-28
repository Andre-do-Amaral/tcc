import os
import json
import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
import logging

from utils.common import import_dataframe
from utils.common import make_lags, make_leads
from utils.common import calculate_metrics
from utils.common import ModeloPrevisaoVolume

import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import sys

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)],
  force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- CONFIGURAÇÕES GLOBAIS (Devem ser consistentes em todas as Lambdas) ---
MLFLOW_URL = os.getenv("MLFLOW_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_KEY")
LOCAL_FILE_PATH = f"data/{S3_KEY}"
MODEL_NAME = "modelo_cantareira_autoreg"
TARGET = 'Volume Útil Armazenado (%)'
# --------------------------------------------------------------------------

logger.info("--- INICIANDO TREINAMENTO COM AUTO-REGRESSÃO ---")

# 1. Configurar MLflow
if not MLFLOW_URL.startswith("http"): MLFLOW_URL = f"http://{MLFLOW_URL}"
mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("experimento_cantareira_autoreg")
# 2. Dados
s3 = boto3.client('s3')
try:
  os.makedirs("data", exist_ok=True)
  s3.download_file(S3_BUCKET_NAME, S3_KEY, LOCAL_FILE_PATH)
except Exception as e:
  logger.error(f"Erro ao carregar dados: {e}")
  raise e

nome_coluna_volume = "Volume Útil Armazenado (%)"
nome_coluna_vazao_natural = "Vazão Natural (m³/s)"
nome_coluna_vazao_jusante = "Vazão Jusante (m³/s)"

df = import_dataframe()
df_volume = df[["Data", nome_coluna_volume]].copy()
df_vazao_natural = df[["Data", nome_coluna_vazao_natural]].copy()
df_vazao_jusante = df[["Data", nome_coluna_vazao_jusante]].copy()

df_volume = df_volume.set_index("Data")
df_vazao_natural = df_vazao_natural.set_index("Data")
df_vazao_jusante = df_vazao_jusante.set_index("Data")

df_volume_series = (
  df_volume
    .groupby('Data').mean()
    .squeeze()
)

df_vazao_natural_series = (
  df_vazao_natural
    .groupby('Data').mean()
    .squeeze()
)

df_vazao_jusante_series = (
  df_vazao_jusante
    .groupby('Data').mean()
    .squeeze()
)

logger.info("--- DADOS CARREGADOS DO S3 ---")

logger.info("--- CRIANDO FEATURES - VAZÃO NATURAL ---")

y_vn = df_vazao_natural_series[df_vazao_natural_series.index >= "2018-01-01"].copy()

fourier = CalendarFourier(freq="YE", order=2)
y_vn = y_vn.asfreq("D")
dp = DeterministicProcess(
  index=y_vn.index,
  constant=False,
  order=1,
  seasonal=False,
  additional_terms=[fourier],
  drop=True,
)
X_full_vn = dp.in_sample()

VALIDATION_SIZE = 1*90

X_vn_train_rec, X_vn_valid_rec, y_vn_train_rec, y_vn_valid_rec = train_test_split(X_full_vn, y_vn, test_size=VALIDATION_SIZE, shuffle=False)

logger.info("--- CRIANDO FEATURES - VAZÃO JUSANTE ---")

y_vj = df_vazao_jusante_series[df_vazao_jusante_series.index >= "2018-01-01"].copy()
X_full_vj = dp.in_sample()

X_vj_train_rec, X_vj_valid_rec, y_vj_train_rec, y_vj_valid_rec = train_test_split(X_full_vj, y_vj, test_size=VALIDATION_SIZE, shuffle=False)

logger.info("--- CRIANDO FEATURES - VOLUME ---")

def get_train_test_split_recursive():
  y_vol = df_volume_series[df_volume_series.index >= "2018-01-01"].copy()
  X_lags = make_lags(y_vol.squeeze(), 1)
  X_Qin_leads = make_leads(y_vn.squeeze(), 1, name="Qin")
  X_Qout_leads = make_leads(y_vj.squeeze(), 1, name="Qout")
  X_full_vol = pd.concat([X_lags, X_Qin_leads, X_Qout_leads], axis=1).dropna()

  y_vol, X_full_vol = y_vol.align(X_full_vol, join='inner', axis=0)

  X_vol_train_rec, X_vol_valid_rec, y_vol_train_rec, y_vol_valid_rec = train_test_split(X_full_vol, y_vol, test_size=VALIDATION_SIZE, shuffle=False)
  return y_vol, X_full_vol, X_vol_train_rec, X_vol_valid_rec, y_vol_train_rec, y_vol_valid_rec

y_vol_rec, X_full_vol, X_vol_train_rec, X_vol_valid_rec, y_vol_train_rec, y_vol_valid_rec = get_train_test_split_recursive()


logger.info("--- TREINANDO MODELO ---")

with mlflow.start_run() as run:
  model = ModeloPrevisaoVolume(LinearRegression(fit_intercept=False))
  model.fit_vazao_natural(X_vn_train_rec, y_vn_train_rec)
  model.fit_vazao_jusante(X_vj_train_rec, y_vj_train_rec)
  model.calculate_lags(X_vn_valid_rec)
  model.fit(X_vol_train_rec, y_vol_train_rec)

  X_vn_train_rec_drop = X_vn_train_rec.drop(pd.Timestamp("2018-01-01"))
  y_fit = model.predict(X_vn_train_rec_drop)
  y_pred = model.predict(X_vn_valid_rec)

  metric = calculate_metrics(
    y_vol_rec.loc[y_fit.index],
    y_vol_rec.loc[y_pred.index],
    y_fit.squeeze(),
    y_pred.squeeze()
  )
  logger.info(metric)

  val_metrics = metric.loc["Validação"]

  for metric_name, value in val_metrics.items():
    mlflow.log_metric(metric_name, float(value))
        
  # Log do Modelo (Artifact)
  mlflow.sklearn.log_model(model, "model", registered_model_name="Modelo Volume")
        
  run_id = run.info.run_id
  logger.info(f"Novo modelo treinado. Run ID: {run_id}. Score MAE: {val_metrics['MAE']}")


# verificar se promove modelo

# download do modelo promovido do mlflow.


logger.info("--- PREVISÃO PARA FUTURO ---")

# Update dos estados internos
model.update(X_full_vol, y_vol_rec, y_vn, y_vj)

# Features para futuro
y_vn = y_vn.asfreq("D")
X_full_vn = dp.in_sample()

X_future_vn = dp.out_of_sample(steps=90)

# Previsão
model.calculate_lags(X_future_vn)
y_pred = model.predict(X_future_vn)
logger.info(y_pred)