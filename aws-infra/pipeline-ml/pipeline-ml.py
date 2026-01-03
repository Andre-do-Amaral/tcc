import os
import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging

from utils.common import import_dataframe
from utils.common import make_lags, make_leads

import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import itertools
import joblib
import json
import sys

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)],
  force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# MARK: CONFIGURAÇÕES GLOBAIS (Devem ser consistentes em todas as Lambdas) ---
MLFLOW_URL = os.getenv("MLFLOW_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_KEY = os.getenv("S3_KEY")
LOCAL_FILE_PATH = f"data/{S3_KEY}"
MODEL_NAME = "modelo_cantareira_autoreg"
TARGET = 'Volume Útil Armazenado (%)'
# --------------------------------------------------------------------------

# Utils:
def safe_concat(df_base: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
  if df_base is None or df_base.empty:
    return df_new.copy()
  return pd.concat([df_base, df_new], ignore_index=True)

# MARK: MlFlow utils
# 1. Configurar MLflow
if not MLFLOW_URL.startswith("http"): MLFLOW_URL = f"http://{MLFLOW_URL}"
mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("experimento_cantareira_autoreg")
MLFLOW_MODEL_NAME = "Modelo Volume"
client = MlflowClient()

def get_latest_version(model_name: str) -> str:
  versions = client.search_model_versions(f"name='{model_name}'")
  latest = max(versions, key=lambda v: int(v.version))
  return latest.version


# MARK: Preparar Dados 
logger.info("--- PREPARANDO DADOS ---")
# 1. Dados
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
df.set_index("Data", inplace=True)

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

# MARK: Criar Features
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


# MARK: Funções para salvar/carregar previsões e avaliações no S3
KEY_PREVISOES = "resultado/previsoes.csv"
LOCAL_PATH_PREVISOES = f"data/{KEY_PREVISOES}"
KEY_AVALIACAO = "resultado/avaliacao.csv"
LOCAL_PATH_AVALIACAO = f"data/{KEY_AVALIACAO}"
os.makedirs("data/resultado", exist_ok=True)

def load_previsoes_from_s3():
  try:
    logger.info(f"Baixando {KEY_PREVISOES} do S3...")
    s3.download_file(S3_BUCKET_NAME, KEY_PREVISOES, LOCAL_PATH_PREVISOES)
    df_previsoes = pd.read_csv(LOCAL_PATH_PREVISOES, sep=',')
    if not df_previsoes.empty:
      df_previsoes["inicio"] = pd.to_datetime(df_previsoes["inicio"]).dt.date
    return df_previsoes
  except:
    logger.error(f"Erro ao carregar previsoes.csv")
    raise

def load_avaliacao_from_s3():
  try:
    logger.info(f"Baixando {KEY_AVALIACAO} do S3...")
    s3.download_file(S3_BUCKET_NAME, KEY_AVALIACAO, LOCAL_PATH_AVALIACAO)
    df_avaliacao = pd.read_csv(LOCAL_PATH_AVALIACAO, sep=',')
    if not df_avaliacao.empty:
      df_avaliacao["inicio"] = pd.to_datetime(df_avaliacao["inicio"]).dt.date
    return df_avaliacao
  except:
    logger.error(f"Erro ao carregar avaliacao.csv")
    raise

def save_previsoes_s3(df_previsoes: pd.DataFrame):
  logger.info(f"Salvando previsoes.csv no S3...")
  df_previsoes.to_csv(LOCAL_PATH_PREVISOES, index=False)
  s3.upload_file(LOCAL_PATH_PREVISOES, S3_BUCKET_NAME, KEY_PREVISOES)

def save_avaliacao_s3(df_avaliacao: pd.DataFrame):
  logger.info(f"Salvando avaliacao.csv no S3...")
  df_avaliacao.to_csv(LOCAL_PATH_AVALIACAO, index=False)
  s3.upload_file(LOCAL_PATH_AVALIACAO, S3_BUCKET_NAME, KEY_AVALIACAO)

# MARK: Verificar retreino
logger.info("--- VERIFICAR NECESSIDADE DE RETREINO DO MODELO ---")

rodar_retreino = False
try:
  df_previsoes = load_previsoes_from_s3()
  df_avaliacao = load_avaliacao_from_s3()
except Exception as e:
  rodar_retreino = True
  colunas_MAEs = [f"MAE{7*(i+1)}"for i in range(13)]
  colunas = ["modelo", "inicio"] + colunas_MAEs
  df_avaliacao = pd.DataFrame(columns=colunas)

  save_avaliacao_s3(df_avaliacao)


# Atualizar avaliações MAE
HORIZONTES = [7*(i+1) for i in range(13)]
df_mae_pendente = df_avaliacao[df_avaliacao.isna().any(axis=1)]
for idx, row in df_mae_pendente.iterrows():
  inicio = row["inicio"]
  prev = df_previsoes[df_previsoes["inicio"] == inicio]
  if prev.empty:
    continue
  prev = prev.iloc[0]
  for h in HORIZONTES:
    col_mae = f"MAE{h}"
    # pula se já estiver calculado
    if not pd.isna(row[col_mae]):
      continue
    # datas a avaliar
    datas_h = pd.date_range(
      start=inicio + pd.Timedelta(days=1),
      periods=h,
      freq="D"
    )
    # Verificar se as datas a avaliar existem no df de valores reais.
    if not datas_h.isin(df.index).all():
      continue
    y_true = df.loc[datas_h, nome_coluna_volume].values
    y_pred = prev[[f"y_{i}" for i in range(1, h + 1)]].values
    df_avaliacao.at[idx, col_mae] = mean_absolute_error(y_true, y_pred)
save_avaliacao_s3(df_avaliacao)

if df_avaliacao.empty:
  rodar_retreino = True
else:
  # Verificar necessidade de retreino:
  modelo_mais_recente = (
    df_avaliacao
    .iloc[-1]["modelo"]
  )
  # Isolar só modelo mais recente
  df_avaliacao_modelo_atual = df_avaliacao[df_avaliacao["modelo"] == modelo_mais_recente]
  cols_mae = [c for c in df_avaliacao_modelo_atual.columns if c.startswith("MAE")]
  tem_algum_mae = df_avaliacao_modelo_atual[cols_mae].notna().any().any()
  if not tem_algum_mae:
    rodar_retreino = False
  else:
    LIMIAR_MAE = 5.0
    rodar_retreino = (df_avaliacao_modelo_atual[cols_mae] > LIMIAR_MAE).sum().sum() > 0
#rodar_retreino = True

# MARK: configurações - modelo
# --- CONFIGURAÇÃO GLOBAL ---
VALIDATION_SIZE = 90
N_SPLITS = 5
COL_VOL = "Volume Útil Armazenado (%)"
COL_VN = "Vazão Natural (m³/s)"
COL_VJ = "Vazão Jusante (m³/s)"
LAGS_CICLO = 2

# =============================================================================
# 1. FUNÇÕES AUXILIARES
# =============================================================================
def calcular_climatologia_robusta(df_train, col_target, window_days=7):
  temp = df_train[[col_target]].copy()
  temp['day_of_year'] = temp.index.dayofyear
  climatologia = {}
  for day in range(1, 367):
    d_min = day - (window_days // 2)
    d_max = day + (window_days // 2)
    mask = (temp['day_of_year'] >= d_min) & (temp['day_of_year'] <= d_max)
    vals = temp.loc[mask, col_target]
    if len(vals) > 0:
      vals = vals[(vals >= vals.quantile(0.10)) & (vals <= vals.quantile(0.90))]
      media = vals.mean() if len(vals) > 0 else 0
    else:
      media = 0
    climatologia[day] = media
  return pd.Series(climatologia, name='clim_base')

def get_clim_feature_array(index, curva_clim):
  days = index.dayofyear
  return np.array([curva_clim.get(d, curva_clim.iloc[-1]) for d in days])

def criar_feat_fourier(index):
  # Cria features determinísticas (Tendência + Sazonalidade Anual)
  dp = DeterministicProcess(
    index=index, constant=True, order=1, seasonal=False,
    additional_terms=[CalendarFourier("YE", 2)], drop=True
  )
  return dp.in_sample()

def criar_lags(series, lags):
  df_lag = pd.DataFrame(index=series.index)
  for l in range(1, lags + 1):
    df_lag[f'lag_{l}'] = series.shift(l)
  return df_lag

# =============================================================================
# 2. CLASSE GENERALISTA (COM SUPORTE A VOLUME HÍBRIDO)
# =============================================================================
class ReservoirForecaster:
  def __init__(self, config):
    self.cfg = config
    self.models = {}
    self.scalers = {}
    self.history = {}
    self.vol_features_ = []

  def _train_flow_model(self, df_train, col_name, prefix):
    """Treina modelo de vazão (Natural ou Jusante) usando Sazonalidade + ML no Resíduo."""
    y = df_train[col_name]

    # 1. Climatologia
    clim_curve = calcular_climatologia_robusta(df_train, col_name)
    self.history[f'{prefix}_clim'] = clim_curve

    # 2. Resíduo via Fourier (Tendência Matemática)
    X_math = criar_feat_fourier(df_train.index)
    model_math = LinearRegression()
    model_math.fit(X_math, y)
    self.models[f'{prefix}_math'] = model_math

    residuo = y - model_math.predict(X_math)

    # 3. ML no Ciclo (Resíduo)
    X_lags = criar_lags(residuo, LAGS_CICLO).dropna()
    y_target = residuo.loc[X_lags.index]

    model_cycle = clone(self.cfg['model_flow_cycle'])
    model_cycle.fit(X_lags, y_target)
    self.models[f'{prefix}_cycle'] = model_cycle

    return residuo.tail(LAGS_CICLO).values

  def fit(self, df_train):
    """Treina todos os submodelos (Vazões e Volume)."""
    # Treina modelos de vazão (sempre treinamos para poder prever o futuro recursivamente)
    self.last_res_vn = self._train_flow_model(df_train, COL_VN, 'vn')
    self.last_res_vj = self._train_flow_model(df_train, COL_VJ, 'vj')

    # Prepara features do Volume
    y = df_train[COL_VOL]
    X = pd.DataFrame(index=df_train.index)
    X['vol_lag1'] = df_train[COL_VOL].shift(1)

    if self.cfg.get('use_diff1', True):
      X['vol_diff1'] = df_train[COL_VOL].shift(1) - df_train[COL_VOL].shift(2)

    if self.cfg.get('use_diff2', True):
      X['vol_diff2'] = (df_train[COL_VOL].shift(1) - df_train[COL_VOL].shift(2)) - \
                       (df_train[COL_VOL].shift(2) - df_train[COL_VOL].shift(3))

    # Ativação de Exógenas via Config
    if self.cfg.get('use_vn', True):
      X['vn_input'] = df_train[COL_VN]
    if self.cfg.get('use_vj', True):
      X['vj_input'] = df_train[COL_VJ]

    if self.cfg.get('use_vol_clim', True):
      self.history['vol_clim'] = calcular_climatologia_robusta(df_train, COL_VOL)
      X['vol_clim'] = get_clim_feature_array(df_train.index, self.history['vol_clim'])

    if self.cfg.get('vol_feat_fourier', True):
      X_fourier = criar_feat_fourier(df_train.index)
      X = pd.concat([X, X_fourier], axis=1)

    X_full = X.dropna()
    self.vol_features_ = X_full.columns.tolist()

    # Scaling e Treino do Volume
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    self.scalers['vol'] = scaler

    model_vol = self.cfg['model_volume']
    model_vol.fit(X_scaled, y.loc[X_full.index])
    self.models['volume'] = model_vol

    # Salva estado para o forecast futuro
    self.last_vol_data = df_train[COL_VOL].tail(3).values.tolist()
    self.last_date = df_train.index[-1]

  def _predict_loop(self, dates, start_vol_buffer, start_res_vn, start_res_vj):
    """Loop recursivo para prever N dias."""
    preds_vol = []
    curr_vol = list(start_vol_buffer)
    curr_res_vn = list(start_res_vn)
    curr_res_vj = list(start_res_vj)

    X_trend = None
    if self.cfg.get('vol_feat_fourier', True):
      X_trend = criar_feat_fourier(pd.DatetimeIndex(dates))

    for date in dates:
      # --- 1. PREVER VAZÕES (Recursivo) ---
      def predict_single_flow(prefix, res_buffer):
        base = get_clim_feature_array(pd.DatetimeIndex([date]), self.history[f'{prefix}_clim'])[0]
        lags = np.array(res_buffer[-LAGS_CICLO:][::-1])
        res_pred = self.models[f'{prefix}_cycle'].predict(pd.DataFrame([lags], columns=[f'lag_{k}' for k in range(1, LAGS_CICLO+1)]))[0]

        # Lógica de Damping
        if self.cfg.get('cycle_logic') == 'damping_specific' and res_pred > 0:
            res_pred *= 0.2 if (res_pred - res_buffer[-1] > 0) else 0.8

        return max(0, base + res_pred), res_pred

      vn_val, vn_res = predict_single_flow('vn', curr_res_vn)
      vj_val, vj_res = predict_single_flow('vj', curr_res_vj)
      curr_res_vn.append(vn_res)
      curr_res_vj.append(vj_res)

      # --- 2. PREVER VOLUME ---
      feat_dict = {'vol_lag1': curr_vol[-1]}
      if 'vol_diff1' in self.vol_features_: feat_dict['vol_diff1'] = curr_vol[-1] - curr_vol[-2]
      if 'vol_diff2' in self.vol_features_: feat_dict['vol_diff2'] = (curr_vol[-1] - curr_vol[-2]) - (curr_vol[-2] - curr_vol[-3])
      if 'vn_input' in self.vol_features_: feat_dict['vn_input'] = vn_val
      if 'vj_input' in self.vol_features_: feat_dict['vj_input'] = vj_val
      if 'vol_clim' in self.vol_features_:
        feat_dict['vol_clim'] = get_clim_feature_array(pd.DatetimeIndex([date]), self.history['vol_clim'])[0]

      if X_trend is not None:
        row_trend = X_trend.loc[[date]].reset_index(drop=True)
        input_df = pd.concat([pd.DataFrame([feat_dict]), row_trend], axis=1)
      else:
        input_df = pd.DataFrame([feat_dict])
      input_df = input_df[self.vol_features_]
      input_scaled = self.scalers['vol'].transform(input_df)
      pred_v = self.models['volume'].predict(input_scaled)[0]

      preds_vol.append(pred_v)
      curr_vol.append(pred_v)
      curr_vol.pop(0)

    return preds_vol

  def forecast(self, days):
    """Faz a previsão para o futuro além dos dados conhecidos."""
    future_dates = pd.date_range(start=self.last_date + pd.Timedelta(days=1), periods=days)
    return self._predict_loop(future_dates, self.last_vol_data, self.last_res_vn, self.last_res_vj)

  def evaluate(self, df):
    """Validação cruzada temporal."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=VALIDATION_SIZE)
    rmse_scores = []
    mae_scores = []
    for train_idx, test_idx in tscv.split(df):
      df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

      self.fit(df_train)
      buffer_vol = df_train[COL_VOL].tail(3).values
      preds = self._predict_loop(df_test.index, buffer_vol, self.last_res_vn, self.last_res_vj)

      rmse = np.sqrt(mean_squared_error(df_test[COL_VOL], preds))
      rmse_scores.append(rmse)
      mae = mean_absolute_error(df_test[COL_VOL], preds)
      mae_scores.append(mae)
    return np.mean(rmse_scores), np.mean(mae_scores)

# MARK: Treinando modelo
logger.info("--- TREINANDO MODELO ---")
if(rodar_retreino):
  # =============================================================================
  # 3. EXECUÇÃO DO GRID SEARCH
  # =============================================================================

  param_grid = {
    'use_vn': [True],
    'use_vj': [
      False,
      True
    ],
    # Modelo Vazão (Fixo na sua melhor configuração)
    'model_flow_cycle': [
      KNeighborsRegressor(n_neighbors=5),
      RandomForestRegressor(max_depth=3),
      RandomForestRegressor(max_depth=5)
    ],
    'cycle_logic': [
      'damping_specific',
      None
    ],
    # Modelo Volume (Variações)
    'model_volume': [
      LinearRegression(),
      Ridge(alpha=1.0)
    ],
    'use_diff1': [
      True,
      False
    ],
    'use_diff2': [
      True,
      False
    ],
    'use_vol_clim': [True, False],
    'vol_feat_fourier': [
      True,
      False
    ] # Testa se incluir Fourier direto no volume ajuda
  }

  keys, values = zip(*param_grid.items())
  combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

  logger.info(f"Total de configurações a testar: {len(combinations)}")
  logger.info("=== INICIANDO GRID SEARCH ===")

  # Inicia a Run "Pai" que vai agrupar todas as tentativas
  with mlflow.start_run(run_name="Estudo_Grid_Search") as parent_run:
    # Logar parâmetros globais do estudo (opcional)
    mlflow.log_param("total_trials", len(combinations))

    results = []
    for i, config in enumerate(combinations):
      cfg_run = config.copy()
      with mlflow.start_run(run_name=f"iter_{i}", nested=True) as child_run:
        try:
          # 1. Logar os parâmetros DESTA iteração antes de rodar
          mlflow.log_param("use_vn", cfg_run['use_vn'])
          mlflow.log_param("use_vj", cfg_run['use_vj'])
          mlflow.log_param("model_flow_cycle", str(cfg_run['model_flow_cycle']))
          mlflow.log_param("cycle_logic", str(cfg_run['cycle_logic']))
          mlflow.log_param("model_volume", str(cfg_run['model_volume']))
          mlflow.log_param("use_diff1", cfg_run['use_diff1'])
          mlflow.log_param("use_diff2", cfg_run['use_diff2'])
          mlflow.log_param("use_vol_clim", cfg_run['use_vol_clim'])
          mlflow.log_param("vol_feat_fourier", cfg_run['vol_feat_fourier'])

          # 2. Executar o modelo
          forecaster = ReservoirForecaster(cfg_run)
          rmse, mae = forecaster.evaluate(df)

          # 3. Logar Métricas
          mlflow.log_metric("rmse", rmse)
          mlflow.log_metric("mae", mae)

          # 4. Tag
          run_name_dynamic = f"RMSE_{rmse:.4f}MAE{mae:.4f}Iter{i}"
          mlflow.set_tag("mlflow.runName", run_name_dynamic)

          results.append({
            "run_id": child_run.info.run_id,
            "rmse": rmse,
            "mae": mae,
            "forecaster_obj": forecaster,
            "config": cfg_run,
            "iter": i
          })

          logger.info(f"Iter {i}: RMSE={rmse:.4f} | MAE={mae:.4f} | Sucesso")
        except Exception as e:
          logger.info(f"Erro na config {i}: {e}")
          mlflow.set_tag("status", "failed")
          mlflow.log_param("error_msg", str(e))

    # =============================================================================
    # ANÁLISE FINAL (OPCIONAL NO CÓDIGO, POIS O MLFLOW UI JÁ FAZ ISSO)
    # =============================================================================
    logger.info("\n=== SALVANDO ARTEFATOS DOS TOP 5 ===")
    sorted_results = sorted(results, key=lambda x: x['rmse'])
    top_5 = sorted_results[:5]

    for rank, item in enumerate(top_5):
      run_id = item['run_id']
      forecaster = item['forecaster_obj']
      config_cfg = item['config']
      rmse = item['rmse']

      logger.info(f"Salvando Rank #{rank+1} (RMSE: {rmse:.4f}) na run {run_id}...")

      # Reabre a run específica
      with mlflow.start_run(run_id=run_id, nested=True):
        # 1. Serializa o objeto completo para um arquivo local temporário
        local_filename = f"forecaster_rank_{rank+1}.pkl"
        joblib.dump(forecaster, local_filename)

        # 2. Envia esse arquivo como artefato para o MLflow
        mlflow.log_artifact(local_filename, artifact_path="model_full_class")

        # Tag para facilitar busca
        mlflow.set_tag("ranking", f"Top_{rank+1}")

        # 3. O melhor modelo é registrado no MLflow Model Registry
        if(rank == 0):
          model_info = mlflow.sklearn.log_model(
            sk_model=forecaster,
            name="model"
          )

          result = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MLFLOW_MODEL_NAME
          )

          # =============================================================================
          # verificar se promove modelo
          client.set_registered_model_alias(
            name=MLFLOW_MODEL_NAME,
            version=result.version,
            alias="production"
          )

        # Limpa arquivo local
        if os.path.exists(local_filename):
          os.remove(local_filename)


# MARK: Prever!!
logger.info("--- PREVISÃO PARA FUTURO ---")
model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}@production")
model_version = client.get_model_version_by_alias(
  name=MLFLOW_MODEL_NAME,
  alias="production"
)

config_cfg = model.cfg
config_cfg['model_volume'] = clone(config_cfg['model_volume'])
config_cfg['model_flow_cycle'] = clone(config_cfg['model_flow_cycle'])

logger.info("Treinando modelos com o histórico completo...")
final_model = ReservoirForecaster(config_cfg)
final_model.fit(df)
# horizonte:
horizonte = 90
logger.info(f"Gerando previsão para os próximos {horizonte} dias...")
forecast_values = final_model.forecast(horizonte)

# =============================================================================
# 5. RESULTADOS E VISUALIZAÇÃO
# =============================================================================

# Criar DataFrame com o resultado
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizonte)

df_resultado = pd.DataFrame(
  data=forecast_values,
  index=future_dates,
  columns=['Previsao_Volume']
)


logger.info("=============")
logger.info(df_resultado)
logger.info("=============")
y_pred = df_resultado['Previsao_Volume']


# MARK: Salvar Previsão
logger.info("--- SALVAR PREVISÃO NO S3 ---")
from datetime import date, datetime

year = datetime.today().year
month = datetime.today().month
day = datetime.today().day
data_previsao = date(year, month, day)

coluna_previsoes = [f"y_{i+1}" for i in range(90)]
linha_inserir = pd.DataFrame(columns=["modelo", "inicio", "alterou_modelo"] + coluna_previsoes)
nome_modelo = f"{MLFLOW_MODEL_NAME}_v{model_version.version}"
linha_inserir.loc[0, "modelo"] = nome_modelo
linha_inserir.loc[0, "inicio"] = data_previsao
linha_inserir.loc[0, "alterou_modelo"] = 1 if rodar_retreino else 0

valores = y_pred.squeeze().to_numpy()
linha_inserir.loc[0, coluna_previsoes] = valores

try:
  df_previsoes = load_previsoes_from_s3()
except Exception as e:
  colunas = ["modelo", "inicio", "alterou_modelo"] + coluna_previsoes
  df_previsoes = pd.DataFrame(columns=colunas)

# Adiciona nova linha se data_previsao não existir
if data_previsao not in df_previsoes["inicio"].values:
  # Salvar previsão
  df_previsoes = safe_concat(
    df_previsoes,
    linha_inserir
  )
  save_previsoes_s3(df_previsoes)

  # Salvar nova linha para avaliação
  df_avaliacao = load_avaliacao_from_s3()
  df_nova_avaliacao = pd.DataFrame(
    [{
      "modelo": nome_modelo,
      "inicio": data_previsao,
    }],
    columns=df_avaliacao.columns
  )
  df_avaliacao = safe_concat(
    df_avaliacao,
    df_nova_avaliacao
  )
  save_avaliacao_s3(df_avaliacao)

logger.info("--- PIPELINE EXECUTADO COM SUCESSO ---")