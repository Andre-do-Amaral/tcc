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
    s3.download_file(S3_BUCKET_NAME, KEY_PREVISOES, LOCAL_PATH_PREVISOES)
    df_previsoes = pd.read_csv(LOCAL_PATH_PREVISOES, sep=',')
    return df_previsoes
  except:
    logger.error(f"Erro ao carregar previsoes.csv")
    raise

def load_avaliacao_from_s3():
  try:
    s3.download_file(S3_BUCKET_NAME, KEY_AVALIACAO, LOCAL_PATH_AVALIACAO)
    df_avaliacao = pd.read_csv(LOCAL_PATH_AVALIACAO, sep=',')
    return df_avaliacao
  except:
    logger.error(f"Erro ao carregar avaliacao.csv")
    raise

def save_previsoes_s3(df_previsoes: pd.DataFrame):
  df_previsoes.to_csv(LOCAL_PATH_PREVISOES, index=False)
  s3.upload_file(LOCAL_PATH_PREVISOES, S3_BUCKET_NAME, KEY_PREVISOES)

def save_avaliacao_s3(df_avaliacao: pd.DataFrame):
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
  colunas_MAEs = [f"MAE{7*(i+1)}"for i in range(15)]
  colunas = ["modelo", "inicio"] + colunas_MAEs
  df_avaliacao = pd.DataFrame(columns=colunas)

  save_avaliacao_s3(df_avaliacao)


# Atualizar avaliações MAE
HORIZONTES = [7*(i+1) for i in range(15)]
df_mae_pendente = df_avaliacao[df_avaliacao.isna().any(axis=1)]
for idx, row in df_mae_pendente.iterrows():
  inicio = pd.to_datetime(row["inicio"])
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
    y_true = df.loc[datas_h, "y"].values
    y_pred = prev[[f"y_{i}" for i in range(1, h + 1)]].values
    df_avaliacao.at[idx, col_mae] = mean_absolute_error(y_true, y_pred)
save_avaliacao_s3(df_avaliacao)

if df_avaliacao.empty:
  rodar_retreino = True
else:
  # Verificar necessidade de retreino:
  modelo_mais_recente = (
    df_avaliacao
    .sort_values("inicio")
    .iloc[-1]["modelo"]
  )
  # Isolar só modelo mais recente
  df_avaliacao_modelo_atual = df_avaliacao[df_avaliacao["modelo"] == modelo_mais_recente]
  cols_mae = [c for c in df_avaliacao_modelo_atual.columns if c.startswith("MAE")]
  tem_algum_mae = df_avaliacao_modelo_atual[cols_mae].notna().any().any()
  if not tem_algum_mae:
    rodar_retreino = False
  else:
    LIMIAR_MAE = 10.0
    mae_atual = (
      df_avaliacao_modelo_atual
        .sort_values("inicio")
        .iloc[-1][cols_mae]
        .dropna()
        .iloc[-1]
    )
    rodar_retreino = mae_atual > LIMIAR_MAE
#rodar_retreino = True

# MARK: configurações - modelo
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import itertools
import joblib

# --- CONFIGURAÇÃO GLOBAL ---
VALIDATION_SIZE = 90
N_SPLITS = 5
COL_VOL = "Volume Útil Armazenado (%)"
COL_VN = "Vazão Natural (m³/s)"
LAGS_CICLO = 2
LAGS_VOLUME = 1

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

  def _train_flow(self, df_train):
    y = df_train[COL_VN]

    # 1. Climatologia (Base)
    clim_curve = calcular_climatologia_robusta(df_train, COL_VN)
    self.history['flow_clim'] = clim_curve

    # 2. Estratégia: Extração de Resíduo via Fourier
    X_math = criar_feat_fourier(df_train.index)
    model_math = LinearRegression()
    model_math.fit(X_math, y)
    ### Adicionei para evitar treinar de novo no predict
    self.models['flow_math'] = model_math

    y_math_clean = pd.Series(model_math.predict(X_math), index=y.index)
    residuo = y - y_math_clean

    # 3. Treina ML no Resíduo (Ciclo)
    X_lags = criar_lags(residuo, LAGS_CICLO).dropna()
    y_target = residuo.loc[X_lags.index]

    model_cycle = self.cfg['model_flow_cycle']
    model_cycle.fit(X_lags, y_target)
    self.models['flow_cycle'] = model_cycle

    return residuo.tail(LAGS_CICLO).values

  def _train_volume(self, df_train):
    y = df_train[COL_VOL]
    X = pd.DataFrame(index=df_train.index)

    # --- Feature Engineering Dinâmica ---
    X['vol_lag1'] = df_train[COL_VOL].shift(1)

    if self.cfg.get('use_diff1', True):
      X['vol_diff'] = df_train[COL_VOL].shift(1) - df_train[COL_VOL].shift(2)

    if self.cfg.get('use_diff2', False):
      X['vol_diff2'] = (df_train[COL_VOL].shift(1) - df_train[COL_VOL].shift(2)) - \
                       (df_train[COL_VOL].shift(2) - df_train[COL_VOL].shift(3))

    # Input Exógeno: Vazão
    X['vazao_input'] = df_train[COL_VN]

    # Input Exógeno: Climatologia Volume
    if self.cfg.get('use_vol_clim', False):
      clim_vol = calcular_climatologia_robusta(df_train, COL_VOL)
      self.history['vol_clim'] = clim_vol
      X['vol_clim'] = get_clim_feature_array(df_train.index, clim_vol)

    # Input Exógeno: Tendência Híbrida (Fourier no Volume) -> IMPLEMENTADO AQUI
    if self.cfg.get('vol_hybrid_trend', False):
      # Cria Fourier para o índice de treino
      X_fourier = criar_feat_fourier(df_train.index)
      # Concatena (Pandas alinha pelo índice)
      X = pd.concat([X, X_fourier], axis=1)

    X_full = X.dropna()
    y_train = y.loc[X_full.index]

    # Salva features para garantir ordem no predict
    self.vol_features_ = X_full.columns.tolist()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    self.scalers['vol'] = scaler

    # Treino
    model = self.cfg['model_volume']
    model.fit(X_scaled, y_train)
    self.models['volume'] = model

  def predict(self, df_test, buffer_vol, last_resid_flow):
    preds_vol = []

    curr_buffer_vol = list(buffer_vol)
    curr_resid_flow = list(last_resid_flow)

    # --- PREPARAÇÃO TREND HÍBRIDA (SE NECESSÁRIO) ---
    # Gera o Fourier para todo o período de teste de uma vez (mais eficiente)
    X_trend_test = None
    if self.cfg.get('vol_hybrid_trend', False):
      X_trend_test = criar_feat_fourier(df_test.index)

    for i in range(len(df_test)):
      date = df_test.index[i]

      # --- 1. VAZÃO (Híbrida: Base Clim + Ciclo ML) ---
      val_base_clim = get_clim_feature_array(pd.DatetimeIndex([date]), self.history['flow_clim'])[0]

      lags_cycle = np.array(curr_resid_flow[-LAGS_CICLO:][::-1])
      cycle_pred = self.models['flow_cycle'].predict(pd.DataFrame([lags_cycle], columns=[f'lag_{k}' for k in range(1, LAGS_CICLO+1)]))[0]

      # Damping Logic
      if self.cfg.get('cycle_logic') == 'damping_specific':
        if cycle_pred > 0:
          last_r = curr_resid_flow[-1]
          if cycle_pred - last_r > 0:
            cycle_pred *= 0.2
          else:
            cycle_pred *= 0.8

      vazao_final = max(0, val_base_clim + cycle_pred)
      curr_resid_flow.append(cycle_pred)

      # --- 2. VOLUME ---
      vol_t1 = curr_buffer_vol[-1]
      vol_t2 = curr_buffer_vol[-2]
      vol_t3 = curr_buffer_vol[-3]

      # Monta features base (Dicionário não garante ordem, DF garante)
      feat_dict = {'vol_lag1': vol_t1}

      if self.cfg.get('use_diff1', True):
        feat_dict['vol_diff'] = vol_t1 - vol_t2

      if self.cfg.get('use_diff2', False):
        feat_dict['vol_diff2'] = (vol_t1 - vol_t2) - (vol_t2 - vol_t3)

      feat_dict['vazao_input'] = vazao_final

      if self.cfg.get('use_vol_clim', False):
        feat_dict['vol_clim'] = get_clim_feature_array(pd.DatetimeIndex([date]), self.history['vol_clim'])[0]

      # Cria DataFrame parcial
      input_vol = pd.DataFrame([feat_dict])

      # Adiciona Trend Híbrida se necessário
      if self.cfg.get('vol_hybrid_trend', False):
        # Pega a linha correspondente do Fourier pré-calculado
        # Reset index para concatenar lateralmente sem problemas de índice (input_vol tem index 0)
        row_trend = X_trend_test.iloc[[i]].reset_index(drop=True)
        input_vol = pd.concat([input_vol, row_trend], axis=1)

      # REORDENAÇÃO CRÍTICA: Garante que as colunas estão na mesma ordem do fit
      # Isso corrige qualquer erro de concatenação ou ordem de dicionário
      input_vol = input_vol[self.vol_features_]

      # Scaling
      input_vol_scaled = self.scalers['vol'].transform(input_vol)

      # Predict
      pred_vol = self.models['volume'].predict(input_vol_scaled)[0]
      preds_vol.append(pred_vol)

      curr_buffer_vol.append(pred_vol)
      curr_buffer_vol.pop(0)

    return preds_vol

  def evaluate(self, df):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=VALIDATION_SIZE)
    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
      df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

      last_resid_flow = self._train_flow(df_train)
      self._train_volume(df_train)

      buffer_vol = df_train[COL_VOL].tail(3).values
      preds_vol = self.predict(df_test, buffer_vol, last_resid_flow)

      rmse = np.sqrt(mean_squared_error(df_test[COL_VOL], preds_vol))
      rmse_scores.append(rmse)
      mae = mean_absolute_error(df_test[COL_VOL], preds_vol)
      mae_scores.append(mae)

    return np.mean(rmse_scores), np.mean(mae_scores)

  ### Coloquei aqui para não treinar modelo ao prever.
  def init_flow_state(self, df_hist):
    """
    Reconstrói o estado do resíduo da vazão
    SEM treinar nenhum modelo
    """
    y = df_hist[COL_VN]

    # Base climatológica (já treinada e salva)
    X_math = criar_feat_fourier(df_hist.index)

    # Usa o modelo linear já treinado
    y_math = pd.Series(
      self.models['flow_math'].predict(X_math),
      index=y.index
    )

    residuo = y - y_math
    return residuo.tail(LAGS_CICLO).values

  def init_volume_buffer(self, df_hist):
    return df_hist[COL_VOL].tail(3).values


# MARK: Treinando modelo
logger.info("--- TREINANDO MODELO ---")
if(rodar_retreino):
  # =============================================================================
  # 3. EXECUÇÃO DO GRID SEARCH
  # =============================================================================

  param_grid = {
    # Modelo Vazão (Fixo na sua melhor configuração)
    'model_flow_cycle': [
      KNeighborsRegressor(n_neighbors=5),
      #RandomForestRegressor(max_depth=3),
      #RandomForestRegressor(max_depth=5)
    ],
    'cycle_logic': [
      'damping_specific',
      #None
    ],
    # Modelo Volume (Variações)
    'model_volume': [
      LinearRegression(),
      #Ridge(alpha=1.0)
    ],
    'use_diff1': [
      True,
      #False
    ],
    'use_diff2': [
      True,
      #False
    ],
    'use_vol_clim': [True, False],
    # NOVA FEATURE PARA TESTE
    'vol_hybrid_trend': [
      #True,
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
          mlflow.log_param("model_volume", str(cfg_run['model_volume']))
          mlflow.log_param("model_flow", str(cfg_run['model_flow_cycle']))
          mlflow.log_param("cycle_logic", str(cfg_run['cycle_logic']))
          mlflow.log_param("use_diff2", cfg_run['use_diff2'])
          mlflow.log_param("vol_hybrid_trend", cfg_run['vol_hybrid_trend'])

          # 2. Executar o modelo
          forecaster = ReservoirForecaster(cfg_run)
          rmse, mae = forecaster.evaluate(df)

          # 3. Logar Métricas
          mlflow.log_metric("rmse", rmse)
          mlflow.log_metric("mae", mae)

          # 4. Tag
          run_name_dynamic = f"RMSE_{rmse:.4f}MAE{mae:.4f}Iter{i}"
          mlflow.set_tag("mlflow.runName", run_name_dynamic)

          # 5. Logar o Modelo (Artifact)
          results.append({
            "run_id": child_run.info.run_id,
            "rmse": rmse,
            "mae": mae,
            "forecaster_obj": forecaster,
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

# Histórico conhecido
dados_prev = df.iloc[:-90, :]

# Warm-up de estado (SEM FIT)
last_resid_flow = model.init_flow_state(dados_prev)
buffer_vol = model.init_volume_buffer(dados_prev)

# Horizonte
DAYS_AHEAD = 90
last_date = dados_prev.index[-1]
future_dates = pd.date_range(
  start=last_date + pd.Timedelta(days=1),
  periods=DAYS_AHEAD,
  freq='D'
)

df_future = pd.DataFrame(index=future_dates)

# Predict puro
forecast_values = model.predict(
  df_future,
  buffer_vol,
  last_resid_flow
)

df_resultado = pd.DataFrame(
  forecast_values,
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
if not df_previsoes.empty:
  df_previsoes["inicio"] = pd.to_datetime(df_previsoes["inicio"]).dt.date
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