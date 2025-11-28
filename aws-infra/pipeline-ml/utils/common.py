import pandas as pd

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone


def import_dataframe():
  df = pd.read_csv('data/dados_sabesp_cantareira_historico.csv', sep=';', parse_dates=['Data'], dayfirst=True)
  for col in df.columns:
    if df[col].dtype == "object":
      try:
        cleaned = df[col].str.replace(",", ".", regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors="raise")
        df[col] = converted
      except ValueError:
        pass

  minimo_data_numerica = df["Data"].apply(lambda x: x.toordinal()).min()
  df["Data_num"] = df["Data"].apply(lambda x: x.toordinal() - minimo_data_numerica)
  df = df.copy()
  return df


def make_lags(ts, lags, lead_time=1, name='y'):
  return pd.concat(
    {
      f'{name}_lag_{i}': ts.shift(i)
      for i in range(lead_time, lags + lead_time)
    },
    axis=1)


def make_leads(ts, leads, name='y'):
  return pd.concat(
    {
      f'{name}_lead_{i}': ts.shift(-i)
      for i in reversed(range(leads))
    },
    axis=1)


def wasserstein_row_error(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
    
  row_distances = [wasserstein_distance(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
  return np.mean(row_distances)


def get_wesserstein_error(y_train, y_valid, y_fit_train, y_fit_valid):
  use_row_error = False
  if isinstance(y_fit_train, pd.Series):
    use_row_error = False
  elif isinstance(y_fit_train, pd.DataFrame):
    use_row_error = len(y_fit_train.columns) > 1
  else:
    raise RuntimeError("Objeto deve ser Série ou Dataframe")
  if use_row_error:
    return [
      wasserstein_row_error(y_train, y_fit_train), 
      wasserstein_row_error(y_valid, y_fit_valid)
    ]
  else:
    return [
      wasserstein_distance(y_train, y_fit_train),
      wasserstein_distance(y_valid, y_fit_valid)
    ]


def calculate_metrics(y_train, y_valid, y_fit_train, y_fit_valid, label=None):
  metrics = {
    'RMSE': [root_mean_squared_error(y_train, y_fit_train), root_mean_squared_error(y_valid, y_fit_valid)],
    'MSE': [mean_squared_error(y_train, y_fit_train), mean_squared_error(y_valid, y_fit_valid)],
    'MAE': [mean_absolute_error(y_train, y_fit_train), mean_absolute_error(y_valid, y_fit_valid)],
    #'R2': [r2_score(y_train, y_fit_train), r2_score(y_valid, y_fit_valid)],
    #'Wasserstein': get_wesserstein_error(y_train, y_valid, y_fit_train, y_fit_valid)
  }

  metrics_df = pd.DataFrame(metrics, index=['Treino', 'Validação'])
  if label is not None:
    metrics_df.index = pd.MultiIndex.from_product([[label], metrics_df.index], names=['Modelo', 'Dados'])
  return metrics_df


class HybridBase:
  """ Classe Base """
  def __init__(self, base_model_trend, base_model_cycle, lags=1):
    self.base_model_trend = base_model_trend
    self.base_model_cycle = base_model_cycle
    self.lags = lags
    self.y_columns = None

  def decompose(self, X, y):
    """ Treinar Modelo para tendência e retornar resíduos."""
    self.trend_model = clone(self.base_model_trend)
    self.trend_model.fit(X, y)
    y_fit = pd.DataFrame(
      self.trend_model.predict(X),
      index=y.index,
      columns=y.columns
    )
    resid = y - y_fit
    self.y_fit = y_fit
    return y_fit, resid
  
  def update(self, resid_new: pd.DataFrame | pd.Series):
    """
    Atualiza estado interno
    """
    if isinstance(resid_new, pd.Series):
      resid_new = resid_new.to_frame(name=self.resid_full.columns[0])

    col = self.resid_full.columns[0]

    # Garante que os índices existem no histórico
    missing_idx = resid_new.index.difference(self.resid_full.index)
    if len(missing_idx) > 0:
      self.resid_full = self.resid_full.reindex(
        self.resid_full.index.union(missing_idx)
      ).sort_index()

    self.resid_full.loc[resid_new.index, col] = resid_new[col]

    self.last_index = self.resid_full.index.max()

  def calculate_lags(self, X_future):
    """ Calcular residuos futuros utilizando ciclo"""
    if(self.resid_full is None):
      raise RuntimeError("Treinar o modelo primeiro")
    while(X_future.index.max() not in self.resid_full.index):
      next_date = self.resid_full.index[-1] + pd.Timedelta(days=1)
      resid_column_name = self.resid_full.columns[0]
      new_row = pd.DataFrame({resid_column_name: [0.0]}, index=[next_date])

      new_residuals = pd.concat([self.resid_full, new_row])
      new_residuals.index.name = self.resid_full.index.name
      new_lags = make_lags((new_residuals).squeeze(), self.lags)

      last_row = new_lags.tail(1)
      new_resid = self.cycle_model.predict(last_row).ravel()
      cycle_predicted = pd.DataFrame(
        [{resid_column_name: x} for x in new_resid],
        index=[next_date + i* pd.Timedelta(days=1) for i, _ in enumerate(new_resid)]
      )
      missing = cycle_predicted.index.difference(self.resid_full.index)

      self.resid_full = self.resid_full.reindex(
        self.resid_full.index.union(missing)
      )
      self.resid_full.loc[cycle_predicted.index, resid_column_name] = cycle_predicted.loc[cycle_predicted.index, resid_column_name]


class HybridRecursive(HybridBase):
  """ Estratégia recursiva """
  def fit(self, X, y):
    self.y_columns = [y.name]
    self.y_target_name = self.y_columns[0]

    y_fit, resid = self.decompose(X, y.to_frame())
    resid_full = resid.copy()
    self.resid_full = resid_full
    X_cycle = make_lags(resid_full.squeeze(), self.lags).dropna()
    y_cycle = resid.loc[X_cycle.index]

    self.cycle_model = clone(self.base_model_cycle)
    self.cycle_model.fit(X_cycle, y_cycle)
    
    self.last_index = y.index[-1]
    self.X_cycle_columns = X_cycle.columns

  def predict(self, X_future):
    """Recursive forecast for given future regressors (X_future)."""
    y_trend_pred = pd.DataFrame(
      self.trend_model.predict(X_future),
      columns=[self.y_target_name],
      index=X_future.index
    )

    all_lags = make_lags((self.resid_full).squeeze(), self.lags).fillna(0.0)
    all_lags = all_lags.loc[X_future.index]
    y_cycle_pred = pd.DataFrame(
      self.cycle_model.predict(all_lags),
      columns=self.y_columns,
      index=all_lags.index
    )
    result = y_trend_pred.add(y_cycle_pred, fill_value=0)
    return result
  

def plot_plotly(y, y_forecast, title=None, showlegend=False, hover=False):
  palette_kwargs_ = dict(palette='husl', n_colors=64, desat=None)
  palette = sns.color_palette(**palette_kwargs_)
  rand_number = np.random.randint(0, 99999999)
  print(rand_number)
  rgb = tuple(int(c * 255) for c in palette[rand_number % len(palette)])
  fig = go.Figure()
  fig.add_trace(go.Scatter(
    x=y.index,
    y=y.values.flatten() if isinstance(y, pd.DataFrame) else y,
    mode='markers+lines',
    name='Observado',
    line=dict(color='rgba(0, 0, 0, 0.4)', width=2),
    marker=dict(size=5)
  ))
  fig.add_trace(go.Scatter(
    x=y_forecast.index,
    y=y_forecast.values.flatten() if isinstance(y_forecast, pd.DataFrame) else y_forecast,
    mode='lines',
    line=dict(color=f'rgb{rgb}', width=1.5),
    name=f'Forecast'
  ))
  fig.update_layout(
    title=title or "Multistep Forecast",
    xaxis_title="Data",
    yaxis_title="Valor",
    hovermode='x unified' if hover else False,
    template='plotly_white',
    showlegend=showlegend,
    height=500
  )
  fig.show()


class AutoregressiveEngine:
  def __init__(self, modelo_volume):
    self.modelo = modelo_volume
    self.modelo_volume = self.modelo.modelo_volume
    self.vol_history = None
    self.X_feat_hist = None

  def fit(self, X, y):
    self.vol_history = self.modelo.vol_history.copy()
    self.X_feat_hist = self.modelo.X_feat_hist.copy()

  def get_X_exog_fore(self, X_future, new_row):
    try:
      X_exog_fore = X_future.loc[new_row.index]
      X_exog_fore.index.name = X_future.index.name
      return X_exog_fore
    except KeyError:
      data = new_row.index.astype(str).to_list()[0]
      raise RuntimeError("A data "+data+" deve estar presente do dataframe para previsão.")
    
  def get_X_to_forecast(self, last_row_lag, X_exog_fore):
    # Previsão Vazão de Entrada
    y_vn_fore = self.modelo.modelo_vazao_natural.predict(X_exog_fore)

    y_vn_fore = pd.Series(y_vn_fore.values.ravel(), index=y_vn_fore.index)
    y_vn_fore.index.name = self.vol_history.index.name
    X_leads_Qin = make_leads(y_vn_fore, 1, name="Qin")

    if(self.modelo.usar_jusante):
      y_vj_fore = self.modelo.modelo_vazao_jusante.predict(X_exog_fore)
      y_vj_fore = pd.Series(y_vj_fore.values.ravel(), index=y_vj_fore.index)
      y_vj_fore.index.name = self.vol_history.index.name
      X_leads_Qout = make_leads(y_vj_fore, 1, name="Qout")
      X_fore = pd.concat([last_row_lag, X_leads_Qin, X_leads_Qout], axis=1).dropna()
    else:
      X_fore = pd.concat([last_row_lag, X_leads_Qin], axis=1).dropna()
    
    self.X_feat_hist = pd.concat([self.X_feat_hist, X_fore])
    return X_fore

  def step(self, X_future, step_date):
    """
    Executa exatamente UM passo autoregressivo.
    """
    vol_column_name = self.vol_history.columns[0]

    # Cria linha dummy
    new_row = pd.DataFrame(
      {vol_column_name: [0.0]},
      index=[step_date]
    )

    # Exógenas do futuro
    X_exog_fore = self.get_X_exog_fore(X_future, new_row)

    # Atualiza histórico
    new_hist = pd.concat([self.vol_history, new_row])
    new_hist.index.name = self.vol_history.index.name

    # Calcula lags
    new_lags = make_lags(new_hist.squeeze(), 1)
    last_row_lag = new_lags.tail(1)

    # Monta features completas (lags + leads)
    X_fore = self.get_X_to_forecast(last_row_lag, X_exog_fore)

    # Previsão do volume
    y_new_volume = self.modelo_volume.predict(X_fore)

    # Atualiza histórico de volumes
    self.vol_history.loc[step_date, vol_column_name] = y_new_volume.ravel()[0]

    return y_new_volume.ravel()[0]

  def get_features(self, X_future):
    return self.X_feat_hist.loc[X_future.index]

  def predict(self, X_future):
    """
    Executa previsão autoregressiva completa.
    """
    if self.vol_history is None:
      raise RuntimeError("Engine não treinada")

    preds = []

    for timestamp in X_future.index:
      if timestamp not in self.vol_history.index:
        y_hat = self.step(X_future, timestamp)
      else:
        X_to_predict = self.get_features(X_future)
        X_row = X_to_predict.loc[[timestamp]]
        y_hat = self.modelo_volume.predict(X_row)[0]

      preds.append(y_hat)
    
    return pd.Series(preds, index=X_future.index)

  def update(self, X_vol_new, vol_new, vn_new, vj_new):
    """
    Atualiza o histórico usando valores reais observados.
    Usado após validação, antes da previsão futura.
    """
    self.modelo.modelo_vazao_natural.update(vn_new)
    if self.modelo.usar_jusante:
      self.modelo.modelo_vazao_jusante.update(vj_new)
    if isinstance(vol_new, pd.Series):
      vol_new = vol_new.to_frame(name=self.vol_history.columns[0])
    col = self.vol_history.columns[0]

    # Garante que os índices existem no histórico
    missing_idx = vol_new.index.difference(self.vol_history.index)
    if len(missing_idx) > 0:
      self.vol_history = self.vol_history.reindex(
        self.vol_history.index.union(missing_idx)
      ).sort_index()

    # SOBRESCREVE resíduos previstos com resíduos reais
    self.vol_history.loc[vol_new.index, col] = vol_new[col]

    self.X_feat_hist.loc[vol_new.index] = X_vol_new

    self.last_index = self.vol_history.index.max()


class ModeloPrevisaoVolume():
  def __init__(self, base_model):
    self.base_model = clone(base_model)
    self.usar_jusante = False
    self.engine = None

  """ Calcula lags para variáveis exógenas """
  def calculate_lags(self, X_valid_rec):
    self.modelo_vazao_natural.calculate_lags(X_valid_rec)
    if(self.usar_jusante):
      self.modelo_vazao_jusante.calculate_lags(X_valid_rec)

  """ Treinamento para Vazão Natural """
  def fit_vazao_natural(self, X, y):
    # Lasso/KNeighborsRegressor
    self.modelo_vazao_natural = HybridRecursive(Lasso(), KNeighborsRegressor(), lags=2)
    self.modelo_vazao_natural.fit(X, y)

  """ Treinamento para Vazão Jusante """
  def fit_vazao_jusante(self, X, y):
    self.usar_jusante = True
    # LinearRegression/RandomForestRegressor
    self.modelo_vazao_jusante = HybridRecursive(LinearRegression(), RandomForestRegressor(), lags=2)
    self.modelo_vazao_jusante.fit(X, y)

  def fit(self, X, y):
    y_to_fit = y.copy() if isinstance(y, pd.DataFrame) else y.to_frame().copy()

    self.vol_history = y_to_fit[y_to_fit.columns[0]].to_frame()
    self.y_columns = y_to_fit.columns

    self.modelo_volume = self.base_model

    if self.usar_jusante:
      self.modelo_volume.fit(X, y)
      self.X_feat_hist = X
    else:
      X_sem_jusante = X[[x for x in X.columns if not x.startswith("Qout_")]]
      self.modelo_volume.fit(X_sem_jusante, y)
      self.X_feat_hist = X_sem_jusante

    # Agora entra a engine
    self.engine = AutoregressiveEngine(self)
    self.engine.fit(X, y)

  def predict(self, X_future):
    return self.engine.predict(X_future)

  def update(self, X_vol_new, vol_new, vn_new, vj_new):
    if self.usar_jusante:
      self.X_feat_hist = X_vol_new
    else:
      X_sem_jusante = X_vol_new[[x for x in X_vol_new.columns if not x.startswith("Qout_")]]
      self.X_feat_hist = X_sem_jusante
    self.engine.update(self.X_feat_hist, vol_new, vn_new, vj_new)