import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 6)
import seaborn as sns
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

from sklearn.base import clone
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


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
  df = df.copy() # Para evitar dataframe com memória fragmentada (problema relatado na warning abaixo)
  return df


def seasonal_plot(X, y, period, freq, ax=None):
  if ax is None:
    _, ax = plt.subplots()
  palette = sns.color_palette(
    "husl",
    n_colors=X[period].nunique(),
  )
  ax = sns.lineplot(
    x=freq,
    y=y,
    hue=period,
    data=X,
    errorbar=('ci', False),
    ax=ax,
    palette=palette,
    legend=False,
  )
  ax.set_title(f"Seasonal Plot ({period}/{freq})")
  for line, name in zip(ax.lines, X[period].unique()):
    y_ = line.get_ydata()[-1]
    ax.annotate(
      name,
      xy=(1, y_),
      xytext=(6, 0),
      color=line.get_color(),
      xycoords=ax.get_yaxis_transform(),
      textcoords="offset points",
      size=10,
      va="center",
    )
  return ax


def plot_periodogram(ts, detrend='linear', ax=None):
  from scipy.signal import periodogram
  fs = pd.Timedelta("365D") / pd.Timedelta("1D")
  freqencies, spectrum = periodogram(
    ts,
    fs=fs,
    detrend=detrend,
    window="boxcar",
    scaling='spectrum',
  )
  if ax is None:
    _, ax = plt.subplots()
  ax.step(freqencies, spectrum, color="purple")
  ax.set_xscale("log")
  ax.set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 12, 26, 52, 104])
  ax.set_xticklabels(
    [
      "Hexadecanual (0.0625)",
      "Octoanual (0.125)",
      "Tetraanual (0.25)",
      "Bienal (0.5)",
      "Anual (1)",
      "Semestral (2)",
      "Trimestral (4)",
      "Bimestral (6)",
      "Mensal (12)",
      "Bisemanal (26)",
      "Semanal (52)",
      "Semisemanal (104)",
    ],
    rotation=30,
  )
  ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
  ax.set_ylabel("Variance")
  ax.set_title("Periodogram")
  return ax


def lagplot(x, y=None, shift=1, standardize=False, ax=None, **kwargs):
  from matplotlib.offsetbox import AnchoredText
  import statsmodels.api as sm
  x_ = x.shift(shift)
  if standardize:
    x_ = (x_ - x_.mean()) / x_.std()
  if y is not None:
    y_ = (y - y.mean()) / y.std() if standardize else y
  else:
    y_ = x
  corr = y_.corr(x_)
  if ax is None:
    fig, ax = plt.subplots()
  scatter_kws = dict(
    alpha=0.75,
    s=3,
  )
  line_kws = dict(color='C3', )
  ax = sns.regplot(
    x=x_,
    y=y_,
    scatter_kws=scatter_kws,
    line_kws=line_kws,
    lowess=False,
    ax=ax,
    **kwargs)
  at = AnchoredText(
    f"{corr:.2f}",
    prop=dict(size="large"),
    frameon=True,
    loc="upper left",
  )
  at.patch.set_boxstyle("square, pad=0.0")
  ax.add_artist(at)
  title = f"Lag {shift}" if shift > 0 else f"Lead {shift}"
  ax.set(title=f"Lag {shift}", xlabel=x_.name, ylabel=y_.name)
  return ax


def plot_lags(
    x,
    y=None,
    lags=6,
    leads=None,
    nrows=1,
    lagplot_kwargs={},
    **kwargs):
  import math
  kwargs.setdefault('nrows', nrows)
  orig = leads is not None
  leads = leads or 0
  kwargs.setdefault('ncols', math.ceil((lags + orig + leads) / nrows))
  kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
  fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
  for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
    k -= leads + orig
    if k + 1 <= lags:
      ax = lagplot(x, y, shift=k + 1, ax=ax, **lagplot_kwargs)
      title = f"Lag {k + 1}" if k + 1 >= 0 else f"Lead {-k - 1}"
      ax.set_title(title, fontdict=dict(fontsize=14))
      ax.set(xlabel="", ylabel="")
    else:
      ax.axis('off')
  plt.setp(axs[-1, :], xlabel=x.name)
  plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
  fig.tight_layout(w_pad=0.1, h_pad=0.1)
  return fig


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


def make_multistep_target(ts, steps, reverse=False):
  shifts = reversed(range(steps)) if reverse else range(steps)
  return pd.concat({f'y_step_{i + 1}': ts.shift(-i) for i in shifts}, axis=1)


def print_equation(model):
  eq_text = f"y = {model.intercept_:.5f} + "
  eq_text += " + ".join([f"{c:.5f}·X{i+1}" for i, c in enumerate(model.coef_)])
  return eq_text


plot_params = {
  'color': '0.75',
  'style': '.-',
  'markeredgecolor': '0.25',
  'markerfacecolor': '0.25',
  'legend': False
}


def plot_linear_model(model, X, y, title="Afluência"):
  y_pred = pd.Series(model.predict(X), index=X.index, name=y.to_frame().columns[0])
  ax = y.plot(**plot_params, alpha=0.5, title=title, ylabel="vazão (m³/s)")
  ax = y_pred.plot(ax=ax, linewidth=3, label="Tendência", color='C0')

  eq_text = print_equation(model)
  ax.text(
    0.05, 0.95, eq_text,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment='top'
  )
  ax.legend();


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
    #'MSE': [mean_squared_error(y_train, y_fit_train), mean_squared_error(y_valid, y_fit_valid)],
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


class HybridDirect(HybridBase):
  """ Estratégia direta """
  def fit(self, X, y):
    self.y_columns = y.columns
    self.y_target_name = self.y_columns[0]
    self.horizon = len(y.columns)

    _, resid = self.decompose(X, y[self.y_target_name].to_frame())
    resid_full = resid.copy()

    self.trend_model = MultiOutputRegressor(clone(self.base_model_trend))
    self.trend_model.fit(X, y)

    X_cycle_full = make_lags(resid_full.squeeze(), self.lags).dropna()
    resid_y_fore = make_multistep_target(resid_full.squeeze(), steps=self.horizon).dropna()

    idx = resid_y_fore.index.intersection(X_cycle_full.index)
    y_cycle = resid_y_fore.loc[idx]
    X_cycle = X_cycle_full.loc[idx]

    self.cycle_model = MultiOutputRegressor(clone(self.base_model_cycle))
    self.cycle_model.fit(X_cycle, y_cycle)

    self.resid_full = resid_full
    self.X_cycle_columns = X_cycle_full.columns

  def predict(self, X_future):
    preds_trend = pd.DataFrame(
      self.trend_model.predict(X_future),
      index=X_future.index,
      columns=self.y_columns
    )

    all_lags = make_lags(self.resid_full.squeeze(), self.lags).fillna(0.0)
    all_lags = all_lags.loc[X_future.index]
    preds_cycle = pd.DataFrame(
      self.cycle_model.predict(all_lags),
      index=all_lags.index,
      columns=self.y_columns
    )

    result = preds_trend.add(preds_cycle, fill_value=0.0)
    return result


class HybridDirRec(HybridBase):
  def fit(self, X, y):
    self.y_columns = y.columns
    self.y_target_name = self.y_columns[0]
    self.horizon = len(y.columns)

    _, resid = self.decompose(X, y[self.y_target_name].to_frame())
    self.resid_full = resid.copy()

    X_cycle_full = make_lags(self.resid_full.squeeze(), self.lags).dropna()
    resid_y_fore = make_multistep_target(self.resid_full.squeeze(), steps=self.horizon).dropna()

    X_cycle_fore = X_cycle_full.loc[resid_y_fore.index.intersection(X_cycle_full.index)]
    y_cycle = resid_y_fore.loc[resid_y_fore.index.intersection(X_cycle_full.index)]

    base_model = clone(self.base_model_cycle)
    self.cycle_model = RegressorChain(base_model)
    self.cycle_model.fit(X_cycle_fore, y_cycle)

    self.trend_model.fit(X, y[self.y_target_name])

  def predict(self, X_future):
    y_trend_pred = pd.DataFrame(
      self.trend_model.predict(X_future),
      columns=[self.y_target_name],
      index=X_future.index
    )
    values = [y_trend_pred.shift(-i) for i in range(self.horizon)]
    preds_trend = pd.concat(values, axis=1).dropna()
    preds_trend.columns = self.y_columns

    all_lags = make_lags(self.resid_full.squeeze(), self.lags).fillna(0.0)
    all_lags = all_lags.loc[X_future.index]

    y_cycle_pred = pd.DataFrame(
      self.cycle_model.predict(all_lags),
      columns=self.y_columns,
      index=all_lags.index
    )
    y_cycle_pred = y_cycle_pred.loc[preds_trend.index]

    return preds_trend.add(y_cycle_pred, fill_value=0)
  

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


def plot_multistep_plotly(y, y_forecast, every=1, palette_kwargs=None, title=None, showlegend=False, hover=False):
  palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
  if palette_kwargs is not None:
    palette_kwargs_.update(palette_kwargs)
  palette = sns.color_palette(**palette_kwargs_)

  fig = go.Figure()

  fig.add_trace(go.Scatter(
    x=y.index,
    y=y.values.flatten() if isinstance(y, pd.DataFrame) else y,
    mode='markers+lines',
    name='Observado',
    line=dict(color='rgba(0, 0, 0, 0.4)', width=2),
    marker=dict(size=5)
  ))

  for i, (date, preds) in enumerate(y_forecast[::every].iterrows()):
    preds.index = pd.period_range(start=date, periods=len(preds))
    preds = preds.to_timestamp()
    rgb = tuple(int(c * 255) for c in palette[i % len(palette)])
    fig.add_trace(go.Scatter(
      x=preds.index,
      y=preds.values,
      mode='lines',
      line=dict(color=f'rgb{rgb}', width=1.5),
      name=f'Forecast {date}'
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

  return fig


def get_model_name(model):
  return f'{model}'.split('(')[0]


def highlight_val_only(s):
  if 'Validação' not in s.index.get_level_values('Dados'):
    return [''] * len(s)
  mask = s.index.get_level_values('Dados') == 'Validação'

  if s.name == 'R2':
    best_val = s[mask].max()
    worst_val = s[mask].min()
  else:
    best_val = s[mask].min()
    worst_val = s[mask].max()

  return [
    'background-color: #ff3333; color: white; font-weight: bold;' if v == worst_val and m else
    'background-color: #00cc66; color: white; font-weight: bold;' if v == best_val and m else ''
      for v, m in zip(s, mask)
  ]


def rodar_treino_vazoes_kfold(
  model_name,
  model1,
  model2,
  splitter,
  results_array,
  dados,
  recursivo=None,
  direto=None,
  dir_rec=None
):
  if (recursivo == None
      and direto == None
      and dir_rec == None):
    raise Exception("Uma das estratégias deve estar selecionada (True): recursivo, direto ou dir_rec.")
  
  folds_metrics = []
  X_usado = dados["X_rec"] if recursivo else dados["X_dir"]
  y_usado = dados["y_rec"] if recursivo else dados["y_dir"]
  y_plot = dados["y_rec"]
  for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_usado)):
    X_train_fold = X_usado.iloc[train_idx]
    X_test_fold = X_usado.iloc[test_idx]
    y_train_fold = y_usado.iloc[train_idx]
    y_test_fold = y_usado.iloc[test_idx]
    if recursivo:
      model = HybridRecursive(
        clone(model1),
        clone(model2),
        lags=2
      )
    elif direto:
      model = HybridDirect(
        clone(model1),
        clone(model2),
        lags=2
      )
    elif dir_rec:
      model = HybridDirRec(
        clone(model1),
        clone(model2),
        lags=2
      )
    model.fit(X_train_fold, y_train_fold)
    model.calculate_lags(X_test_fold)

    y_fit = model.predict(X_train_fold)
    y_pred = model.predict(X_test_fold)

    metrics = calculate_metrics(
      y_usado.loc[y_fit.index],
      y_usado.loc[y_pred.index],
      y_fit.squeeze(),
      y_pred.squeeze(),
      label = model_name
    )
    folds_metrics.append(metrics)
  df_folds = pd.concat(folds_metrics)
  df_agg = df_folds.groupby(['Modelo', 'Dados']).agg(['mean', 'std', 'min', 'max'])
  results_array.append(df_agg)
  return df_agg


def rodar_treino_vazoes_view(
  model1,
  model2,
  dados,
  VALIDATION_SIZE=360,
  recursivo=None,
  direto=None,
  dir_rec=None
):
  if (recursivo == None
      and direto == None
      and dir_rec == None):
    raise Exception("Uma das estratégias deve estar selecionada (True): recursivo, direto ou dir_rec.")
  if recursivo:
    model = HybridRecursive(
      clone(model1),
      clone(model2),
      lags=2
    )
  elif direto:
    model = HybridDirect(
      clone(model1),
      clone(model2),
      lags=2
    )
  elif dir_rec:
    model = HybridDirRec(
      clone(model1),
      clone(model2),
      lags=2
    )

  if recursivo:
    X_train, X_valid, y_train, y_valid = train_test_split(dados['X_rec'], dados['y_rec'], test_size=VALIDATION_SIZE, shuffle=False)
  else:
    X_train, X_valid, y_train, y_valid = train_test_split(dados['X_dir'], dados['y_dir'], test_size=VALIDATION_SIZE, shuffle=False)

  model.fit(X_train, y_train)
  model.calculate_lags(X_valid)

  y_fit = model.predict(X_train)
  y_pred = model.predict(X_valid)

  y_view = dados['y_rec']
  if recursivo:
    plot_plotly(
      y_view.loc[y_fit.index],
      y_fit,
      title="Forecast - Treino"
    )

    plot_plotly(
      y_view.loc[y_pred.index],
      y_pred,
      title="Forecast - Validação"
    )
  else:
    fig1 = plot_multistep_plotly(
      y_view.loc[y_fit.index],
      y_fit,
      every=10,
      palette_kwargs=dict(palette='husl', n_colors=64),
      title="Forecast - Treino"
    )
    fig1.show()

    fig2 = plot_multistep_plotly(
      y_view.loc[y_pred.index],
      y_pred,
      every=10,
      palette_kwargs=dict(palette='husl', n_colors=64),
      title="Forecast - Validação"
    )
    fig2.show()


def rodar_treino_volume_kfold(
  model_name,
  model_base_volume,
  model_vn_folds,
  model_vj_folds,
  splitter,
  results_array,
  dados,
  plot = False
):
  folds_metrics = []

  y_usado = dados["y_vol_rec"]  
  for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dados["X_vol_rec"])):
    X_vol_train_fold = dados["X_vol_rec"].iloc[train_idx]
    y_vol_train_fold = dados["y_vol_rec"].iloc[train_idx]
    X_vn_test_fold = dados["X_vn_rec"].iloc[test_idx]

    model = ModeloPrevisaoVolume(
      base_model=model_base_volume,
      usar_jusante=True,
      modelo_vazao_natural=model_vn_folds[fold_idx],
      modelo_vazao_jusante=model_vj_folds[fold_idx]
    )

    model.fit(X_vol_train_fold, y_vol_train_fold)
    X_vn_train_rec_drop = X_vol_train_fold.iloc[1:]
    y_fit = model.predict(X_vn_train_rec_drop)
    y_pred = model.predict(X_vn_test_fold)
    
    metrics = calculate_metrics(
      y_usado.loc[y_fit.index],
      y_usado.loc[y_pred.index],
      y_fit.squeeze(),
      y_pred.squeeze(),
      label = model_name
    )
    folds_metrics.append(metrics)

    if plot:
      fig2 = plot_plotly(
        dados["y_vol_rec"].loc[y_pred.index],
        y_pred,
        title="Forecast - Validação",
        showlegend=True
      )
  df_folds = pd.concat(folds_metrics)
  df_agg = df_folds.groupby(['Modelo', 'Dados']).agg(['mean', 'std', 'min', 'max'])
  results_array.append(df_agg)
  return df_agg


def rodar_treino_volume_view(
  model_base_volume,
  model_vn,
  model_vj,
  dados,
  VALIDATION_SIZE=90
):
  df_volume_series = dados['df_volume_series']
  y_vn = dados['y_vn']
  y_vj = dados['y_vj']
  X_full_vn = dados['X_full_vn']
  X_full_vj = dados['X_full_vj']
  y_vol_rec = dados['y_vol_rec']
  
  y_vol = df_volume_series[df_volume_series.index >= "2018-01-01"].copy()
  X_lags = make_lags(y_vol.squeeze(), 1)
  X_Qin_leads = make_leads(y_vn.squeeze(), 1, name="Qin")
  X_Qout_leads = make_leads(y_vj.squeeze(), 1, name="Qout")
  X_full_vol = pd.concat([X_lags, X_Qin_leads, X_Qout_leads], axis=1).dropna()
  y_vol, X_full_vol = y_vol.align(X_full_vol, join='inner', axis=0)
  X_vol_train_rec, X_vol_valid_rec, y_vol_train_rec, y_vol_valid_rec = train_test_split(
    X_full_vol, y_vol, test_size=VALIDATION_SIZE, shuffle=False)

  X_vn_train_rec, X_vn_valid_rec, y_vn_train_rec, y_vn_valid_rec = train_test_split(
    X_full_vn, y_vn, test_size=VALIDATION_SIZE, shuffle=False)
  X_vj_train_rec, X_vj_valid_rec, y_vj_train_rec, y_vj_valid_rec = train_test_split(
    X_full_vj, y_vj, test_size=VALIDATION_SIZE, shuffle=False)

  model = ModeloPrevisaoVolume(
    base_model=model_base_volume,
    usar_jusante=True,
    modelo_vazao_natural=model_vn,
    modelo_vazao_jusante=model_vj
  )

  model.fit_vazao_natural(X_vn_train_rec, y_vn_train_rec)
  model.fit_vazao_jusante(X_vj_train_rec, y_vj_train_rec)
  model.calculate_lags(X_vn_valid_rec)
  model.fit(X_vol_train_rec, y_vol_train_rec)

  X_vn_train_rec_drop = X_vn_train_rec.iloc[1:]
  y_fit = model.predict(X_vn_train_rec_drop)
  y_pred = model.predict(X_vn_valid_rec)

  fig1 = plot_plotly(
    y_vol_rec.loc[y_fit.index],
    y_fit,
    title="Forecast - Treino",
    showlegend=True
  )

  fig2 = plot_plotly(
    y_vol_rec.loc[y_pred.index],
    y_pred,
    title="Forecast - Validação",
    showlegend=True
  )


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
  def __init__(
    self,
    base_model,
    usar_jusante=False,
    modelo_vazao_natural=HybridRecursive(Lasso(), KNeighborsRegressor(), lags=2),
    modelo_vazao_jusante=HybridRecursive(LinearRegression(), RandomForestRegressor(), lags=2)
  ):
    self.base_model = clone(base_model)
    self.modelo_vazao_natural = modelo_vazao_natural
    self.modelo_vazao_jusante = modelo_vazao_jusante
    self.usar_jusante = usar_jusante
    self.engine = None

  """ Calcula lags para variáveis exógenas """
  def calculate_lags(self, X_valid_rec):
    self.modelo_vazao_natural.calculate_lags(X_valid_rec)
    if(self.usar_jusante):
      self.modelo_vazao_jusante.calculate_lags(X_valid_rec)

  """ Treinamento para Vazão Natural """
  def fit_vazao_natural(self, X, y):
    self.modelo_vazao_natural.fit(X, y)

  """ Treinamento para Vazão Jusante """
  def fit_vazao_jusante(self, X, y):
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