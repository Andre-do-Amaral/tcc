import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 6)
import pandas as pd
import seaborn as sns

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

from sklearn.base import clone
from sklearn.multioutput import RegressorChain

import plotly.graph_objects as go


def import_dataframe():
  df = pd.read_csv('../data/tab_cantareira.csv', sep=';', parse_dates=['Data'], dayfirst=True)
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
    'MSE': [mean_squared_error(y_train, y_fit_train), mean_squared_error(y_valid, y_fit_valid)],
    'MAE': [mean_absolute_error(y_train, y_fit_train), mean_absolute_error(y_valid, y_fit_valid)],
    'R2': [r2_score(y_train, y_fit_train), r2_score(y_valid, y_fit_valid)],
    'Wasserstein': get_wesserstein_error(y_train, y_valid, y_fit_train, y_fit_valid)
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


class HybridDirect(HybridBase):
  """ Estratégia direta """
  def fit(self, X, y):
    self.y_columns = y.columns
    self.y_target_name = self.y_columns[0]
    self.horizon = len(y.columns)

    _, resid = self.decompose(X, y[self.y_target_name].to_frame())
    resid_full = resid.copy()
    X_cycle_full = make_lags(resid_full.squeeze(), self.lags).dropna()
    self.X_cycle_full = X_cycle_full
    self.X_cycle_columns = X_cycle_full.columns

    resid_y_fore = make_multistep_target(resid_full.squeeze(), steps=self.horizon).dropna()

    self.cycle_model = clone(self.base_model_cycle)
    y_cycle = resid_y_fore.loc[resid_y_fore.index.intersection(X_cycle_full.index)]
    X_cycle_fore = X_cycle_full.loc[resid_y_fore.index.intersection(X_cycle_full.index)]
    self.cycle_model.fit(X_cycle_fore, y_cycle)

    self.resid_full = resid_full
    self.X_cycle = X_cycle_full

  def predict(self, X_future):
    y_trend_pred = pd.DataFrame(
      self.trend_model.predict(X_future),
      columns=[self.y_target_name],
      index=X_future.index
    )
    values = [y_trend_pred.shift(-i) for i in range(self.horizon)]
    preds_trend = pd.concat(values, axis=1).dropna()
    preds_trend.columns = self.y_columns

    all_lags = make_lags((self.resid_full).squeeze(), self.lags).fillna(0.0)
    all_lags = all_lags.loc[X_future.index]
    y_cycle_pred = pd.DataFrame(
      self.cycle_model.predict(all_lags),
      columns=self.y_columns,
      index=all_lags.index
    )
    
    y_cycle_pred = y_cycle_pred.loc[preds_trend.index]
    
    self.x1 = X_future
    self.x2 = all_lags
    self.result1 = preds_trend
    self.result2 = y_cycle_pred

    result = preds_trend.add(y_cycle_pred, fill_value=0)
    return result


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
    
    self.resid_hist = resid.iloc[-self.lags:].to_numpy().ravel()
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