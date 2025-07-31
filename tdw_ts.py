from __future__ import annotations
import os, sys, time
from contextlib import contextmanager
from functools   import reduce

import numpy  as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf_yw
from pybuc.buc import BayesianUnobservedComponents
from prophet import Prophet


# ───────────────── suppress stdout ───────────────── #
@contextmanager
def _silence():
    devnull, old = open(os.devnull, "w"), (sys.stdout, sys.stderr)
    sys.stdout = sys.stderr = devnull
    try:    yield
    finally:
        sys.stdout, sys.stderr = old
        devnull.close()


# ───────────────── helpers ───────────────── #
def _safe_log1p(x: pd.Series | np.ndarray, off: float) -> np.ndarray:
    return np.log1p(x + off)


def _safe_expm1(x: np.ndarray, off: float) -> np.ndarray:
    res = np.expm1(x) - off
    res[~np.isfinite(res)] = 0
    return res


def _scale_series(s: pd.Series, mean: float, std: float) -> pd.Series:
    std = std or 1.0
    return (s - mean) / std


# ───────────────── выбросы ───────────────── #
def _remove_outliers(y: pd.Series, k: float = 5) -> pd.Series:
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    with _silence():
        model = Prophet(yearly_seasonality=True).fit(df)
    resid = df['y'] - model.predict(df)['yhat']
    df.loc[np.abs(resid) > k * resid.std(), 'y'] -= resid[np.abs(resid) > k * resid.std()]
    return df.set_index('ds')['y']


# ───────────────── регрессор «структурный сдвиг» ───────────────── #
def _shift_regressor(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    # позитивная лин. регрессия описывает тренд -> остаток → change-points
    model = sm.LinearRegression(positive=True).fit(X.values, y.values)
    resid = y.values - model.predict(X.values)

    bkps = rpt.Binseg(model='l2').fit(resid).predict(pen=7_000)[:-1]  # точки разбиений
    L, ext = len(resid), 31
    mask = np.zeros(L + ext, bool)
    segs = [0] + bkps + [L + ext]
    for s, e in zip(segs[::2], segs[1::2]):
        mask[s:e] = True
    index_ext = pd.date_range(y.index[0], periods=L + ext, freq='D')
    return pd.Series(~mask, index=index_ext)


# ───────────────── авто-SARIMA ───────────────── #
def _acf_pacf(series: pd.Series, nlags=50, alpha=0.05, step=1) -> tuple[int, int]:
    n = len(series)
    z = abs(np.percentile(np.random.normal(size=100_000), 100 * (1-alpha/2)))
    conf = z / np.sqrt(n)
    p = (abs(pacf_yw(series, nlags, method='mle')[::step]) > conf).sum() - 1
    q = (abs(acf    (series, nlags)[::step])       > conf).sum() - 1
    return max(0, min(6, p)), max(0, min(6, q))


def _stationary(x: pd.Series) -> bool:
    from statsmodels.tsa.stattools import adfuller, kpss
    if x.empty: return False
    return adfuller(x.dropna())[1] < .05 and kpss(x.dropna(), 'c')[1] > .05


def _sarima_orders(y: pd.Series) -> tuple[tuple,int,int,int, tuple]:
    d = 1 if _stationary(y.diff())  else 2
    D = 1 if _stationary(y.diff(7)) else 2
    p, q = _acf_pacf(y.diff(d).dropna())     if d else _acf_pacf(y)
    P, Q = _acf_pacf(y.diff(7*D).dropna(), step=7)
    return (p, d, q), (P, D, Q, 7)


# ────────────────── базовые модели ────────────────── #
def _sarima(y: pd.Series, n: int,
            order, sorder, exog: pd.DataFrame | None = None) -> pd.Series:
    m = sm.tsa.statespace.SARIMAX(y, exog=exog, order=order,
                                  seasonal_order=sorder,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
    idx = pd.date_range(y.index.max()+pd.Timedelta(days=1), periods=n, freq='D')
    x_fut = exog.reindex(idx) if exog is not None else None
    return pd.Series(m.get_forecast(n, x_fut).predicted_mean, idx)


def _prophet(y: pd.Series, n: int, exog: pd.DataFrame | None = None) -> pd.Series:
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    with _silence():
        m = Prophet(weekly_seasonality=True, mcmc_samples=800)
        if exog is not None:
            for c in exog.columns:
                m.add_regressor(c)
                df[c] = exog[c].values
        m.fit(df)
        fut = m.make_future_dataframe(n, freq='D')
        if exog is not None:
            for c in exog.columns:
                fut[c] = exog.reindex(fut['ds'])[c].values
        return m.predict(fut).set_index('ds')['yhat'].tail(n)


def _bsts(y: pd.Series, n: int, exog: pd.DataFrame | None = None) -> pd.Series:
    Xc = sm.add_constant(exog) if exog is not None else sm.add_constant(pd.DataFrame(index=y.index))
    ols = sm.OLS(y, Xc).fit()
    resid = y - ols.predict(Xc)
    uc = BayesianUnobservedComponents(
        resid, level=True, stochastic_level=True,
        trend=True, stochastic_trend=True,
        trig_seasonal=((7, 0),), stochastic_trig_seasonal=(True,), seed=123)
    uc.sample(10_000)
    idx = pd.date_range(y.index.max()+pd.Timedelta(days=1), periods=n, freq='D')
    _, draws = uc.forecast(n, burn=1_000)
    resid_fc = draws.mean(axis=0).squeeze()
    Xf = sm.add_constant(exog.reindex(idx) if exog is not None else pd.DataFrame(index=idx))
    return pd.Series(ols.predict(Xf) + resid_fc, idx)


# ─────────────────── основной класс ──────────────────── #
class UniversalEnsemble:
    """Ансамбль из SARIMA, Prophet, BSTS с произвольным набором X-фичей."""

    def __init__(self,
                 y: pd.Series,
                 X: pd.DataFrame | dict[str, pd.Series] | None = None,
                 *,
                 remove_outliers: bool = True,
                 detect_shifts:   bool = True):

        # приводим X к DataFrame
        if X is None:
            X = pd.DataFrame(index=y.index)
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        # гарантируем синхронизацию по датам
        X = X.reindex(y.index)

        # 1) очистка выбросов
        self.y = _remove_outliers(y) if remove_outliers else y.copy()
        # 2) экзогенные
        self.X = X.copy()
        # 3) регрессор «сдвиг»
        self.shift = (_shift_regressor(self.y, self.X)
                      if detect_shifts else
                      pd.Series(False, index=self.y.index))
        # 4) auto-SARIMA параметры
        self.order, self.sorder = _sarima_orders(self.y)

        self.scalers: dict[str, tuple[float,float]] = {}
        self.weights: dict[str, float] = {}
        self.use_log  = True
        self.offset   = 0
        self.n_future = 0
        self.errs     = {}

    # ──────────────── подготовка X (масштаб, fill) ──────────────── #
    @staticmethod
    def _prep_X(X: pd.DataFrame, idx: pd.Index,
                scalers: dict[str, tuple[float,float]] | None = None
                ) -> pd.DataFrame:
        X = X.reindex(idx).fillna(method='ffill').fillna(0)
        if scalers is None:  # обучение – формируем статистики
            scalers = {c: (X[c].mean(), X[c].std() or 1) for c in X}
        Xs = pd.DataFrame({c: _scale_series(X[c], *scalers[c]) for c in X}, index=idx)
        return Xs, scalers

    # ─────────────── back-testing для расчёта весов ─────────────── #
    def fit(self, *, n_future: int = 31, n_splits: int = 3, alpha: float = .5,
            use_log: bool = True) -> None:

        self.n_future, self.use_log = n_future, use_log
        # общий индекс
        idx_all = reduce(pd.Index.intersection,
                         [self.y.dropna().index,
                          self.X.dropna(how='all').index])
        y = self.y.loc[idx_all]
        X = self.X.loc[idx_all].copy()
        # добавляем булев флаг сдвига
        X['shift'] = self.shift.reindex(idx_all).astype(int)

        errs = {m: [] for m in ('sarima', 'prophet', 'bsts')}

        for k in range(1, n_splits+1):
            tr, te = -n_future*k, -n_future*(k-1) or None
            y_tr, y_te = y.iloc[:tr], y.iloc[tr:te]
            X_tr, X_te = X.iloc[:tr], X.iloc[tr:te]

            off = abs(y_tr.min()) + 1 if use_log else 0
            y_tr_t = _safe_log1p(y_tr, off) if use_log else y_tr

            # scale на train-части
            X_tr_s, scalers = self._prep_X(X_tr, X_tr.index)
            X_te_s, _       = self._prep_X(X_te, X_te.index, scalers)

            # ── модели
            sar = _sarima(y_tr_t, n_future, self.order, self.sorder, X_tr_s)
            prp = _prophet(y_tr_t, n_future, X_tr_s)
            bts = _bsts  (y_tr_t, n_future, X_tr_s)

            preds = {'sarima': sar, 'prophet': prp, 'bsts': bts}
            if use_log:
                preds = {m: _safe_expm1(v, off) for m, v in preds.items()}

            y_true_sum = y_te.sum()
            for m in errs:
                errs[m].append((y_true_sum - preds[m].sum())**2)

        w_fold = np.array([alpha*(1-alpha)**i for i in range(n_splits)])
        w_fold /= w_fold.sum()
        inv = {m: 1/np.dot(w_fold, errs[m]) for m in errs}
        s = sum(inv.values())
        self.weights = {m: inv[m]/s for m in inv}
        self.errs = errs
        # финальные скейлеры по всей истории
        self.X_scaled, self.scalers = self._prep_X(X, X.index)
        self.offset = off

    # ─────────────── итоговый прогноз ─────────────── #
    def predict(self, n_future: int | None = None) -> pd.DataFrame:
        if not self.weights:
            raise RuntimeError('Сначала вызовите fit()')

        n_future = n_future or self.n_future
        idx_fc = pd.date_range(self.y.index.max()+pd.Timedelta(days=1),
                               periods=n_future, freq='D')

        # подготавливаем всю X + масштабир. будущего
        X_hist_scaled = self.X_scaled
        X_future_raw  = self.X.reindex(idx_fc).copy()
        X_future_raw['shift'] = self.shift.reindex(idx_fc).astype(int)
        X_future_scaled, _ = self._prep_X(X_future_raw, idx_fc, self.scalers)

        y_t = _safe_log1p(self.y, self.offset) if self.use_log else self.y

        sar = _sarima(y_t, n_future, self.order, self.sorder, X_hist_scaled)
        prp = _prophet(y_t, n_future, X_hist_scaled)
        bts = _bsts  (y_t, n_future, X_hist_scaled)

        preds_t = {'sarima': sar.values,
                   'prophet': prp.values,
                   'bsts': bts.values}
        preds = {m: _safe_expm1(v, self.offset) if self.use_log else v
                 for m, v in preds_t.items()}

        ensemble = sum(self.weights[m]*preds[m] for m in preds)
        return pd.DataFrame({**preds, 'ensemble': ensemble}, index=idx_fc)

    # ─────────────── небольшая печать ─────────────── #
    def summary(self):
        print('Weights:', {k: round(v,3) for k,v in self.weights.items()})
        print('SARIMA order', self.order, 'seasonal', self.sorder)
        print('CV MSE folds:', self.errs)
