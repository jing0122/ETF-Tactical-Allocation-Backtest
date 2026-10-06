"""
engine.py — 跨資產 ETF 戰術配置系統:核心引擎
=================================================
由期末專題 notebook(main B1143036.ipynb)Part 0 模組化而來,
邏輯與原報告一致;回測與組合迴圈改用 NumPy 陣列運算以加速互動介面。

架構:
    資料層   load_data / build_features
    策略層   strat_A ~ strat_D(含 walk-forward HMM)
    回測層   backtest(交易成本 / no-trade band / 回撤熔斷)、metrics
    組合層   equal / risk_parity / max_sharpe / hrp、combine、efficient_frontier
    延伸     apply_vol_target(波動目標 + 動態槓桿)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
VIX_TICKER = "^VIX"
START = "2005-01-01"
TRADING_DAYS = 252
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
PRICE_CACHE = os.path.join(CACHE_DIR, "etf_prices.parquet")
VIX_CACHE = os.path.join(CACHE_DIR, "vix.parquet")

STRATEGY_INFO = {
    "A": ("Buy & Hold SPY", "被動基準,全期持有 SPY"),
    "B": ("動能輪動", "3/6/12 月混合動能取前 2 名,12 月動能為負則轉現金"),
    "C1": ("月頻 Regime", "SPY 站上 200 日均線且 12 月動能 > 0 才進攻,否則持 TLT/GLD"),
    "C2": ("Regime + VIX 防禦", "C1 加上 VIX 狀態機:VIX ≥ 門檻時曝險減半,回落才解除"),
    "D": ("Walk-forward HMM", "隱馬可夫模型偵測市場狀態,每季僅用過去資料重訓"),
}
METHOD_LABELS = {
    "equal": "等權 1/N",
    "risk_parity": "風險平價",
    "max_sharpe": "Max Sharpe",
    "hrp": "HRP 階層風險平價",
}


# =================================================================
# 資料層
# =================================================================
def load_data(force_download=False):
    """讀取本地快取;force_download=True 時改由 yfinance 下載最新資料。"""
    if (not force_download) and os.path.exists(PRICE_CACHE) and os.path.exists(VIX_CACHE):
        prices = pd.read_parquet(PRICE_CACHE)
        vix = pd.read_parquet(VIX_CACHE)["vix"]
        return prices, vix

    import yfinance as yf
    raw = yf.download(TICKERS + [VIX_TICKER], start=START, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    prices = close[TICKERS].dropna(how="all").ffill(limit=5).dropna()
    vix = close[VIX_TICKER].reindex(prices.index).ffill(limit=5)
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices.to_parquet(PRICE_CACHE)
    vix.to_frame("vix").to_parquet(VIX_CACHE)
    return prices, vix


def build_features(prices, vix):
    """所有特徵僅使用過去資料,無 look-ahead。"""
    daily_ret = prices.pct_change()
    f = {
        "prices": prices,
        "vix": vix,
        "daily_ret": daily_ret,
        "mom_1m": prices.pct_change(21),
        "mom_3m": prices.pct_change(63),
        "mom_6m": prices.pct_change(126),
        "mom_12m": prices.pct_change(252),
        "ma200_ratio": prices / prices.rolling(200).mean(),
        "ma50_ratio": prices / prices.rolling(50).mean(),
        "vol_20d": daily_ret.rolling(20).std() * np.sqrt(252),
        "vol_60d": daily_ret.rolling(60).std() * np.sqrt(252),
        "vix_chg": vix.pct_change(),
        "vix_ma20_ratio": vix / vix.rolling(20).mean(),
    }
    f["mom_blend"] = (f["mom_3m"] + f["mom_6m"] + f["mom_12m"]) / 3
    return f


# =================================================================
# 策略層
# =================================================================
def _month_ends(index):
    s = pd.Series(range(len(index)), index=index)
    return index[s.groupby([index.year, index.month]).last().values]


def _expand_monthly(decisions, full_index):
    """月底決策 → 每日目標權重(決策日隔日生效)。"""
    w = pd.DataFrame(0.0, index=full_index, columns=TICKERS)
    dec_dates = sorted(decisions.keys())
    for i, d in enumerate(dec_dates):
        end = dec_dates[i + 1] if i + 1 < len(dec_dates) else full_index[-1]
        mask = (full_index > d) & (full_index <= end)
        for tkr, wt in decisions[d].items():
            w.loc[mask, tkr] = wt
    return w


def strat_A_buyhold(feats):
    w = pd.DataFrame(0.0, index=feats["prices"].index, columns=TICKERS)
    w["SPY"] = 1.0
    return w


def strat_B_momentum(feats, top_n=2):
    mom, abs_mom = feats["mom_blend"], feats["mom_12m"]
    idx = feats["prices"].index
    decisions = {}
    for d in _month_ends(idx):
        scores = mom.loc[d].dropna()
        if len(scores) == 0:
            continue
        ranked = scores.sort_values(ascending=False)
        picks = [t for t in ranked.index[:top_n] if abs_mom.loc[d, t] > 0]
        decisions[d] = {t: 1.0 / len(picks) for t in picks} if picks else {}
    return _expand_monthly(decisions, idx)


def strat_C1_regime(feats):
    idx = feats["prices"].index
    decisions = {}
    for d in _month_ends(idx):
        if pd.isna(feats["ma200_ratio"].loc[d, "SPY"]):
            continue
        risk_on = (feats["ma200_ratio"].loc[d, "SPY"] > 1.0) and (feats["mom_12m"].loc[d, "SPY"] > 0)
        if risk_on:
            pick = "QQQ" if feats["mom_6m"].loc[d, "QQQ"] > feats["mom_6m"].loc[d, "SPY"] else "SPY"
            decisions[d] = {pick: 1.0}
        else:
            decisions[d] = {"TLT": 0.5, "GLD": 0.5}
    return _expand_monthly(decisions, idx)


def strat_C2_regime_vix(feats, vix_threshold=28.0, vix_exit=20.0):
    base = strat_C1_regime(feats).copy()
    vix_prev = feats["vix"].shift(1).reindex(base.index).values
    defensive, flags = False, np.zeros(len(base), dtype=bool)
    for i, v in enumerate(vix_prev):
        if not np.isnan(v):
            if (not defensive) and v >= vix_threshold:
                defensive = True
            elif defensive and v <= vix_exit:
                defensive = False
        flags[i] = defensive
    base.loc[flags] = base.loc[flags] * 0.5
    return base


def strat_D_hmm_walkforward(feats, init_train=756, refit_every=63):
    """Walk-forward HMM:每 refit_every 天以「截至當日」資料重訓,杜絕 look-ahead。
    決策只發生在月底,因此只在月底推斷狀態(結果與逐日推斷相同,速度快數十倍)。"""
    from hmmlearn.hmm import GaussianHMM

    idx = feats["prices"].index
    X = pd.DataFrame({
        "vix_chg": feats["vix_chg"],
        "vol20": feats["vol_20d"]["SPY"],
        "spy_ret": feats["daily_ret"]["SPY"],
    }, index=idx).dropna()
    Xv, n = X.values, len(X)
    month_end_set = set(_month_ends(idx))
    state_at = {}
    model, risk_off_state = None, 0

    for i in range(n):
        if i >= init_train and (i - init_train) % refit_every == 0:
            train = Xv[:i]
            try:
                m = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
                m.fit(train)
                st = m.predict(train)
                r0 = train[:, 2][st == 0].mean() if (st == 0).any() else 0
                r1 = train[:, 2][st == 1].mean() if (st == 1).any() else 0
                risk_off_state = 0 if r0 < r1 else 1
                model = m
            except Exception:
                pass
        d = X.index[i]
        if d in month_end_set:
            if model is not None and i >= init_train:
                state_at[d] = 1.0 if model.predict(Xv[: i + 1])[-1] == risk_off_state else 0.0
            else:
                state_at[d] = 0.0

    decisions = {d: ({"TLT": 0.5, "GLD": 0.5} if s == 1.0 else {"SPY": 1.0})
                 for d, s in state_at.items()}
    return _expand_monthly(decisions, idx)


STRATEGY_FUNCS = {
    "A": strat_A_buyhold,
    "B": strat_B_momentum,
    "C1": strat_C1_regime,
    "C2": strat_C2_regime_vix,
    "D": strat_D_hmm_walkforward,
}


# =================================================================
# 回測層
# =================================================================
def backtest(weights, daily_ret, cost=0.0005, band=0.05, dd_breaker=-0.20, breaker_scale=0.5):
    """含交易成本、no-trade band、回撤熔斷(含遲滯)的逐日回測。"""
    idx = weights.index
    W = weights[TICKERS].fillna(0.0).values
    R = daily_ret[TICKERS].reindex(idx).fillna(0.0).values
    n, k = W.shape

    held = np.zeros((n, k))
    equity = np.ones(n)
    cur_w = np.zeros(k)
    eq_val, peak = 1.0, 1.0
    total_traded, breaker_days = 0.0, 0
    breaker_on, breaker_release = False, dd_breaker / 2

    for i in range(n):
        tgt = W[i].copy()
        dd = eq_val / peak - 1.0
        if (not breaker_on) and dd <= dd_breaker:
            breaker_on = True
        elif breaker_on and dd >= breaker_release:
            breaker_on = False
        if breaker_on:
            tgt = tgt * breaker_scale
            breaker_days += 1

        drift = np.abs(tgt - cur_w).sum()
        if drift >= band:
            total_traded += drift
            eq_val *= (1 - cost * drift)
            cur_w = tgt
        held[i] = cur_w
        eq_val *= (1 + float(cur_w @ R[i]))
        equity[i] = eq_val
        peak = max(peak, eq_val)

    info = {
        "turnover": total_traded / (n / TRADING_DAYS),
        "breaker_days": breaker_days,
        "held_weights": pd.DataFrame(held, index=idx, columns=TICKERS),
    }
    return pd.Series(equity, index=idx), info


def metrics(equity, bench_ret=None):
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    yrs = len(eq) / TRADING_DAYS
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ret.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min()
    out = {
        "CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": maxdd, "Calmar": cagr / abs(maxdd) if maxdd < 0 else np.nan,
    }
    m_ret = eq.resample("ME").last().pct_change().dropna()
    out["WinRate"] = (m_ret > 0).mean() if len(m_ret) else np.nan
    var95 = ret.quantile(0.05)
    out["CVaR95"] = ret[ret <= var95].mean() if (ret <= var95).any() else np.nan
    if bench_ret is not None:
        b = bench_ret.reindex(ret.index).dropna()
        r = ret.reindex(b.index)
        if len(b) > 10 and b.var() > 0:
            beta = np.cov(r, b)[0, 1] / b.var()
            out["Beta"] = beta
            out["Alpha"] = (r.mean() - beta * b.mean()) * TRADING_DAYS
            excess = r - b
            te = excess.std() * np.sqrt(TRADING_DAYS)
            out["IR"] = (excess.mean() * TRADING_DAYS) / te if te > 0 else np.nan
    return out


def metrics_table(equities, bench_ret):
    return pd.DataFrame({k: metrics(v, bench_ret) for k, v in equities.items()}).T


# =================================================================
# 組合層
# =================================================================
def w_equal(R):
    return np.ones(R.shape[1]) / R.shape[1]


def w_risk_parity(R, iters=500, lr=0.01):
    cov = R.cov().values * TRADING_DAYS
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (cov @ w)
        w = np.clip(w - lr * (rc - (w @ cov @ w) / n), 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.ones(n) / n
    return w


def w_max_sharpe(R):
    mu = R.mean().values * TRADING_DAYS
    cov = R.cov().values * TRADING_DAYS
    n = len(mu)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    return res.x if res.success else np.ones(n) / n


def w_hrp(R):
    cov = R.cov().values * TRADING_DAYS
    corr = R.corr().values
    n = len(cov)
    if n == 1:
        return np.array([1.0])
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    diag = np.diag(cov).copy()
    diag[diag <= 0] = 1e-8
    np.fill_diagonal(cov, diag)

    dist = np.nan_to_num(np.sqrt(np.clip(0.5 * (1 - corr), 0, None)), nan=0.0)
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))

    def cluster_var(items):
        sub = cov[np.ix_(items, items)]
        iv = 1 / np.diag(sub)
        iv = iv / iv.sum()
        return iv @ sub @ iv

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        new = []
        for c in clusters:
            if len(c) <= 1:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2:]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            new += [left, right]
        clusters = new
    return w.reindex(range(n)).values


METHODS = {"equal": w_equal, "risk_parity": w_risk_parity, "max_sharpe": w_max_sharpe, "hrp": w_hrp}


def combine(comp_ret, method="hrp", lookback=252, cost=0.0005):
    """月底以回看窗估計權重,隔日生效,扣除權重變動成本。"""
    comp_ret = comp_ret.dropna()
    idx, cols = comp_ret.index, comp_ret.columns
    fn = METHODS[method]
    month_end_pos = set(pd.Series(range(len(idx)), index=idx)
                        .groupby([idx.year, idx.month]).last().values)

    whist = np.zeros((len(idx), len(cols)))
    cur = np.ones(len(cols)) / len(cols)
    for i in range(len(idx)):
        if i in month_end_pos and i >= lookback:
            cur = fn(comp_ret.iloc[i - lookback + 1: i + 1])
        whist[i] = cur
    whist = pd.DataFrame(whist, index=idx, columns=cols).shift(1).fillna(0.0)

    turn = whist.diff().abs().sum(axis=1)
    net = (whist * comp_ret).sum(axis=1) - turn * cost
    return (1 + net).cumprod(), whist


def efficient_frontier(comp_ret, n_port=8000, seed=42):
    rng = np.random.default_rng(seed)
    mu = comp_ret.mean().values * TRADING_DAYS
    cov = comp_ret.cov().values * TRADING_DAYS
    W = rng.dirichlet(np.ones(len(mu)), n_port)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    def point(w):
        r, v = w @ mu, np.sqrt(w @ cov @ w)
        return {"ret": r, "vol": v, "sharpe": r / v, "w": w}

    return {
        "scatter": pd.DataFrame({"vol": vols, "ret": rets, "sharpe": rets / vols}),
        "max_sharpe": point(w_max_sharpe(comp_ret)),
        "min_var": point(W[np.argmin(vols)]),
        "equal": point(w_equal(comp_ret)),
        "cols": list(comp_ret.columns),
    }


# =================================================================
# 延伸:波動目標 + 動態槓桿
# =================================================================
def apply_vol_target(equity, target_vol=0.10, lookback=20, max_leverage=1.0,
                     cost=0.0005, finance_rate=0.02):
    ret = equity.pct_change().fillna(0.0)
    realized = ret.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized).shift(1).clip(upper=max_leverage).fillna(0.0)
    borrowed = (scale - 1.0).clip(lower=0.0)
    net = scale * ret - scale.diff().abs().fillna(0.0) * cost - borrowed * finance_rate / TRADING_DAYS
    return (1 + net).cumprod()


CRISES = {
    "2008 金融海嘯": ("2008-01-01", "2009-03-31"),
    "2020 疫情閃崩": ("2020-02-01", "2020-04-30"),
    "2022 股債雙殺": ("2022-01-01", "2022-12-31"),
}
