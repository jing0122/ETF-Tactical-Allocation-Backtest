"""
跨資產 ETF 戰術配置回測平台(Streamlit 介面)
執行:streamlit run app.py
"""
import logging
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as E

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

st.set_page_config(page_title="ETF 戰術配置回測平台", page_icon="📈", layout="wide")

COLORS = {
    "A": "#8B95A5", "B": "#E8833A", "C1": "#4C8BF5", "C2": "#1F4FB8", "D": "#2BAE66",
    "equal": "#E8833A", "risk_parity": "#2BAE66", "max_sharpe": "#D6455D", "hrp": "#7B5CD6",
    "SPY": "#222222", "組合策略": "#7B5CD6", "組合+VT": "#C2185B",
}
PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                   legend=dict(orientation="h", yanchor="top", y=-0.08, x=0))


# -----------------------------------------------------------------
# 快取層:資料與策略權重只算一次
# -----------------------------------------------------------------
@st.cache_data(show_spinner="載入價格資料…")
def get_data(refresh_token=0):
    prices, vix = E.load_data(force_download=refresh_token > 0)
    return prices, vix


@st.cache_data(show_spinner="計算五個策略訊號(含 walk-forward HMM,首次約 10 秒)…")
def get_weights(prices, vix, vix_th, vix_exit):
    feats = E.build_features(prices, vix)
    out = {}
    for k, fn in E.STRATEGY_FUNCS.items():
        out[k] = fn(feats, vix_threshold=vix_th, vix_exit=vix_exit) if k == "C2" else fn(feats)
    return out, feats["daily_ret"]


@st.cache_data(show_spinner="回測中…")
def run_backtests(weights, daily, start, end, cost, band, dd_breaker):
    eqs, infos = {}, {}
    for k, w in weights.items():
        ws = w.loc[start:end]
        eq, info = E.backtest(ws, daily, cost=cost, band=band, dd_breaker=dd_breaker)
        eqs[k], infos[k] = eq, {"turnover": info["turnover"], "breaker_days": info["breaker_days"],
                                "last": info["held_weights"].iloc[-1]}
    return eqs, infos


@st.cache_data(show_spinner="計算組合配置…")
def run_combo(comp_ret, method, lookback, cost):
    return E.combine(comp_ret, method=method, lookback=lookback, cost=cost)


@st.cache_data(show_spinner=False)
def run_frontier(asset_ret):
    return E.efficient_frontier(asset_ret)


def pct(x, d=1):
    return "-" if pd.isna(x) else f"{x:+.{d}%}"


def style_perf(df):
    fmt = {c: "{:+.1%}" for c in ["CAGR", "AnnVol", "MaxDD", "Alpha", "CVaR95", "WinRate"] if c in df}
    fmt.update({c: "{:+.2f}" for c in ["Sharpe", "Sortino", "Calmar", "Beta", "IR"] if c in df})
    good = [c for c in ["Sharpe", "Sortino", "CAGR", "Calmar"] if c in df]
    bad = [c for c in ["AnnVol"] if c in df]
    s = df.style.format(fmt, na_rep="-")
    if len(df) > 1:
        s = s.background_gradient(subset=good, cmap="Greens", axis=0)
        s = s.background_gradient(subset=["MaxDD"], cmap="Reds_r", axis=0)
        if bad:
            s = s.background_gradient(subset=bad, cmap="Reds", axis=0)
    return s


def equity_fig(series_dict, title, log=True):
    fig = go.Figure()
    for name, s in series_dict.items():
        dash = "dash" if name == "SPY" else None
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(color=COLORS.get(name), width=2.2 if name == "SPY" else 1.6, dash=dash)))
    fig.update_layout(title=title, yaxis_type="log" if log else "linear",
                      yaxis_title="淨值(起始 = 1)", height=440, **PLOT_LAYOUT)
    return fig


def drawdown_fig(series_dict, title):
    fig = go.Figure()
    for name, s in series_dict.items():
        dd = s / s.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, mode="lines",
                                 fill="tozeroy" if (name != "SPY" and len(series_dict) <= 2) else None,
                                 line=dict(color=COLORS.get(name), width=1.2,
                                           dash="dash" if name == "SPY" else None)))
    fig.update_layout(title=title, yaxis_tickformat=".0%", height=320, **PLOT_LAYOUT)
    return fig


def rebase(s):
    s = s.dropna()
    return s / s.iloc[0]


# -----------------------------------------------------------------
# 側邊欄:參數
# -----------------------------------------------------------------
if "refresh" not in st.session_state:
    st.session_state.refresh = 0

prices, vix = get_data(st.session_state.refresh)
dmin, dmax = prices.index.min().date(), prices.index.max().date()

with st.sidebar:
    st.header("⚙️ 回測參數")

    st.subheader("回測期間")
    c1, c2 = st.columns(2)
    start = c1.date_input("起始", value=date(2006, 1, 1), min_value=dmin, max_value=dmax)
    end = c2.date_input("結束", value=dmax, min_value=dmin, max_value=dmax)

    st.subheader("交易與風控")
    cost_bp = st.slider("單邊交易成本(bp)", 0, 30, 5, help="1 bp = 0.01%")
    band = st.slider("No-trade band", 0.0, 0.20, 0.05, 0.01,
                     help="目標權重與現有持倉偏離總和低於此值時不調倉,減少無謂換手")
    dd_breaker = st.slider("回撤熔斷門檻", -0.50, -0.05, -0.20, 0.05,
                           help="回撤超過此值時曝險減半,回撤收斂一半後解除")

    st.subheader("C2 策略:VIX 門檻")
    vix_th = st.slider("進入防禦(VIX ≥)", 18.0, 45.0, 28.0, 1.0)
    vix_exit = st.slider("解除防禦(VIX ≤)", 12.0, 35.0, 20.0, 1.0)

    st.subheader("組合配置")
    combo_target = st.radio("組合對象", ["組合五個策略", "組合四個資產"], horizontal=True)
    method = st.selectbox("配權方法", list(E.METHODS), index=3, format_func=lambda m: E.METHOD_LABELS[m])
    lookback = st.slider("回看窗(交易日)", 63, 504, 252, 21)

    st.subheader("延伸:波動目標")
    use_vt = st.toggle("啟用 Volatility Targeting", value=False)
    target_vol = st.slider("目標年化波動", 0.05, 0.20, 0.10, 0.01, disabled=not use_vt)
    max_lev = st.slider("最大槓桿", 1.0, 2.0, 1.0, 0.1, disabled=not use_vt,
                        help="> 1.0 代表允許槓桿,超過 1 倍部分按年化 2% 計融資成本")

    st.divider()
    if st.button("🔄 從 yfinance 更新最新資料", width="stretch"):
        st.session_state.refresh += 1
        st.cache_data.clear()
        st.rerun()
    st.caption(f"資料:{dmin} ~ {dmax}({len(prices):,} 個交易日)")

if start >= end:
    st.error("起始日必須早於結束日")
    st.stop()
if vix_exit >= vix_th:
    st.sidebar.warning("解除門檻應低於進入門檻,否則防禦狀態會反覆切換")

cost = cost_bp / 10000
s_str, e_str = str(start), str(end)

# -----------------------------------------------------------------
# 計算
# -----------------------------------------------------------------
weights, daily = get_weights(prices, vix, vix_th, vix_exit)
eqs, infos = run_backtests(weights, daily, s_str, e_str, cost, band, dd_breaker)
spy_eq = rebase((1 + daily["SPY"].loc[s_str:e_str].fillna(0)).cumprod())
spy_ret = daily["SPY"]

strat_ret = pd.DataFrame({k: v.pct_change() for k, v in eqs.items()}).dropna()
asset_ret = daily[E.TICKERS].loc[s_str:e_str].dropna()
comp = strat_ret if combo_target == "組合五個策略" else asset_ret
combo_eq, combo_w = run_combo(comp, method, lookback, cost)
combo_eq = rebase(combo_eq)
final_eq = rebase(E.apply_vol_target(combo_eq, target_vol, max_leverage=max_lev, cost=cost)) if use_vt else combo_eq
final_name = "組合+VT" if use_vt else "組合策略"

m_final = E.metrics(final_eq, spy_ret)
m_spy = E.metrics(spy_eq, spy_ret)

# -----------------------------------------------------------------
# 頁首
# -----------------------------------------------------------------
st.title("📈 跨資產 ETF 戰術配置回測平台")
st.markdown(
    "以 **SPY / QQQ / TLT / GLD** 四檔 ETF 為標的、**VIX** 為訊號,"
    "建構五種戰術策略並以四種組合方法配權,目標是**風險調整後報酬(Sharpe)勝過大盤**。"
    "全程扣除交易成本,HMM 採 walk-forward 訓練以杜絕前視偏誤。"
)

label = f"{E.METHOD_LABELS[method]}・{combo_target.replace('組合', '')}" + ("・波動目標" if use_vt else "")
st.markdown(f"##### 目前組合:{label}　vs　SPY 買進持有")
k = st.columns(5)
k[0].metric("Sharpe", f"{m_final['Sharpe']:.2f}", f"{m_final['Sharpe'] - m_spy['Sharpe']:+.2f} vs SPY")
k[1].metric("年化報酬 CAGR", pct(m_final["CAGR"]), f"{(m_final['CAGR'] - m_spy['CAGR']) * 100:+.1f} pp")
k[2].metric("最大回撤", pct(m_final["MaxDD"]), f"{(m_final['MaxDD'] - m_spy['MaxDD']) * 100:+.1f} pp")
k[3].metric("年化波動", pct(m_final["AnnVol"]), f"{(m_final['AnnVol'] - m_spy['AnnVol']) * 100:+.1f} pp",
            delta_color="inverse")
k[4].metric("Alpha(年化)", pct(m_final.get("Alpha", np.nan)), f"Beta {m_final.get('Beta', np.nan):.2f}",
            delta_color="off")

tabs = st.tabs(["📊 總覽", "🧭 個別策略", "🧩 組合配置", "🎯 效率前緣", "🌪️ 危機壓測", "📘 方法說明"])

# ---- 總覽 ----
with tabs[0]:
    st.plotly_chart(equity_fig({final_name: final_eq, "SPY": spy_eq}, "淨值曲線(對數刻度)"),
                    width="stretch")
    st.plotly_chart(drawdown_fig({final_name: final_eq, "SPY": spy_eq}, "回撤曲線"), width="stretch")

    yr = pd.DataFrame({
        final_name: final_eq.resample("YE").last().pct_change(),
        "SPY": spy_eq.resample("YE").last().pct_change(),
    })
    first = pd.DataFrame({final_name: [final_eq.resample("YE").last().iloc[0] - 1],
                          "SPY": [spy_eq.resample("YE").last().iloc[0] - 1]},
                         index=[final_eq.resample("YE").last().index[0]])
    yr = pd.concat([first, yr.iloc[1:]])
    yr.index = yr.index.year
    fig = go.Figure()
    for c in yr.columns:
        fig.add_bar(x=yr.index, y=yr[c], name=c, marker_color=COLORS.get(c))
    fig.update_layout(title="逐年報酬", yaxis_tickformat=".0%", barmode="group", height=340, **PLOT_LAYOUT)
    st.plotly_chart(fig, width="stretch")

# ---- 個別策略 ----
with tabs[1]:
    info_df = pd.DataFrame(
        [{"代號": k, "策略": v[0], "邏輯": v[1],
          "年換手": f"{infos[k]['turnover']:.2f}", "熔斷天數": infos[k]["breaker_days"]}
         for k, v in E.STRATEGY_INFO.items()]
    ).set_index("代號")
    st.dataframe(info_df, width="stretch")

    t1 = E.metrics_table(eqs, spy_ret)
    st.markdown("**各策略績效**")
    st.dataframe(style_perf(t1), width="stretch")
    st.plotly_chart(equity_fig({**eqs, "SPY": spy_eq}, "各策略淨值曲線(對數刻度)"), width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        corr = strat_ret.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1,
                        title="策略報酬相關係數")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**最新持倉(期末權重)**")
        last = pd.DataFrame({k: v["last"] for k, v in infos.items()}).T
        last["現金"] = (1 - last.sum(axis=1)).clip(lower=0)
        st.dataframe(last.style.format("{:.0%}"), width="stretch")
        st.caption("相關性越低,組合起來的分散效果越好;C1 與 C2 高度相關是因為 C2 由 C1 延伸。")

# ---- 組合配置 ----
with tabs[2]:
    combos = {m: rebase(run_combo(comp, m, lookback, cost)[0]) for m in E.METHODS}
    tc = E.metrics_table({**combos, "SPY": spy_eq}, spy_ret)
    tc.index = [E.METHOD_LABELS.get(i, i) for i in tc.index]
    st.markdown(f"**{combo_target}:四種配權方法比較**(回看 {lookback} 日、月底再平衡)")
    st.dataframe(style_perf(tc), width="stretch")
    st.plotly_chart(drawdown_fig({**combos, "SPY": spy_eq}, "回撤比較"), width="stretch")

    wplot = combo_w.loc[combo_w.sum(axis=1) > 0].resample("ME").last()
    fig = go.Figure()
    for c in wplot.columns:
        fig.add_trace(go.Scatter(x=wplot.index, y=wplot[c], name=c, stackgroup="one", mode="lines",
                                 line=dict(width=0.5, color=COLORS.get(c))))
    fig.update_layout(title=f"{E.METHOD_LABELS[method]}:權重隨時間變化", yaxis_tickformat=".0%",
                      yaxis_range=[0, 1], height=360, **PLOT_LAYOUT)
    st.plotly_chart(fig, width="stretch")

# ---- 效率前緣 ----
with tabs[3]:
    ef = run_frontier(asset_ret)
    sc = ef["scatter"]
    fig = px.scatter(sc, x="vol", y="ret", color="sharpe", color_continuous_scale="Viridis", opacity=0.5,
                     labels={"vol": "年化波動", "ret": "年化報酬", "sharpe": "Sharpe"})
    fig.update_traces(marker=dict(size=4))
    for key, sym, col, lab in [("max_sharpe", "star", "#D6455D", "Max Sharpe"),
                               ("min_var", "diamond", "#4C8BF5", "最小變異"),
                               ("equal", "square", "#E8833A", "等權")]:
        p = ef[key]
        fig.add_trace(go.Scatter(x=[p["vol"]], y=[p["ret"]], mode="markers", name=lab,
                                 marker=dict(symbol=sym, size=18, color=col, line=dict(color="black", width=1))))
    fig.update_layout(title="四資產效率前緣(Monte Carlo 8,000 組隨機權重)", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", height=500, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    pts = pd.DataFrame({lab: ef[k]["w"] for k, lab in
                        [("max_sharpe", "Max Sharpe"), ("min_var", "最小變異"), ("equal", "等權")]},
                       index=ef["cols"]).T
    st.dataframe(pts.style.format("{:.1%}"), width="stretch")
    st.caption("注意:效率前緣使用全期資料估計,屬事後最適,僅供理解資產特性;實際回測的配權皆只用回看窗內的過去資料。")

# ---- 危機壓測 ----
with tabs[4]:
    rows = {}
    for cname, (s, e) in E.CRISES.items():
        row = {}
        for name, eq in {**eqs, final_name: final_eq, "SPY": spy_eq}.items():
            seg = eq.loc[s:e]
            if len(seg) > 1:
                row[name] = seg.iloc[-1] / seg.iloc[0] - 1
        if row:
            rows[cname] = row
    if rows:
        ct = pd.DataFrame(rows).T
        st.markdown("**三次市場危機期間總報酬**")
        st.dataframe(ct.style.format("{:+.1%}", na_rep="-").background_gradient(axis=None, cmap="RdYlGn"),
                     width="stretch")
        fig = go.Figure()
        for c in [final_name, "SPY"]:
            if c in ct:
                fig.add_bar(x=ct.index, y=ct[c], name=c, marker_color=COLORS.get(c),
                            text=[pct(v) for v in ct[c]], textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", barmode="group", height=360, **PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("目前選擇的回測期間未涵蓋任何危機事件,請調整側邊欄的起始日期。")

# ---- 方法說明 ----
with tabs[5]:
    st.markdown("""
### 系統架構
```
yfinance ─▶ 資料層(快取 / 清洗) ─▶ 特徵工程(動能、均線、波動、VIX)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                          ▼
           策略層 A / B / C1 / C2 / D                   組合資產(四檔 ETF)
                     │                                          │
                     ▼                                          │
       回測引擎(成本 / no-trade band / 熔斷)                    │
                     │                                          │
                     └──────────▶ 組合層(等權 / 風險平價 / Max Sharpe / HRP)
                                          │
                                          ▼
                         波動目標 / 動態槓桿 ─▶ 績效指標與視覺化
```

### 防止前視偏誤(look-ahead bias)的設計
- 所有特徵只用當日以前的資料計算;月底決策**隔日才生效**
- C2 使用**前一日** VIX,並以進出門檻不同的狀態機避免反覆切換
- HMM 採 **walk-forward**:每 63 個交易日只用截至當日的資料重訓
- 組合配權只用**回看窗**內資料,波動目標使用前一日以前的已實現波動

### 實務摩擦
- 每次調倉依權重變動量扣除交易成本
- No-trade band:偏離不夠大就不交易,降低換手
- 回撤熔斷:深度回撤時曝險減半,具遲滯效果

### 績效指標
| 指標 | 意義 |
|---|---|
| Sharpe / Sortino | 每單位總波動 / 下檔波動的報酬 |
| MaxDD / Calmar | 最大回撤 / 年化報酬除以最大回撤 |
| Alpha / Beta / IR | 相對 SPY 的超額報酬、連動程度、超額報酬穩定度 |
| CVaR95 | 最差 5% 交易日的平均日報酬(尾部風險) |

**技術棧:** Python・pandas・NumPy・SciPy・hmmlearn・Streamlit・Plotly
""")

st.divider()
st.caption("長庚大學 B1143036｜演算法交易期末專題｜本平台僅供學術研究,不構成任何投資建議。")
