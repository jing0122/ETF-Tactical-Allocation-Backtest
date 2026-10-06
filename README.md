# 跨資產 ETF 戰術配置回測平台

以 SPY / QQQ / TLT / GLD 四檔 ETF 為標的、VIX 為訊號，建構五種戰術策略並以四種組合方法配權的互動式回測系統。
由演算法交易期末專題（`main B1143036.ipynb`）模組化並加上 Streamlit 介面。

## 功能
- **五種策略**：Buy & Hold、動能輪動、月頻 Regime、Regime + VIX 防禦、Walk-forward HMM
- **四種組合方法**：等權、風險平價、Max Sharpe、HRP（階層風險平價）
- **回測引擎**：交易成本、no-trade band、回撤熔斷
- **延伸**：波動目標（Volatility Targeting）與動態槓桿（含融資成本）
- **分析**：績效指標（Sharpe / Sortino / Calmar / Alpha / Beta / IR / CVaR）、效率前緣、危機壓力測試
- 所有參數可在側邊欄即時調整

## 系統架構
```
yfinance → 資料層(快取/清洗) → 特徵工程 → 策略層 → 回測引擎 → 組合層 → 波動目標 → 視覺化
```

| 檔案 | 說明 |
|---|---|
| `engine.py` | 核心引擎：資料、策略、回測、組合、績效指標 |
| `app.py` | Streamlit 互動介面 |
| `data_cache/` | 價格快取（離線即可執行；側邊欄可一鍵更新） |

## 防止前視偏誤
- 特徵只用過去資料，月底決策隔日生效
- HMM 採 walk-forward，每 63 個交易日僅用截至當日資料重訓
- 組合配權只用回看窗內資料

## 執行
```bash
pip install -r requirements.txt
streamlit run app.py
```
Windows 可直接雙擊 `run_app.bat`。

## 技術棧
Python · pandas · NumPy · SciPy · hmmlearn · Streamlit · Plotly

---
長庚大學 B1143036｜僅供學術研究，不構成投資建議
