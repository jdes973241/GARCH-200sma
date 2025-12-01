import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import json
import os
from datetime import datetime

# ==========================================
# 1. 策略參數設定
# ==========================================
CONFIG = {
    "UPRO": {
        "garch_exit": 2.0,
        "garch_entry": 1.0,
        "sma_window": 200
    },
    "EURL": {
        "garch_exit": 3.75,
        "garch_entry": 1.75,
        "sma_window": 200
    },
    "EDC": {
        "garch_exit": 1.75,
        "garch_entry": -0.25,
        "sma_window": 200
    }
}

def calculate_strategy(ticker, config):
    print(f"正在分析 {ticker} ...")
    
    # 1. 下載數據 (取 10 年以確保足夠的歷史來回溯訊號)
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if len(data) < 500:
        return None

    # 2. 計算 200 SMA
    data['SMA'] = data['Close'].rolling(window=config['sma_window']).mean()
    
    # 3. 準備 GARCH 數據
    data['Ret_Pct'] = data['Close'].pct_change() * 100
    clean_data = data.dropna()
    train_data = clean_data['Ret_Pct'].tail(1200)
    
    # 4. 訓練 GARCH
    model = arch_model(train_data, vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = model.fit(disp='off')
    
    # 5. 計算 Z-Score
    garch_vol = res.conditional_volatility
    garch_vol_ann = garch_vol * np.sqrt(252)
    
    z_window = 126
    vol_mean = garch_vol_ann.rolling(window=z_window).mean()
    vol_std = garch_vol_ann.rolling(window=z_window).std()
    z_score = (garch_vol_ann - vol_mean) / vol_std
    
    # 6. 產生訊號與回溯 (關鍵修改部分)
    # 我們需要足夠長的歷史來找到上次變盤點，這裡取最後 500 天
    analysis_df = pd.DataFrame({'Z_Score': z_score}).dropna().tail(500)
    
    # 遲滯邏輯重建
    analysis_df['Raw_Signal'] = np.nan
    analysis_df.loc[analysis_df['Z_Score'] > config['garch_exit'], 'Raw_Signal'] = 0.0 # 賣出
    analysis_df.loc[analysis_df['Z_Score'] < config['garch_entry'], 'Raw_Signal'] = 1.0 # 買進
    
    # 填補狀態 (當前持倉)
    analysis_df['State'] = analysis_df['Raw_Signal'].ffill().fillna(1.0)
    
    # 取得最新狀態
    last_idx = data.index[-1]
    current_z = z_score.loc[last_idx] if last_idx in z_score.index else z_score.iloc[-1]
    garch_state = analysis_df['State'].iloc[-1] # 1.0 或 0.0
    
    # --- 回溯觸發點邏輯 ---
    trigger_msg = "無變動"
    state_changes = analysis_df['State'].diff()
    
    # 尋找最近一次「變盤」的日子
    if garch_state == 0.0:
        # 目前是賣出，找最近一次從 1 變 0 的日子
        change_dates = analysis_df[state_changes == -1.0].index
        state_str = "賣出"
    else:
        # 目前是持有，找最近一次從 0 變 1 的日子
        change_dates = analysis_df[state_changes == 1.0].index
        state_str = "持有"
        
    if len(change_dates) > 0:
        last_change_date = change_dates[-1]
        trigger_z = analysis_df.loc[last_change_date, 'Z_Score']
        date_str = last_change_date.strftime("%Y-%m-%d")
        
        if garch_state == 0.0:
            # 顯示這天是因為超過閾值而賣出
            trigger_msg = f"於 {date_str} 觸發賣出 (Z={trigger_z:.2f})"
        else:
            # 顯示這天是因為低於閾值而買回
            trigger_msg = f"於 {date_str} 觸發買回 (Z={trigger_z:.2f})"
    else:
        # 500天內都沒變過
        trigger_msg = "長期維持此狀態 (>500天)"

    # --- 7. 整合 SMA 與輸出 ---
    current_price = data.loc[last_idx, 'Close']
    current_sma = data.loc[last_idx, 'SMA']
    sma_signal = 1.0 if current_price > current_sma else 0.0
    
    blend_score = (0.5 * garch_state) + (0.5 * sma_signal)
    
    if blend_score == 1.0: final_text = "持有 100%"
    elif blend_score == 0.5: final_text = "持有 50%"
    else: final_text = "持有 0% (空手)"

    return {
        "ticker": ticker,
        "date": last_idx.strftime("%Y-%m-%d"),
        "price": round(current_price, 2),
        "z_score": round(current_z, 2),
        "is_above_sma": bool(current_price > current_sma),
        "sma_price": round(current_sma, 2),
        "garch_signal": "持有" if garch_state == 1.0 else "賣出",
        "sma_signal": "持有" if sma_signal == 1.0 else "賣出",
        "final_decision": final_text,
        "thresholds": f"GARCH(Exit>{config['garch_exit']}, Entry<{config['garch_entry']})",
        "trigger_reason": trigger_msg  # 新增欄位
    }

if __name__ == "__main__":
    final_results = []
    print("=== 開始執行每日量化分析 ===")
    for ticker, conf in CONFIG.items():
        try:
            res = calculate_strategy(ticker, conf)
            if res: final_results.append(res)
        except Exception as e:
            print(f"Error {ticker}: {e}")
            
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": final_results
    }
    
    with open("signals.json", "w", encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
