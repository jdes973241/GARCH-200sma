import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import json
import os
from datetime import datetime

# ==========================================
# 1. 策略參數設定 (基於您的回測優化結果)
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
    
    # 1. 下載數據
    # 取 10 年數據以確保 SMA200 和 GARCH 滾動窗口有足夠樣本
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
    
    # 處理 yfinance 多層索引問題
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if len(data) < 500:
        print(f"警告: {ticker} 數據不足")
        return None

    # 2. 計算 200 SMA
    data['SMA'] = data['Close'].rolling(window=config['sma_window']).mean()
    
    # 3. 準備 GARCH 數據
    data['Ret_Pct'] = data['Close'].pct_change() * 100
    clean_data = data.dropna()
    
    # 為了運算效率，我們取最後 1200 天來訓練模型即可 (足夠涵蓋 Z-Score 的 126 天窗口)
    # 這樣可以避免 GitHub Actions 超時
    train_data = clean_data['Ret_Pct'].tail(1200)
    
    # 4. 訓練 GARCH(1,1) 模型
    # 使用 Student's t 分佈 (dist='t') 以適應肥尾風險
    model = arch_model(train_data, vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = model.fit(disp='off')
    
    # 5. 取得條件波動率並計算 Z-Score
    # conditional_volatility 是模型擬合出的歷史波動率序列
    garch_vol = res.conditional_volatility
    garch_vol_ann = garch_vol * np.sqrt(252) # 年化
    
    # 計算 Z-Score (基準窗口 126 天)
    z_window = 126
    vol_mean = garch_vol_ann.rolling(window=z_window).mean()
    vol_std = garch_vol_ann.rolling(window=z_window).std()
    z_score = (garch_vol_ann - vol_mean) / vol_std
    
    # 6. 產生當前訊號
    # 取得最新一天的數據索引
    last_date = data.index[-1]
    
    # --- 6.1 SMA 訊號 ---
    current_price = data.loc[last_date, 'Close']
    current_sma = data.loc[last_date, 'SMA']
    # 邏輯：價格 > SMA 為持有 (1.0)
    sma_signal = 1.0 if current_price > current_sma else 0.0
    
    # --- 6.2 GARCH 訊號 (遲滯邏輯) ---
    # 我們需要回溯一段時間來確定當前的"狀態"，因為遲滯依賴於過去
    # 取最後 200 天的 Z-Score 來跑狀態
    recent_z = z_score.tail(200)
    
    raw_garch_sig = pd.Series(np.nan, index=recent_z.index)
    # 賣出條件
    raw_garch_sig[recent_z > config['garch_exit']] = 0.0
    # 買回條件
    raw_garch_sig[recent_z < config['garch_entry']] = 1.0
    
    # 填補中間空缺 (ffill)，假設最開始是持有 (fillna(1.0))
    # 這樣最新的值就代表了經過遲滯判斷後的當下狀態
    garch_signal_series = raw_garch_sig.ffill().fillna(1.0)
    garch_signal = garch_signal_series.iloc[-1]
    
    # --- 6.3 混合訊號 (Blend) ---
    blend_score = (0.5 * garch_signal) + (0.5 * sma_signal)
    
    # 7. 格式化輸出
    current_z = z_score.iloc[-1]
    
    if blend_score == 1.0:
        final_text = "持有 100%"
    elif blend_score == 0.5:
        final_text = "持有 50%"
    else:
        final_text = "持有 0% (空手)"
        
    garch_status_text = "持有" if garch_signal == 1.0 else "賣出"
    sma_status_text = "持有" if sma_signal == 1.0 else "賣出"
    
    # 閾值說明文字
    threshold_desc = f"GARCH(Exit>{config['garch_exit']}, Entry<{config['garch_entry']})"

    return {
        "ticker": ticker,
        "date": last_date.strftime("%Y-%m-%d"),
        "price": round(current_price, 2),
        "z_score": round(current_z, 2),
        "is_above_sma": bool(current_price > current_sma),
        "sma_price": round(current_sma, 2),
        "garch_signal": garch_status_text,
        "sma_signal": sma_status_text,
        "final_decision": final_text,
        "thresholds": threshold_desc
    }

# ==========================================
# 主程式執行區塊
# ==========================================
if __name__ == "__main__":
    final_results = []
    
    print("=== 開始執行每日量化分析 ===")
    for ticker, conf in CONFIG.items():
        try:
            result = calculate_strategy(ticker, conf)
            if result:
                final_results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            
    # 建立最終 JSON 結構
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": final_results
    }
    
    # 寫入檔案 (供 GitHub Pages 或 Raw 讀取)
    # ensure_ascii=False 確保中文字能正常顯示
    with open("signals.json", "w", encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    print("\n分析完成！signals.json 已生成。")
