import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import json
import os
from datetime import datetime, timedelta

# ==========================================
# 0. 數據驗證模組 (Data Validator Class)
# ==========================================
class DataValidator:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.logs = [] 
        self.is_valid = True

    def log(self, message):
        self.logs.append(f"{message}")

    def run_all_checks(self):
        """執行所有檢查，返回 (Pass/Fail, 錯誤訊息)"""
        try:
            if not self.check_empty(): return False, self.logs
            if not self.check_recency(): return False, self.logs
            if not self.check_length(): return False, self.logs
            if not self.check_zeros_and_nans(): return False, self.logs
            # check_extreme_moves 可以視情況開啟或調整，這裡設為 50%
            if not self.check_extreme_moves(threshold=0.50): return False, self.logs
            
            return True, self.logs
        except Exception as e:
            self.log(f"檢查過程錯誤: {str(e)}")
            return False, self.logs

    def check_empty(self):
        if self.data is None or self.data.empty:
            self.log("數據為空")
            self.is_valid = False
            return False
        return True

    def check_recency(self, max_lag_days=5):
        last_date = self.data.index[-1]
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
        
        # 取得當前 UTC 時間 (GitHub Actions 也是 UTC)
        now = datetime.utcnow()
        days_diff = (now - last_date).days
        
        if days_diff > max_lag_days:
            self.log(f"數據過期 (落後 {days_diff} 天, 最後日期: {last_date.date()})")
            self.is_valid = False
            return False
        return True

    def check_length(self, min_days=500):
        if len(self.data) < min_days:
            self.log(f"數據長度不足 ({len(self.data)} < {min_days})")
            self.is_valid = False
            return False
        return True

    def check_zeros_and_nans(self):
        if (self.data['Close'] == 0).any():
            self.log("發現收盤價為 0")
            self.is_valid = False
            return False
            
        if self.data['Close'].isnull().any():
            nan_count = self.data['Close'].isnull().sum()
            if nan_count > 10:
                self.log(f"發現過多 NaN ({nan_count})")
                self.is_valid = False
                return False
            else:
                # 少量 NaN 自動填補
                self.data['Close'] = self.data['Close'].ffill()
        return True

    def check_extreme_moves(self, threshold=0.50):
        pct_change = self.data['Close'].pct_change().dropna()
        extreme_days = pct_change[abs(pct_change) > threshold]
        if not extreme_days.empty:
            last_extreme_date = extreme_days.index[-1].date()
            val = extreme_days.iloc[-1]
            self.log(f"異常價格跳動 ({last_extreme_date}: {val:.1%})")
            self.is_valid = False
            return False
        return True

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

# ==========================================
# 2. 核心計算邏輯 (整合驗證器)
# ==========================================
def calculate_strategy(ticker, config):
    print(f"正在分析 {ticker} ...")
    
    # 1. 下載數據 (包在 try-except 中)
    try:
        data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    except Exception as e:
        print(f"下載失敗 {ticker}: {e}")
        return {
            "ticker": ticker,
            "final_decision": "數據錯誤",
            "garch_signal": "N/A",
            "sma_signal": "N/A",
            "trigger_reason": f"API 連線失敗",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "price": 0.0, "z_score": 0.0, "is_above_sma": False, "sma_price": 0.0,
            "thresholds": "API Error"
        }

    # === 2. 數據源自我檢查 (Data Validation) ===
    validator = DataValidator(ticker, data)
    is_clean, logs = validator.run_all_checks()
    
    if not is_clean:
        error_msg = "; ".join(logs)
        print(f"[{ticker}] 數據驗證失敗: {error_msg}")
        
        # 回傳錯誤狀態 JSON，讓 App 顯示紅色/灰色警告
        return {
            "ticker": ticker,
            "final_decision": "數據異常", # App 會識別這個狀態
            "garch_signal": "錯誤",
            "sma_signal": "錯誤",
            "trigger_reason": error_msg, # App 上顯示具體錯誤原因
            "date": data.index[-1].strftime("%Y-%m-%d") if not data.empty else "N/A",
            "price": round(data['Close'].iloc[-1], 2) if not data.empty else 0.0,
            "z_score": 0.0,
            "is_above_sma": False,
            "sma_price": 0.0,
            "thresholds": "Validation Failed"
        }

    # === 3. 數據通過檢查，開始正常計算 ===
    
    # 計算 200 SMA
    data['SMA'] = data['Close'].rolling(window=config['sma_window']).mean()
    
    # 準備 GARCH 數據
    data['Ret_Pct'] = data['Close'].pct_change() * 100
    clean_data = data.dropna()
    train_data = clean_data['Ret_Pct'].tail(1200)
    
    # 訓練 GARCH
    try:
        model = arch_model(train_data, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = model.fit(disp='off')
    except Exception as e:
        return {
            "ticker": ticker, "final_decision": "模型錯誤", 
            "trigger_reason": f"GARCH 收斂失敗: {str(e)}",
            "date": datetime.now().strftime("%Y-%m-%d"), "price": 0.0, "z_score": 0.0, 
            "is_above_sma": False, "sma_price": 0.0, "thresholds": "Model Error", "garch_signal": "N/A", "sma_signal": "N/A"
        }
    
    # 計算 Z-Score
    garch_vol = res.conditional_volatility
    garch_vol_ann = garch_vol * np.sqrt(252)
    
    z_window = 126
    vol_mean = garch_vol_ann.rolling(window=z_window).mean()
    vol_std = garch_vol_ann.rolling(window=z_window).std()
    z_score = (garch_vol_ann - vol_mean) / vol_std
    
    # 產生訊號與回溯
    analysis_df = pd.DataFrame({'Z_Score': z_score}).dropna().tail(500)
    
    # 遲滯邏輯
    analysis_df['Raw_Signal'] = np.nan
    analysis_df.loc[analysis_df['Z_Score'] > config['garch_exit'], 'Raw_Signal'] = 0.0
    analysis_df.loc[analysis_df['Z_Score'] < config['garch_entry'], 'Raw_Signal'] = 1.0
    analysis_df['State'] = analysis_df['Raw_Signal'].ffill().fillna(1.0)
    
    # 取得最新狀態
    last_idx = data.index[-1]
    current_z = z_score.loc[last_idx] if last_idx in z_score.index else z_score.iloc[-1]
    garch_state = analysis_df['State'].iloc[-1]
    
    # 回溯觸發原因
    trigger_msg = "維持原判"
    state_changes = analysis_df['State'].diff()
    
    if garch_state == 0.0:
        change_dates = analysis_df[state_changes == -1.0].index
        action = "賣出"
    else:
        change_dates = analysis_df[state_changes == 1.0].index
        action = "買回"
        
    if len(change_dates) > 0:
        last_change_date = change_dates[-1]
        trigger_z = analysis_df.loc[last_change_date, 'Z_Score']
        date_str = last_change_date.strftime("%Y-%m-%d")
        trigger_msg = f"於 {date_str} 觸發{action} (Z={trigger_z:.2f})"
    else:
        trigger_msg = "長期維持此狀態 (>500天)"

    # 整合 SMA
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
        "trigger_reason": trigger_msg
    }

# ==========================================
# 主程式執行
# ==========================================
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
    
    print("\n分析完成！signals.json 已生成。")
