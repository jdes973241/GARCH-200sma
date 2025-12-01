import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataValidator:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.logs = [] # 儲存檢查紀錄
        self.is_valid = True

    def log(self, message):
        self.logs.append(f"[{self.ticker}] {message}")

    def run_all_checks(self):
        """執行所有檢查，返回 (Pass/Fail, 錯誤訊息)"""
        try:
            if not self.check_empty(): return False, self.logs
            if not self.check_recency(): return False, self.logs
            if not self.check_length(): return False, self.logs
            if not self.check_zeros_and_nans(): return False, self.logs
            if not self.check_extreme_moves(): return False, self.logs
            
            self.log("✅ 通過所有數據檢查")
            return True, self.logs
        except Exception as e:
            self.log(f"❌ 檢查過程發生未預期錯誤: {str(e)}")
            return False, self.logs

    # --- 檢查 1: 是否抓到空資料 ---
    def check_empty(self):
        if self.data is None or self.data.empty:
            self.log("❌ 數據為空 (Download Failed)")
            self.is_valid = False
            return False
        return True

    # --- 檢查 2: 數據是否過期 ---
    def check_recency(self, max_lag_days=5):
        # 考慮到週末與國定假日，允許最大 5 天的落差
        last_date = self.data.index[-1]
        # 移除時區資訊以便比較
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)
            
        days_diff = (datetime.now() - last_date).days
        
        if days_diff > max_lag_days:
            self.log(f"❌ 數據嚴重過期！最後日期: {last_date.date()} (落後 {days_diff} 天)")
            self.is_valid = False
            return False
        return True

    # --- 檢查 3: 長度是否足夠 GARCH 使用 ---
    def check_length(self, min_days=500):
        if len(self.data) < min_days:
            self.log(f"❌ 數據長度不足 ({len(self.data)} < {min_days})，GARCH 無法收斂")
            self.is_valid = False
            return False
        return True

    # --- 檢查 4: 零值與缺失值 ---
    def check_zeros_and_nans(self):
        # 檢查收盤價是否有 0 或 NaN
        if (self.data['Close'] == 0).any():
            self.log("❌ 發現收盤價為 0 的異常數據")
            self.is_valid = False
            return False
            
        if self.data['Close'].isnull().any():
            self.log("❌ 發現 NaN 缺失值")
            # 嘗試修復：通常可以用 ffill 修復，但若太多則報錯
            nan_count = self.data['Close'].isnull().sum()
            if nan_count > 5:
                self.is_valid = False
                return False
            else:
                self.log("⚠️ 發現少量 NaN，已執行自動填補")
                self.data['Close'] = self.data['Close'].ffill()
        return True

    # --- 檢查 5: 極端異常值 (防止數據源錯植) ---
    def check_extreme_moves(self, threshold=0.80):
        # 3倍槓桿 ETF 雖然波動大，但單日跌幅超過 80% 幾乎不可能 (除非數據錯了)
        # 計算單日報酬
        pct_change = self.data['Close'].pct_change().dropna()
        
        # 檢查是否有單日漲跌幅超過 80%
        extreme_days = pct_change[abs(pct_change) > threshold]
        
        if not extreme_days.empty:
            for date, val in extreme_days.items():
                self.log(f"❌ 檢測到異常價格跳動: {date.date()} 漲跌幅 {val:.2%}")
            self.is_valid = False
            return False
        return True
