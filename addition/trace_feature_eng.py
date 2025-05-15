"""
trace_feature_eng.py - 追蹤feature_eng_01.py的執行流程
功能：修改feature_eng_01.py，添加追蹤語句以找出錯誤原因
作者：AI助手
日期：2023-05-25
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import traceback
warnings.filterwarnings('ignore')

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

def trace_label_encoder_issue():
    """追蹤LabelEncoder問題"""
    print("=== 開始追蹤LabelEncoder問題 ===")
    
    try:
        # 載入清洗後的資料
        print("\n載入清洗後的資料...")
        train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
        test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
        
        print(f"訓練集: {train_df.shape}")
        print(f"測試集: {test_df.shape}")
        
        # 假設目標變數是最後一欄
        target_col = train_df.columns[-1]
        print(f"目標變數欄位：{target_col}")
        
        # 檢查目標變數的類型
        print(f"\n目標變數的資料類型:")
        print(f"訓練集: {train_df[target_col].dtype}")
        print(f"測試集: {test_df[target_col].dtype}")
        
        # 檢查目標變數唯一值
        train_unique = train_df[target_col].unique()
        test_unique = test_df[target_col].unique()
        
        print(f"\n訓練集目標變數唯一值 ({len(train_unique)}個):")
        print(train_unique)
        
        print(f"\n測試集目標變數唯一值 ({len(test_unique)}個):")
        print(test_unique)
        
        # 檢查測試集有而訓練集沒有的值
        diff_values = set(test_unique) - set(train_unique)
        if diff_values:
            print(f"\n測試集有而訓練集沒有的值:")
            print(diff_values)
            
            for val in diff_values:
                rows = test_df[test_df[target_col] == val]
                print(f"\n值 {val} 出現在 {len(rows)} 筆資料中")
                print("第一筆資料:")
                print(rows.iloc[0])
        
        # 模擬LabelEncoder的執行
        print("\n模擬LabelEncoder的執行...")
        encoder = LabelEncoder()
        try:
            # 先fit訓練集
            y_train_encoded = encoder.fit_transform(train_df[target_col])
            print(f"訓練集編碼成功，類別: {encoder.classes_}")
            
            # 檢查每個測試資料是否能被轉換
            problematic_indices = []
            for i, val in enumerate(test_df[target_col]):
                if val not in encoder.classes_:
                    problematic_indices.append(i)
            
            if problematic_indices:
                print(f"\n找到 {len(problematic_indices)} 筆可能有問題的測試資料")
                for idx in problematic_indices[:5]:  # 只顯示前5筆
                    print(f"\n問題資料 #{idx}:")
                    print(f"目標值: {test_df.iloc[idx][target_col]}")
                    print(f"資料類型: {type(test_df.iloc[idx][target_col])}")
                    print(f"資料內容:")
                    print(test_df.iloc[idx])
            
            # 嘗試轉換測試集
            try:
                y_test_encoded = encoder.transform(test_df[target_col])
                print("測試集轉換成功")
            except Exception as e:
                print(f"測試集轉換失敗: {e}")
                
                # 詳細分析錯誤
                error_msg = str(e)
                if "previously unseen labels" in error_msg:
                    # 解析錯誤訊息中的標籤值
                    import re
                    matches = re.findall(r'\[.*?\]', error_msg)
                    if matches:
                        print(f"未見過的標籤值: {matches[0]}")
                        
                        # 尋找具有這些標籤的資料
                        for match in matches:
                            try:
                                val_str = match.strip('[]')
                                if 'np.float64' in val_str:
                                    val_str = val_str.replace('np.float64(', '').replace(')', '')
                                val = float(val_str)
                                
                                rows = test_df[test_df[target_col] == val]
                                if not rows.empty:
                                    print(f"\n值 {val} 出現在 {len(rows)} 筆資料中")
                                    print("第一筆資料:")
                                    print(rows.iloc[0])
                            except:
                                print(f"無法解析標籤值: {match}")
        except Exception as e:
            print(f"執行LabelEncoder時發生錯誤: {e}")
        
        # 檢查2nd sem欄位
        print("\n檢查2nd sem欄位...")
        sem_cols = [col for col in train_df.columns if '2nd sem' in col]
        if sem_cols:
            for col in sem_cols:
                print(f"\n欄位: {col}")
                print(f"資料類型: 訓練集={train_df[col].dtype}, 測試集={test_df[col].dtype}")
                
                train_vals = sorted(train_df[col].unique())
                test_vals = sorted(test_df[col].unique())
                
                print(f"訓練集唯一值: {train_vals}")
                print(f"測試集唯一值: {test_vals}")
                
                diff = set(test_vals) - set(train_vals)
                if diff:
                    print(f"測試集有而訓練集沒有的值: {diff}")
                    
                    # 特別注意0值
                    if 0 in diff and 0 not in train_vals:
                        zero_rows = test_df[test_df[col] == 0]
                        print(f"\n值 0 出現在 {len(zero_rows)} 筆資料中")
                        print("第一筆資料:")
                        print(zero_rows.iloc[0])
                        
                        # 嘗試對這個欄位執行LabelEncoder
                        print(f"\n嘗試對 {col} 欄位執行LabelEncoder...")
                        try:
                            col_encoder = LabelEncoder()
                            col_encoder.fit(train_df[col])
                            col_encoder.transform(test_df[col])
                            print("轉換成功")
                        except Exception as e:
                            print(f"轉換失敗: {e}")
        
        print("\n=== 追蹤完成 ===")
                    
    except Exception as e:
        print(f"追蹤過程中發生錯誤: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    trace_label_encoder_issue() 