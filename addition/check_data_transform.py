"""
check_data_transform.py - 檢查資料分割時是否進行了轉換
功能：比較原始資料和分割後的資料，檢查是否有欄位轉換
作者：AI助手
日期：2023-05-25
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

def check_data_transformation():
    """檢查資料分割時是否進行了轉換"""
    print("=== 檢查資料分割時是否進行了轉換 ===")
    
    try:
        # 加載原始資料
        print("加載原始資料...")
        original_df = pd.read_csv(WORK_DIR / 'data.csv', sep=';')
        print(f"原始資料：{len(original_df)} 筆，{len(original_df.columns)} 個欄位")
        
        # 加載分割後的訓練集和測試集
        print("\n加載分割後的訓練集和測試集...")
        train_df = pd.read_csv(WORK_DIR / 'data_train.csv', sep=';')
        test_df = pd.read_csv(WORK_DIR / 'data_test.csv', sep=';')
        print(f"訓練集：{len(train_df)} 筆，{len(train_df.columns)} 個欄位")
        print(f"測試集：{len(test_df)} 筆，{len(test_df.columns)} 個欄位")
        
        # 檢查欄位名稱是否相同
        print("\n檢查欄位名稱...")
        original_cols = set(original_df.columns)
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        if original_cols == train_cols and original_cols == test_cols:
            print("欄位名稱完全相同")
        else:
            print("欄位名稱不同！")
            print(f"原始資料欄位：{original_cols}")
            print(f"訓練集欄位：{train_cols}")
            print(f"測試集欄位：{test_cols}")
        
        # 檢查每個欄位的資料類型是否相同
        print("\n檢查欄位資料類型...")
        for col in original_df.columns:
            if col in train_df.columns and col in test_df.columns:
                orig_dtype = original_df[col].dtype
                train_dtype = train_df[col].dtype
                test_dtype = test_df[col].dtype
                
                if orig_dtype != train_dtype or orig_dtype != test_dtype:
                    print(f"欄位 '{col}' 的資料類型不同:")
                    print(f"  原始資料: {orig_dtype}")
                    print(f"  訓練集: {train_dtype}")
                    print(f"  測試集: {test_dtype}")
        
        # 檢查特定欄位的值範圍是否相同
        print("\n檢查欄位值範圍...")
        target_col = original_df.columns[-1]  # 假設目標變數是最後一欄
        print(f"目標變數：{target_col}")
        
        print(f"原始資料目標變數唯一值：{original_df[target_col].unique()}")
        print(f"訓練集目標變數唯一值：{train_df[target_col].unique()}")
        print(f"測試集目標變數唯一值：{test_df[target_col].unique()}")
        
        # 特別檢查"Curricular units 2nd sem (without evaluations)"欄位
        sem2_cols = [col for col in original_df.columns if "2nd sem" in col and "evaluations" in col]
        if sem2_cols:
            col = sem2_cols[0]
            print(f"\n特別檢查欄位：{col}")
            
            orig_values = sorted(original_df[col].unique())
            train_values = sorted(train_df[col].unique())
            test_values = sorted(test_df[col].unique())
            
            print(f"原始資料唯一值：{orig_values}")
            print(f"訓練集唯一值：{train_values}")
            print(f"測試集唯一值：{test_values}")
            
            # 檢查測試集中有但訓練集中沒有的值
            train_set = set(train_values)
            test_set = set(test_values)
            diff_values = test_set - train_set
            
            if diff_values:
                print(f"\n測試集中有但訓練集中沒有的值：{diff_values}")
                
                # 檢查這些值在原始資料中是否存在
                for val in diff_values:
                    orig_rows = original_df[original_df[col] == val]
                    if not orig_rows.empty:
                        print(f"值 {val} 在原始資料中存在，出現 {len(orig_rows)} 次")
                    else:
                        print(f"值 {val} 在原始資料中不存在！")
        
        # 驗證拆分是否只是簡單分割，沒有轉換
        print("\n驗證拆分是否只是簡單分割...")
        
        # 檢查原始資料的前10行是否與訓練集的前10行相同
        matching_rows = 0
        for i in range(min(10, len(train_df))):
            train_row = train_df.iloc[i:i+1]
            # 在原始資料中查找完全相同的行
            matches = (original_df == train_row.values).all(axis=1).sum()
            if matches > 0:
                matching_rows += 1
        
        if matching_rows == 10:
            print("訓練集的前10行在原始資料中都能找到完全相同的行")
        else:
            print(f"訓練集的前10行中，只有 {matching_rows} 行在原始資料中找到完全相同的行")
            print("這表明數據可能在分割過程中進行了轉換")
        
        print("\n=== 檢查完成 ===")
    
    except Exception as e:
        import traceback
        print(f"檢查過程中發生錯誤：{e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_data_transformation() 