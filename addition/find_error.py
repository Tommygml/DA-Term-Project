"""
find_error.py - 檢查feature_eng_01.py錯誤原因
功能：找出標籤編碼錯誤的原因
作者：AI助手
日期：2023-05-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import traceback

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

def check_data_differences():
    """檢查訓練集和測試集之間的數據差異"""
    print("=== 檢查訓練集和測試集之間的數據差異 ===")
    
    try:
        # 加載原始數據集
        print("加載原始數據集...")
        train_df = pd.read_csv(WORK_DIR / 'data_train.csv', sep=';')
        test_df = pd.read_csv(WORK_DIR / 'data_test.csv', sep=';')
        
        print(f"訓練集有 {len(train_df)} 筆資料和 {len(train_df.columns)} 個欄位")
        print(f"測試集有 {len(test_df)} 筆資料和 {len(test_df.columns)} 個欄位")
        
        # 檢查每個欄位的不同值
        print("\n檢查每個欄位的不同值...")
        for col in train_df.columns:
            train_unique = set(train_df[col].unique())
            test_unique = set(test_df[col].unique())
            
            # 檢查測試集中有而訓練集中沒有的值
            test_only = test_unique - train_unique
            if test_only:
                print(f"\n欄位 '{col}':")
                print(f"  測試集中有而訓練集中沒有的值: {test_only}")
                print(f"  數據類型: {train_df[col].dtype}")
                
                # 如果是 "Curricular units 2nd sem (without evaluations)" 欄位，特別檢查
                if "Curricular units 2nd sem" in col:
                    print(f"  特別關注: 此欄位可能是問題源")
                    for val in test_only:
                        rows = test_df[test_df[col] == val]
                        print(f"  值 {val} 出現在 {len(rows)} 筆資料中")
                        if len(rows) > 0:
                            print("  範例資料:")
                            print(rows.iloc[0].to_string())
        
        # 加載清洗後的數據集
        print("\n\n加載清洗後的數據集...")
        train_cleaned = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
        test_cleaned = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
        
        print(f"清洗後訓練集有 {len(train_cleaned)} 筆資料和 {len(train_cleaned.columns)} 個欄位")
        print(f"清洗後測試集有 {len(test_cleaned)} 筆資料和 {len(test_cleaned.columns)} 個欄位")
        
        # 檢查清洗後的每個欄位
        print("\n檢查清洗後的每個欄位...")
        target_col = train_cleaned.columns[-1]  # 假設目標變數是最後一欄
        print(f"目標變數欄位: {target_col}")
        
        # 特別檢查目標變數
        print(f"\n目標變數 '{target_col}' 的分布:")
        print(f"訓練集: {train_cleaned[target_col].value_counts().to_dict()}")
        print(f"測試集: {test_cleaned[target_col].value_counts().to_dict()}")
        
        # 檢查每個欄位的不同值
        for col in train_cleaned.columns:
            train_unique = set(train_cleaned[col].unique())
            test_unique = set(test_cleaned[col].unique())
            
            # 檢查測試集中有而訓練集中沒有的值
            test_only = test_unique - train_unique
            if test_only:
                print(f"\n清洗後欄位 '{col}':")
                print(f"  測試集中有而訓練集中沒有的值: {test_only}")
                print(f"  數據類型: {train_cleaned[col].dtype}")
                
                # 對於值大約為 0.999996 的特別關注
                for val in test_only:
                    if isinstance(val, (float, int)) and 0.999 < val < 1.0:
                        print(f"  可能的問題值: {val}")
                        rows = test_cleaned[test_cleaned[col] == val]
                        print(f"  此值出現在 {len(rows)} 筆資料中")
                        if len(rows) > 0:
                            print("  範例資料:")
                            print(rows.iloc[0].to_string())
                            
        print("\n=== 檢查完成 ===")
    
    except Exception as e:
        print(f"檢查過程中發生錯誤: {e}")
        traceback.print_exc()

def simulate_error():
    """模擬feature_eng_01.py中的錯誤"""
    print("=== 開始模擬feature_eng_01.py中可能的錯誤 ===")
    
    try:
        # 加載清洗後的訓練和測試數據
        print("加載訓練和測試數據...")
        train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
        test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
        
        # 顯示目標變數列名
        target_col = train_df.columns[-1]
        print(f"目標變數列名: {target_col}")
        
        # 顯示訓練集和測試集中目標變數的唯一值
        train_unique = train_df[target_col].unique()
        test_unique = test_df[target_col].unique()
        print(f"訓練集目標變數的唯一值數量: {len(train_unique)}")
        print(f"測試集目標變數的唯一值數量: {len(test_unique)}")
        
        # 檢查目標變數的類型
        print(f"訓練集目標變數的資料類型: {train_df[target_col].dtype}")
        print(f"測試集目標變數的資料類型: {test_df[target_col].dtype}")
        
        # 顯示目標變數是否真的是字符串類別
        print(f"訓練集中目標變數是否包含 'Graduate': {'Graduate' in train_unique}")
        print(f"訓練集中目標變數是否包含 'Enrolled': {'Enrolled' in train_unique}")
        print(f"訓練集中目標變數是否包含 'Dropout': {'Dropout' in train_unique}")
        
        # 特別檢查 "Curricular units 2nd sem (without evaluations)" 欄位
        sem2_cols = [col for col in train_df.columns if "2nd sem" in col]
        if sem2_cols:
            for col in sem2_cols:
                print(f"\n檢查欄位: {col}")
                train_vals = train_df[col].unique()
                test_vals = test_df[col].unique()
                print(f"訓練集唯一值: {train_vals[:10]}...")
                print(f"測試集唯一值: {test_vals[:10]}...")
                
                # 檢查測試集中有而訓練集中沒有的值
                diff_vals = set(test_vals) - set(train_vals)
                if diff_vals:
                    print(f"測試集中有而訓練集中沒有的值: {diff_vals}")
        
        # 嘗試執行feature_eng_01.py中的標籤編碼部分
        print("\n嘗試執行標籤編碼...")
        
        # 檢查標籤編碼是否應用於目標變數還是其他特徵
        print("檢查label_encoder是否應用在目標變數還是特徵上...")
        
        # 檢查feature_eng_01.py中可能使用LabelEncoder的所有列
        all_cols = train_df.columns.tolist()
        for col in all_cols:
            print(f"\n測試對欄位 '{col}' 進行標籤編碼:")
            try:
                # 準備數據
                X_train = train_df.drop(columns=[target_col])
                y_train = train_df[col]  # 嘗試對不同欄位進行編碼
                X_test = test_df.drop(columns=[target_col])
                y_test = test_df[col]
                
                # 標籤編碼
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                
                try:
                    y_test_encoded = label_encoder.transform(y_test)
                    print(f"欄位 '{col}' 標籤編碼成功")
                except Exception as e:
                    print(f"欄位 '{col}' 標籤編碼失敗: {e}")
                    
                    # 檢查失敗原因
                    train_unique = set(y_train.unique())
                    test_unique = set(y_test.unique())
                    diff = test_unique - train_unique
                    if diff:
                        print(f"  測試集中有而訓練集中沒有的值: {diff}")
                        for val in diff:
                            rows = test_df[test_df[col] == val]
                            if len(rows) > 0:
                                print(f"  值 {val} 的資料範例:")
                                print(rows.iloc[0][['Curricular units 2nd sem (without evaluations)', target_col]])
            except Exception as e:
                print(f"處理欄位 '{col}' 時發生錯誤: {e}")
    
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 檢查數據差異
    check_data_differences()
    
    # 模擬錯誤
    # simulate_error() 