"""
find_specific_error.py - 檢查特定欄位錯誤
功能：專門檢查"Curricular units 2nd sem (without evaluations)"欄位可能造成的錯誤
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

def check_specific_column():
    """專門檢查特定欄位的問題"""
    target_column = "Curricular units 2nd sem (without evaluations)"
    print(f"=== 專門檢查 '{target_column}' 欄位 ===")
    
    try:
        # 加載清洗後的訓練和測試數據
        print("加載清洗後的訓練和測試數據...")
        train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
        test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
        
        # 確認目標欄位存在
        if target_column not in train_df.columns:
            similar_cols = [col for col in train_df.columns if "2nd sem" in col]
            if similar_cols:
                target_column = similar_cols[0]
                print(f"找到相似欄位: {target_column}")
            else:
                print(f"找不到目標欄位或相似欄位，請檢查欄位名稱")
                return
        
        # 檢查該欄位在訓練集和測試集中的數據
        print(f"\n檢查 '{target_column}' 欄位...")
        
        # 檢查數據類型
        print(f"訓練集中此欄位的數據類型: {train_df[target_column].dtype}")
        print(f"測試集中此欄位的數據類型: {test_df[target_column].dtype}")
        
        # 檢查唯一值
        train_values = train_df[target_column].unique()
        test_values = test_df[target_column].unique()
        print(f"訓練集中此欄位的唯一值: {sorted(train_values)}")
        print(f"測試集中此欄位的唯一值: {sorted(test_values)}")
        
        # 檢查測試集中有但訓練集中沒有的值
        diff_values = set(test_values) - set(train_values)
        if diff_values:
            print(f"\n測試集中有但訓練集中沒有的值: {diff_values}")
            
            # 顯示包含這些值的資料行
            for val in diff_values:
                rows = test_df[test_df[target_column] == val]
                print(f"\n值 {val} 出現在 {len(rows)} 筆資料中")
                if len(rows) > 0:
                    # 顯示第一行的重要資訊
                    print("第一行的重要資訊:")
                    for col in rows.columns:
                        print(f"  {col}: {rows.iloc[0][col]}")
        else:
            print("\n測試集中沒有訓練集中沒有的值")
        
        # 嘗試對此欄位進行標籤編碼
        print("\n嘗試對此欄位進行標籤編碼...")
        try:
            encoder = LabelEncoder()
            train_encoded = encoder.fit_transform(train_df[target_column])
            print(f"訓練集編碼成功，類別為: {encoder.classes_}")
            
            try:
                test_encoded = encoder.transform(test_df[target_column])
                print("測試集編碼成功")
            except Exception as e:
                print(f"測試集編碼失敗: {e}")
                
                # 特別測試值為0的情況
                zero_rows = test_df[test_df[target_column] == 0]
                if not zero_rows.empty:
                    print(f"\n測試集中值為0的行數: {len(zero_rows)}")
                    print("第一行數據:")
                    print(zero_rows.iloc[0])
                
                # 檢查測試集中的特殊值
                for val in test_df[target_column].unique():
                    if val not in encoder.classes_:
                        val_rows = test_df[test_df[target_column] == val]
                        print(f"\n測試集中的特殊值 {val}，出現 {len(val_rows)} 次")
                        if not val_rows.empty:
                            # 顯示此值的第一行資料中的目標變數
                            if 'Target' in val_rows.columns:
                                print(f"第一行的目標變數: {val_rows.iloc[0]['Target']}")
                            # 顯示資料中的所有欄位值
                            important_cols = [col for col in val_rows.columns if '2nd sem' in col or 'Success' in col or 'Target' in col]
                            for col in important_cols:
                                print(f"  {col}: {val_rows.iloc[0][col]}")
        except Exception as e:
            print(f"標籤編碼過程中發生錯誤: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"檢查過程中發生錯誤: {e}")
        traceback.print_exc()

def check_target_values_after_cleaning():
    """檢查資料清洗後目標變數的轉換"""
    print("\n=== 檢查資料清洗對目標變數的影響 ===")
    
    try:
        # 加載原始數據
        original_train = pd.read_csv(WORK_DIR / 'data_train.csv', sep=';')
        original_test = pd.read_csv(WORK_DIR / 'data_test.csv', sep=';')
        
        # 加載清洗後的數據
        cleaned_train = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
        cleaned_test = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
        
        # 確認目標欄位
        original_target = original_train.columns[-1]  # 假設原始目標變數是最後一欄
        cleaned_target = cleaned_train.columns[-1]  # 假設清洗後目標變數是最後一欄
        
        print(f"原始目標變數欄位: {original_target}")
        print(f"清洗後目標變數欄位: {cleaned_target}")
        
        # 檢查原始目標變數的值
        original_train_values = original_train[original_target].unique()
        original_test_values = original_test[original_target].unique()
        
        print(f"\n原始訓練集目標變數的唯一值: {original_train_values}")
        print(f"原始測試集目標變數的唯一值: {original_test_values}")
        
        # 檢查清洗後目標變數的值
        cleaned_train_values = cleaned_train[cleaned_target].unique()
        cleaned_test_values = cleaned_test[cleaned_target].unique()
        
        print(f"\n清洗後訓練集目標變數的唯一值數量: {len(cleaned_train_values)}")
        print(f"清洗後測試集目標變數的唯一值數量: {len(cleaned_test_values)}")
        
        # 檢查是否有字符到數值的轉換
        if original_train[original_target].dtype != cleaned_train[cleaned_target].dtype:
            print(f"\n檢測到目標變數類型轉換:")
            print(f"  原始類型: {original_train[original_target].dtype}")
            print(f"  清洗後類型: {cleaned_train[cleaned_target].dtype}")
            
            # 檢查是否有映射關係
            if 'Graduate' in original_train_values and 'Graduate' not in cleaned_train_values:
                print("\n似乎存在字符類別到數值的映射:")
                
                # 嘗試找出映射關係
                mapping = {}
                for original_val in ['Graduate', 'Dropout', 'Enrolled']:
                    if original_val in original_train_values:
                        # 找出包含此原始值的行
                        original_rows = original_train[original_train[original_target] == original_val]
                        if not original_rows.empty:
                            # 找出對應的清洗後值
                            first_row_id = original_rows.index[0]
                            if first_row_id in cleaned_train.index:
                                mapped_val = cleaned_train.loc[first_row_id, cleaned_target]
                                mapping[original_val] = mapped_val
                                print(f"  {original_val} 可能映射到 {mapped_val}")
                
                # 檢查是否有問題的映射關係
                if mapping:
                    problem_vals = []
                    for val in cleaned_test_values:
                        if val not in cleaned_train_values:
                            problem_vals.append(val)
                    
                    if problem_vals:
                        print(f"\n可能有問題的數值: {problem_vals}")
                        for val in problem_vals:
                            rows = cleaned_test[cleaned_test[cleaned_target] == val]
                            if not rows.empty:
                                print(f"值 {val} 出現在 {len(rows)} 筆資料中")
                                print("第一行數據:")
                                print(rows.iloc[0])
    
    except Exception as e:
        print(f"檢查過程中發生錯誤: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_specific_column()
    check_target_values_after_cleaning() 