"""
DA_data_seperate.py - 數據集拆分程式
功能：將原始數據集拆分為訓練集和測試集
作者：Tommy
日期：2025-05-15
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# 設定工作目錄但不改變當前目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

# 設定隨機種子以確保可重現性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(file_path):
    """載入原始數據集"""
    try:
        # 讀取CSV文件，注意分隔符是分號
        full_path = WORK_DIR / file_path
        df = pd.read_csv(full_path, sep=';')
        print(f"成功載入數據集，共 {len(df)} 筆資料")
        return df
    except Exception as e:
        print(f"載入數據時發生錯誤：{e}")
        return None

def explore_data(df):
    """探索數據基本資訊"""
    print("\n=== 數據集基本資訊 ===")
    print(f"資料筆數：{len(df)}")
    print(f"欄位數量：{len(df.columns)}")
    print(f"\n欄位名稱：")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    # 檢查目標變數分布
    target_col = df.columns[-1]  # 假設最後一欄是目標變數
    print(f"\n目標變數 '{target_col}' 的分布：")
    value_counts = df[target_col].value_counts()
    for value, count in value_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{value}: {count} ({percentage:.2f}%)")
    
    return target_col

def split_data(df, target_col, test_size=0.2):
    """
    拆分數據集為訓練集和測試集
    使用分層抽樣確保類別分布均衡
    """
    # 分離特徵和目標變數
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 執行分層拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,  # 分層抽樣
        random_state=RANDOM_SEED
    )
    
    # 合併回完整的DataFrame
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, test_df

def save_datasets(train_df, test_df):
    """儲存訓練集和測試集"""
    # 儲存為CSV格式，使用分號作為分隔符以保持一致性
    train_df.to_csv(WORK_DIR / 'data_train.csv', sep=';', index=False)
    test_df.to_csv(WORK_DIR / 'data_test.csv', sep=';', index=False)
    print("\n成功儲存檔案：")
    print(f"- {WORK_DIR / 'data_train.csv'}")
    print(f"- {WORK_DIR / 'data_test.csv'}")

def generate_split_report(original_df, train_df, test_df, target_col):
    """生成拆分統計報告"""
    report = []
    report.append("=== 數據集拆分報告 ===\n")
    report.append(f"原始數據集大小：{len(original_df)} 筆")
    report.append(f"訓練集大小：{len(train_df)} 筆 ({len(train_df)/len(original_df)*100:.1f}%)")
    report.append(f"測試集大小：{len(test_df)} 筆 ({len(test_df)/len(original_df)*100:.1f}%)")
    
    report.append(f"\n目標變數 '{target_col}' 在各數據集的分布：")
    
    # 原始數據集分布
    report.append("\n原始數據集：")
    original_dist = original_df[target_col].value_counts()
    for value, count in original_dist.items():
        percentage = (count / len(original_df)) * 100
        report.append(f"  {value}: {count} ({percentage:.2f}%)")
    
    # 訓練集分布
    report.append("\n訓練集：")
    train_dist = train_df[target_col].value_counts()
    for value, count in train_dist.items():
        percentage = (count / len(train_df)) * 100
        report.append(f"  {value}: {count} ({percentage:.2f}%)")
    
    # 測試集分布
    report.append("\n測試集：")
    test_dist = test_df[target_col].value_counts()
    for value, count in test_dist.items():
        percentage = (count / len(test_df)) * 100
        report.append(f"  {value}: {count} ({percentage:.2f}%)")
    
    # 儲存報告
    with open(WORK_DIR / 'data_split_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\n報告已儲存至 " + str(WORK_DIR / 'data_split_report.txt'))
    print('\n'.join(report))

def main():
    """主程式"""
    print("=== 數據集拆分程式開始執行 ===")
    
    # 載入數據
    df = load_data('data.csv')
    if df is None:
        return
    
    # 探索數據
    target_col = explore_data(df)
    
    # 拆分數據集
    print("\n開始拆分數據集...")
    train_df, test_df = split_data(df, target_col, test_size=0.2)
    
    # 儲存數據集
    save_datasets(train_df, test_df)
    
    # 生成報告
    generate_split_report(df, train_df, test_df, target_col)
    
    print("\n=== 數據集拆分完成 ===")

if __name__ == "__main__":
    main()
