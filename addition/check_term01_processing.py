"""
check_term01_processing.py - 檢查da_term_01.py對資料的處理情況
功能：分析da_term_01.py程式碼，檢查其對資料的處理
作者：AI助手
日期：2023-05-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import re

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

def analyze_python_file(file_path):
    """分析Python檔案中的資料處理代碼"""
    print(f"=== 分析檔案：{file_path} ===")
    
    try:
        # 讀取檔案內容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查文件長度
        print(f"檔案大小：{len(content)} 字節")
        lines = content.split('\n')
        print(f"檔案行數：{len(lines)}")
        
        # 搜尋資料載入相關代碼
        print("\n搜尋資料載入相關代碼...")
        data_loading_patterns = [
            r"pd\.read_csv\((.*?)\)",
            r"load_data\((.*?)\)",
            r"read_(.*?)data"
        ]
        
        for i, line in enumerate(lines):
            for pattern in data_loading_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    print(f"行 {i+1}: {line.strip()}")
        
        # 搜尋資料處理相關代碼
        print("\n搜尋資料處理相關代碼...")
        data_processing_patterns = [
            r"clean(.*?)data",
            r"process(.*?)data",
            r"transform",
            r"convert",
            r"normalize",
            r"fillna",
            r"drop(.*?)na",
            r"drop_duplicates",
            r"to_csv"
        ]
        
        for i, line in enumerate(lines):
            for pattern in data_processing_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    print(f"行 {i+1}: {line.strip()}")
        
        # 檢查是否有輸出文件
        print("\n檢查輸出文件相關代碼...")
        output_patterns = [
            r"\.to_csv\((.*?)\)",
            r"save(.*?)data",
            r"write(.*?)file"
        ]
        
        for i, line in enumerate(lines):
            for pattern in output_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    print(f"行 {i+1}: {line.strip()}")
        
        # 檢查是否提及train_cleaned.csv和test_cleaned.csv
        print("\n檢查是否提及清洗後的資料文件...")
        cleaned_file_patterns = [
            r"train_cleaned\.csv",
            r"test_cleaned\.csv"
        ]
        
        found = False
        for i, line in enumerate(lines):
            for pattern in cleaned_file_patterns:
                if re.search(pattern, line):
                    print(f"行 {i+1}: {line.strip()}")
                    found = True
        
        if not found:
            print("未找到對清洗後資料文件的直接引用")
            
        # 檢查是否有目標變數的處理
        print("\n檢查目標變數的處理...")
        target_patterns = [
            r"Target",
            r"label",
            r"y_train",
            r"y_test",
            r"encode.*?target",
            r"target.*?encode"
        ]
        
        for i, line in enumerate(lines):
            for pattern in target_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    print(f"行 {i+1}: {line.strip()}")
        
        # 檢查是否有"Curricular units 2nd sem"的處理
        print("\n檢查'Curricular units 2nd sem'的處理...")
        sem2_patterns = [
            r"2nd sem",
            r"Curricular.*?units.*?2nd",
            r"Curricular.*?2nd"
        ]
        
        for i, line in enumerate(lines):
            for pattern in sem2_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    print(f"行 {i+1}: {line.strip()}")
        
        print("\n=== 分析完成 ===")
    
    except Exception as e:
        import traceback
        print(f"分析過程中發生錯誤：{e}")
        traceback.print_exc()

def check_actual_files():
    """檢查實際的檔案內容"""
    print("\n=== 檢查實際的檔案內容 ===")
    
    try:
        # 檢查是否存在清洗後的資料檔案
        train_cleaned_path = WORK_DIR / 'train_cleaned.csv'
        test_cleaned_path = WORK_DIR / 'test_cleaned.csv'
        
        if train_cleaned_path.exists() and test_cleaned_path.exists():
            print("找到清洗後的資料檔案")
            
            # 載入原始資料和清洗後的資料
            train_df = pd.read_csv(WORK_DIR / 'data_train.csv', sep=';')
            test_df = pd.read_csv(WORK_DIR / 'data_test.csv', sep=';')
            
            train_cleaned_df = pd.read_csv(train_cleaned_path)
            test_cleaned_df = pd.read_csv(test_cleaned_path)
            
            print(f"\n原始訓練資料：{train_df.shape}")
            print(f"清洗後訓練資料：{train_cleaned_df.shape}")
            print(f"原始測試資料：{test_df.shape}")
            print(f"清洗後測試資料：{test_cleaned_df.shape}")
            
            # 檢查欄位變化
            print("\n檢查欄位變化...")
            original_cols = set(train_df.columns)
            cleaned_cols = set(train_cleaned_df.columns)
            
            added_cols = cleaned_cols - original_cols
            removed_cols = original_cols - cleaned_cols
            
            if added_cols:
                print(f"新增欄位：{added_cols}")
            if removed_cols:
                print(f"移除欄位：{removed_cols}")
            
            # 檢查目標變數的變化
            original_target = train_df.columns[-1]  # 假設原始目標變數是最後一欄
            cleaned_target = train_cleaned_df.columns[-1]  # 假設清洗後目標變數是最後一欄
            
            print(f"\n原始目標變數：{original_target}")
            print(f"清洗後目標變數：{cleaned_target}")
            
            if original_target == cleaned_target:
                print("目標變數欄位名稱保持不變")
                
                # 檢查目標變數的資料類型
                orig_dtype = train_df[original_target].dtype
                cleaned_dtype = train_cleaned_df[cleaned_target].dtype
                
                print(f"原始目標變數資料類型：{orig_dtype}")
                print(f"清洗後目標變數資料類型：{cleaned_dtype}")
                
                if orig_dtype != cleaned_dtype:
                    print("警告：目標變數的資料類型已變更！")
                    
                    # 進一步檢查值的變化
                    orig_values = sorted(train_df[original_target].unique())
                    cleaned_values = sorted(train_cleaned_df[cleaned_target].unique())
                    
                    print(f"原始目標變數唯一值：{orig_values}")
                    print(f"清洗後目標變數唯一值：{cleaned_values[:10]}...")
                    
                    # 檢查是否為分類到數值的轉換
                    if isinstance(orig_values[0], str) and isinstance(cleaned_values[0], (int, float)):
                        print("目標變數似乎從分類轉換為數值")
                        
                        # 嘗試找出映射關係
                        for orig_val in orig_values:
                            orig_rows = train_df[train_df[original_target] == orig_val]
                            if not orig_rows.empty:
                                index = orig_rows.index[0]
                                if index in train_cleaned_df.index:
                                    mapped_val = train_cleaned_df.loc[index, cleaned_target]
                                    print(f"  {orig_val} -> {mapped_val}")
            
            # 檢查特殊欄位的變化
            sem2_cols = [col for col in train_df.columns if "2nd sem" in col and "evaluations" in col]
            if sem2_cols:
                col = sem2_cols[0]
                if col in train_cleaned_df.columns:
                    print(f"\n檢查特殊欄位：{col}")
                    
                    orig_dtype = train_df[col].dtype
                    cleaned_dtype = train_cleaned_df[col].dtype
                    
                    print(f"原始資料類型：{orig_dtype}")
                    print(f"清洗後資料類型：{cleaned_dtype}")
                    
                    if orig_dtype != cleaned_dtype:
                        print("警告：該欄位的資料類型已變更！")
                        
                    # 檢查值的變化
                    orig_values = sorted(train_df[col].unique())
                    cleaned_values = sorted(train_cleaned_df[col].unique())
                    
                    print(f"原始唯一值：{orig_values}")
                    print(f"清洗後唯一值：{cleaned_values}")
        else:
            print("未找到清洗後的資料檔案，可能尚未進行資料清洗")
        
        print("\n=== 檢查完成 ===")
    
    except Exception as e:
        import traceback
        print(f"檢查過程中發生錯誤：{e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 分析Python檔案
    analyze_python_file(WORK_DIR / 'da_term_01.py')
    
    # 檢查實際檔案
    check_actual_files() 