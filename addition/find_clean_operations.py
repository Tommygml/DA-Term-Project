"""
find_clean_operations.py - 查找清洗後資料檔案的來源
功能：查找清洗後資料檔案的來源和處理方式
作者：AI助手
日期：2023-05-25
"""

import os
from pathlib import Path

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

def scan_python_files():
    """掃描所有Python檔案尋找資料清洗的線索"""
    print("=== 掃描Python檔案尋找資料清洗的線索 ===")
    
    try:
        # 獲取目錄中的所有Python檔案
        py_files = list(WORK_DIR.glob("*.py"))
        print(f"找到 {len(py_files)} 個Python檔案")
        
        # 需要搜尋的關鍵詞
        keywords = [
            "train_cleaned.csv", 
            "test_cleaned.csv",
            "to_csv",
            "clean_data",
            "data cleaning",
            "process_data",
            "process data",
            "transform",
            "label_encoder",
            "LabelEncoder",
            "Target",
            "Curricular units 2nd sem"
        ]
        
        # 搜尋每個文件
        for py_file in py_files:
            file_name = py_file.name
            print(f"\n檢查檔案：{file_name}")
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 檢查每個關鍵詞
                found_keywords = []
                for keyword in keywords:
                    if keyword in content:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"找到關鍵詞：{', '.join(found_keywords)}")
                    
                    # 如果檔案包含資料清洗關鍵詞，詳細檢查
                    if any(k in content for k in ["train_cleaned.csv", "test_cleaned.csv", "to_csv"]):
                        print("檢測到可能產生清洗後資料的檔案!")
                        
                        # 分析檔案的具體內容
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            
                            # 檢查是否包含關鍵詞
                            if "to_csv" in line_lower or "train_cleaned" in line_lower or "test_cleaned" in line_lower:
                                start_line = max(0, i-1)
                                end_line = min(len(lines), i+2)
                                print(f"\n相關代碼 (行 {start_line+1}-{end_line}):")
                                for j in range(start_line, end_line):
                                    print(f"  {j+1}: {lines[j]}")
                else:
                    print("未找到相關關鍵詞")
            
            except Exception as e:
                print(f"讀取檔案時發生錯誤：{e}")
        
        print("\n=== 掃描完成 ===")
    
    except Exception as e:
        print(f"掃描過程中發生錯誤：{e}")

if __name__ == "__main__":
    scan_python_files() 