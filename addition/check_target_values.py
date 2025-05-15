"""
check_target_values.py - 檢查目標變數異常值
功能：檢查訓練集和測試集中目標變數是否有非預期的值
作者：AI助手
日期：2023-05-25
"""

import pandas as pd
from pathlib import Path

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

# 用於存儲所有問題數據的全局列表
problem_records = []

def check_data_file(file_path, expected_values=['Graduate', 'Enrolled', 'Dropout']):
    """檢查數據文件中目標變數的值"""
    print(f"\n檢查文件: {file_path}")
    
    try:
        # 讀取CSV文件，注意分隔符是分號
        df = pd.read_csv(WORK_DIR / file_path, sep=';')
        print(f"成功載入數據集，共 {len(df)} 筆資料")
        
        # 假設目標變數是最後一列
        target_col = df.columns[-1]
        print(f"目標變數列名: {target_col}")
        
        # 獲取目標變數的唯一值
        unique_values = df[target_col].unique()
        print(f"目標變數唯一值數量: {len(unique_values)}")
        
        # 檢查是否有非預期的值
        unexpected_values = [val for val in unique_values if val not in expected_values]
        if unexpected_values:
            print(f"發現非預期的目標變數值，數量: {len(unexpected_values)}")
            
            # 收集具有非預期值的行
            for val in unexpected_values:
                rows = df[df[target_col] == val]
                for idx, row in rows.iterrows():
                    problem_records.append({
                        'file': file_path,
                        'row_index': idx,
                        'target_value': val,
                        'value_type': type(val).__name__,
                        'data': row.to_dict()
                    })
        else:
            print("未發現非預期的目標變數值，所有值都在預期範圍內。")
        
        return True
    except Exception as e:
        print(f"檢查數據時發生錯誤: {e}")
        return False

def check_cleaned_data_file(file_path, expected_values=['Graduate', 'Enrolled', 'Dropout']):
    """檢查清洗後的數據文件中目標變數的值"""
    print(f"\n檢查已清洗文件: {file_path}")
    
    try:
        # 讀取CSV文件
        df = pd.read_csv(WORK_DIR / file_path)
        print(f"成功載入數據集，共 {len(df)} 筆資料")
        
        # 假設目標變數是最後一列
        target_col = df.columns[-1]
        print(f"目標變數列名: {target_col}")
        
        # 獲取目標變數的唯一值
        unique_values = df[target_col].unique()
        print(f"目標變數唯一值數量: {len(unique_values)}")
        
        # 檢查是否有非預期的值
        unexpected_values = [val for val in unique_values if val not in expected_values]
        if unexpected_values:
            print(f"發現非預期的目標變數值，數量: {len(unexpected_values)}")
            
            # 收集具有非預期值的行
            for val in unexpected_values:
                rows = df[df[target_col] == val]
                for idx, row in rows.iterrows():
                    problem_records.append({
                        'file': file_path,
                        'row_index': idx,
                        'target_value': val,
                        'value_type': type(val).__name__,
                        'data': row.to_dict()
                    })
        else:
            print("未發現非預期的目標變數值，所有值都在預期範圍內。")
        
        return True
    except Exception as e:
        print(f"檢查數據時發生錯誤: {e}")
        return False

def summarize_problems():
    """匯總所有問題數據"""
    if not problem_records:
        print("\n=== 未發現任何問題數據 ===")
        return
    
    print("\n" + "="*80)
    print("問題數據匯總報告".center(80))
    print("="*80)
    print(f"總計發現 {len(problem_records)} 筆問題數據")
    
    # 按文件分組顯示問題數
    file_counts = {}
    for record in problem_records:
        file_name = record['file']
        if file_name not in file_counts:
            file_counts[file_name] = 0
        file_counts[file_name] += 1
    
    print("\n各文件問題數量:")
    for file_name, count in file_counts.items():
        print(f"  {file_name}: {count} 筆")
    
    # 顯示每筆問題數據的簡要信息
    print("\n問題數據明細:")
    for i, record in enumerate(problem_records, 1):
        print(f"\n{i}. 文件: {record['file']}")
        print(f"   行索引: {record['row_index']}")
        print(f"   目標值: {record['target_value']} (類型: {record['value_type']})")
        
        # 只顯示目標變數和關鍵字段
        target_col = list(record['data'].keys())[-1]
        key_data = {
            '目標變數名': target_col,
            '目標變數值': record['data'][target_col]
        }
        print(f"   關鍵數據: {key_data}")
    
    # 保存問題數據到CSV文件
    try:
        problem_rows = []
        for record in problem_records:
            row = {
                'file': record['file'],
                'row_index': record['row_index'],
                'target_value': record['target_value'],
                'value_type': record['value_type']
            }
            # 添加原始數據的所有列
            for k, v in record['data'].items():
                row[f'data_{k}'] = v
            problem_rows.append(row)
        
        problem_df = pd.DataFrame(problem_rows)
        output_file = WORK_DIR / 'problem_records.csv'
        problem_df.to_csv(output_file, index=False)
        print(f"\n問題數據已保存到: {output_file}")
    except Exception as e:
        print(f"\n保存問題數據時發生錯誤: {e}")

def main():
    """主程式"""
    print("=== 開始檢查目標變數 ===")
    
    # 檢查原始數據集
    check_data_file('data_train.csv')
    check_data_file('data_test.csv')
    
    # 檢查清洗後的數據集
    check_cleaned_data_file('train_cleaned.csv')
    check_cleaned_data_file('test_cleaned.csv')
    
    # 匯總所有問題
    summarize_problems()
    
    print("\n=== 檢查完成 ===")

if __name__ == "__main__":
    main() 