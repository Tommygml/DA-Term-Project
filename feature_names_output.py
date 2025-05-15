"""
feature_names_output.py - 特徵選擇結果分析
功能：讀取並輸出不同特徵選擇方法所選擇的特徵名稱
作者：Tommy
日期：2025-05-16
"""

import pandas as pd
from pathlib import Path

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

# 要讀取的文件列表
feature_files = [
    "filter_selected_features.csv",
    "wrapper_selected_features.csv",
    "embedded_selected_features.csv",
    "final_selected_features.csv"
]

def read_selected_features(file_name):
    """讀取CSV文件並返回被選中的特徵名稱"""
    try:
        # 嘗試讀取CSV文件
        file_path = WORK_DIR / file_name
        if not file_path.exists():
            print(f"錯誤: 文件 {file_name} 不存在")
            return None
        
        # 讀取CSV文件
        df = pd.read_csv(file_path)
        
        # 檢查文件格式
        if 'feature' in df.columns and 'selected' in df.columns:
            # 獲取selected為True或1的特徵
            selected_features = df[df['selected'].isin([True, 1, '1', 'True'])]['feature'].tolist()
            return selected_features
        else:
            # 假設這是直接包含特徵的文件
            return df.columns.tolist()
    
    except Exception as e:
        print(f"讀取 {file_name} 時發生錯誤: {e}")
        return None

def main():
    """主函數，讀取並顯示各個特徵選擇文件中被選中的特徵名稱"""
    print("=== 特徵選擇結果分析 ===\n")
    
    # 保存所有方法的特徵選擇結果
    all_selected_features = {}
    
    # 遍歷並處理每個文件
    for file_name in feature_files:
        method_name = file_name.replace("_selected_features.csv", "")
        print(f"方法: {method_name}")
        
        selected_features = read_selected_features(file_name)
        
        if selected_features:
            all_selected_features[method_name] = selected_features
            print(f"選中特徵數量: {len(selected_features)}")
            print("選中的特徵:")
            for i, feature in enumerate(selected_features, 1):
                print(f"{i}. {feature}")
            print()  # 空行
        else:
            print(f"無法讀取 {file_name} 中的特徵\n")
    
    # 印出對比分析
    if len(all_selected_features) >= 2:
        print("=== 特徵選擇方法對比 ===")
        methods = list(all_selected_features.keys())
        
        # 比較不同方法之間的共同特徵
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                method1_features = set(all_selected_features[method1])
                method2_features = set(all_selected_features[method2])
                common_features = method1_features.intersection(method2_features)
                
                print(f"{method1} 和 {method2} 共有 {len(common_features)} 個共同特徵")
                if common_features:
                    print("共同特徵：")
                    for feature in common_features:
                        print(f"  - {feature}")
                print()
        
        # 分析最終選擇的特徵
        if "final" in all_selected_features:
            print("=== 最終特徵分析 ===")
            final_features = set(all_selected_features["final"])
            print(f"最終選擇的特徵數量: {len(final_features)}")
            
            # 檢查每個特徵來自哪個方法
            for method, features in all_selected_features.items():
                if method != "final":
                    method_features = set(features)
                    contained = final_features.intersection(method_features)
                    unique = contained - (final_features - method_features)
                    
                    print(f"來自 {method} 方法的特徵數量: {len(contained)}")
                    if unique:
                        print(f"僅在 {method} 方法中發現的特徵數量: {len(unique)}")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main() 