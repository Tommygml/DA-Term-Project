"""
feature_eng_04_hotfix.py - 緊急修復版特徵工程管道整合
功能：整合三個階段的特徵選擇結果，產生最終包含正確目標變數的特徵集
作者：Tommy
日期：2025-05-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

class FeatureEngineeringPipeline:
    def __init__(self):
        self.train_cleaned = None
        self.test_cleaned = None
        self.filter_features = []
        self.embedded_features = []
        self.wrapper_features = []
        self.final_features = []
        self.feature_history = {}
        self.target_column = None  # 將手動設置
        
    def load_all_results(self):
        """載入所有階段的特徵選擇結果"""
        try:
            # 載入清理後的原始數據
            self.train_cleaned = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            self.test_cleaned = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
            # 顯示所有欄位供用戶檢查
            print("train_cleaned.csv 的所有欄位:")
            for i, col in enumerate(self.train_cleaned.columns):
                print(f"{i+1}. {col}")
            
            # ===== 手動設置目標變數 =====
            # 尋找名為 "Target" 的欄位
            found_target = False
            for col in self.train_cleaned.columns:
                if col.lower() == 'target':
                    self.target_column = col
                    found_target = True
                    break
            
            # 如果找不到，請用戶手動指定
            if not found_target:
                print("\n無法自動找到名為 'Target' 的欄位")
                target_idx = int(input("請輸入目標變數的欄位索引 (從1開始): ")) - 1
                self.target_column = self.train_cleaned.columns[target_idx]
            
            print(f"\n選擇的目標變數: {self.target_column}")
            print(f"目標變數的唯一值:")
            print(self.train_cleaned[self.target_column].value_counts())
            
            # 載入各階段選擇的特徵
            filter_df = pd.read_csv(WORK_DIR / 'filter_selected_features.csv')
            self.filter_features = filter_df['feature'].tolist()
            
            embedded_df = pd.read_csv(WORK_DIR / 'embedded_selected_features.csv')
            self.embedded_features = embedded_df['feature'].tolist()
            
            wrapper_df = pd.read_csv(WORK_DIR / 'wrapper_selected_features.csv')
            self.wrapper_features = wrapper_df['feature'].tolist()
            
            print("\n成功載入所有階段的結果")
            print(f"原始特徵數量：{len(self.train_cleaned.columns) - 1}")
            print(f"Filter階段：{len(self.filter_features)} 個特徵")
            print(f"Embedded階段：{len(self.embedded_features)} 個特徵")
            print(f"Wrapper階段：{len(self.wrapper_features)} 個特徵")
            
            return True
        except Exception as e:
            print(f"載入結果時發生錯誤：{e}")
            return False
    
    def identify_derived_features(self):
        """識別在數據清理階段可能創建的衍生特徵"""
        all_features = [col for col in self.train_cleaned.columns if col != self.target_column]
        
        # 檢查是否有潛在的衍生特徵（包含特定名稱模式的特徵）
        derived_features = []
        known_patterns = ['_Success_Rate', '_Ratio', '_Derived', '_Rate', '_Mean', '_Sum', '_Diff']
        
        for feature in all_features:
            # 檢查是否符合衍生特徵命名模式
            if any(pattern in feature for pattern in known_patterns):
                derived_features.append(feature)
        
        if derived_features:
            print(f"\n發現可能的衍生特徵：")
            for feature in derived_features:
                print(f"- {feature}")
        
        return derived_features
    
    def analyze_feature_journey(self):
        """分析特徵在各階段的選擇情況"""
        print("\n=== 特徵選擇歷程分析 ===")
        
        # 獲取所有特徵（不包含目標變數）
        all_features = [col for col in self.train_cleaned.columns if col != self.target_column]
        
        # 記錄每個特徵的選擇歷程
        for feature in all_features:
            self.feature_history[feature] = {
                'original': 1,
                'filter': 1 if feature in self.filter_features else 0,
                'embedded': 1 if feature in self.embedded_features else 0,
                'wrapper': 1 if feature in self.wrapper_features else 0
            }
        
        # 統計分析
        survival_patterns = {}
        for feature, history in self.feature_history.items():
            pattern = f"{history['filter']}{history['embedded']}{history['wrapper']}"
            if pattern not in survival_patterns:
                survival_patterns[pattern] = []
            survival_patterns[pattern].append(feature)
        
        print("\n特徵選擇模式分析（1=選中，0=未選中）：")
        for pattern, features in sorted(survival_patterns.items(), reverse=True):
            print(f"模式 {pattern}: {len(features)} 個特徵")
            if pattern == "111":
                print(f"  完整保留的特徵示例：{features[:5]}")
            elif pattern == "110":
                print(f"  在Wrapper階段被移除的特徵示例：{features[:5]}")
            elif pattern == "100":
                print(f"  只通過Filter階段的特徵示例：{features[:5]}")
            elif pattern == "000":
                print(f"  沒有通過任何階段的特徵: {features}")
    
    def create_final_features(self):
        """創建最終特徵集，確保正確包含衍生特徵和目標變數"""
        print("\n=== 創建最終特徵集 ===")
        
        # 1. 基於 Wrapper 階段結果創建初始特徵集
        self.final_features = self.wrapper_features.copy()
        print(f"Wrapper階段選擇的特徵數量: {len(self.final_features)}")
        
        # 2. 識別並添加衍生特徵
        derived_features = self.identify_derived_features()
        derived_to_add = []
        
        # 檢查這些衍生特徵是否已包含在最終特徵集中
        for feature in derived_features:
            if feature not in self.final_features and feature != self.target_column:
                derived_to_add.append(feature)
        
        if derived_to_add:
            print(f"\n添加 {len(derived_to_add)} 個衍生特徵到最終特徵集:")
            for feature in derived_to_add:
                print(f"- {feature}")
            self.final_features.extend(derived_to_add)
        
        # 3. 確保目標變數不在特徵集中
        if self.target_column in self.final_features:
            print(f"警告: 目標變數 '{self.target_column}' 出現在特徵集中，正在移除...")
            self.final_features.remove(self.target_column)
        
        # 4. 創建最終數據集（包含特徵和目標變數）
        # 確保目標變數存在於清理後的數據集中
        if self.target_column not in self.train_cleaned.columns:
            print(f"錯誤: 目標變數 '{self.target_column}' 不存在於 train_cleaned.csv 中")
            return
        
        train_final = self.train_cleaned[self.final_features + [self.target_column]]
        test_final = self.test_cleaned[self.final_features + [self.target_column]]
        
        # 5. 報告最終結果
        print(f"\n最終特徵數量: {len(self.final_features)}")
        print(f"最終數據集欄位數: {train_final.shape[1]} (包含目標變數)")
        
        # 6. 保存最終數據集
        train_final.to_csv(WORK_DIR / 'train_final.csv', index=False)
        test_final.to_csv(WORK_DIR / 'test_final.csv', index=False)
        print("已儲存最終數據集: train_final.csv, test_final.csv")
        
        # 7. 打印最終數據集的欄位
        print("\n最終數據集欄位:")
        for i, column in enumerate(train_final.columns, 1):
            col_type = "目標變數" if column == self.target_column else "特徵"
            print(f"{i}. {column} ({col_type})")
        
        # 8. 更新和保存最終特徵選擇結果
        final_features_df = pd.DataFrame({
            'feature': self.final_features,
            'selected': 1
        })
        final_features_df.to_csv(WORK_DIR / 'final_selected_features.csv', index=False)
        print("已更新最終特徵選擇結果文件: final_selected_features.csv")
    
    def visualize_feature_reduction(self):
        """視覺化特徵縮減過程"""
        viz_dir = WORK_DIR / 'feature_engineering_results'
        viz_dir.mkdir(exist_ok=True)
        
        # 特徵數量變化圖
        stages = ['原始', 'Filter', 'Embedded', 'Wrapper', '最終']
        feature_counts = [
            len(self.train_cleaned.columns) - 1,  # 減去目標變數
            len(self.filter_features),
            len(self.embedded_features),
            len(self.wrapper_features),
            len(self.final_features)
        ]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(stages, feature_counts, color=['blue', 'green', 'orange', 'red', 'purple'])
        
        # 在條形圖上添加數值
        for bar, count in zip(bars, feature_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('特徵選擇階段', fontsize=12)
        plt.ylabel('特徵數量', fontsize=12)
        plt.title('特徵工程各階段特徵數量變化', fontsize=14)
        plt.ylim(0, max(feature_counts) * 1.1)
        
        # 添加百分比標註
        for i in range(1, len(feature_counts)):
            reduction = (feature_counts[i-1] - feature_counts[i]) / feature_counts[i-1] * 100
            plt.text(i-0.5, max(feature_counts)*0.9, f'-{reduction:.1f}%',
                    ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_reduction_process.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特徵保留率圓餅圖
        all_features = [col for col in self.train_cleaned.columns if col != self.target_column]
        final_retained = len(self.final_features)
        final_removed = len(all_features) - final_retained
        
        plt.figure(figsize=(9, 9))
        plt.pie([final_retained, final_removed], 
               labels=[f'保留的特徵\n({final_retained}個)', 
                      f'移除的特徵\n({final_removed}個)'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'fontsize': 12})
        plt.title(f'最終特徵保留情況\n總特徵數：{len(all_features)}', fontsize=14)
        plt.tight_layout()
        plt.savefig(viz_dir / 'final_feature_retention.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self):
        """生成最終報告"""
        try:
            report = []
            report.append("=== 特徵工程管道整合報告 (修正版) ===\n")
            
            # 概要統計
            report.append("=== 特徵數量變化 ===")
            report.append(f"原始特徵數量：{len(self.train_cleaned.columns) - 1}")
            report.append(f"Filter階段後：{len(self.filter_features)} ({len(self.filter_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
            report.append(f"Embedded階段後：{len(self.embedded_features)} ({len(self.embedded_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
            report.append(f"Wrapper階段後：{len(self.wrapper_features)} ({len(self.wrapper_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
            report.append(f"最終特徵數量：{len(self.final_features)} ({len(self.final_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
            report.append(f"總體縮減率：{(1 - len(self.final_features)/(len(self.train_cleaned.columns)-1))*100:.1f}%")
            
            # 資料集統計
            report.append("\n=== 資料集統計 ===")
            report.append(f"訓練集筆數: {len(self.train_cleaned)}")
            report.append(f"測試集筆數: {len(self.test_cleaned)}")
            report.append(f"目標變數: {self.target_column}")
            
            # 目標變數分佈
            report.append("\n=== 目標變數分佈 ===")
            target_dist = self.train_cleaned[self.target_column].value_counts()
            for value, count in target_dist.items():
                percentage = (count / len(self.train_cleaned)) * 100
                report.append(f"{value}: {count} ({percentage:.1f}%)")
            
            # 最終特徵列表
            report.append(f"\n=== 最終保留的特徵 ({len(self.final_features)}) ===")
            for i, feature in enumerate(self.final_features, 1):
                is_derived = any(pattern in feature for pattern in ['_Success_Rate', '_Ratio', '_Derived'])
                feature_type = "衍生特徵" if is_derived else "原始特徵"
                report.append(f"{i}. {feature} ({feature_type})")
            
            # 儲存報告
            with open(WORK_DIR / 'feature_engineering_final_report_fixed.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print("\n最終報告已儲存至: feature_engineering_final_report_fixed.txt")
        
        except Exception as e:
            print(f"生成最終報告時發生錯誤: {e}")
            print("將跳過此步驟並繼續執行後續操作")
    
    def run_pipeline_integration(self):
        """執行特徵工程管道整合"""
        print("=== 開始特徵工程管道整合 (緊急修復版) ===")
        print("本版本將手動識別並設置正確的目標變數\n")
        
        # 載入所有結果
        if not self.load_all_results():
            return
        
        # 分析特徵選擇歷程
        self.analyze_feature_journey()
        
        # 創建最終特徵集
        self.create_final_features()
        
        # 視覺化結果
        self.visualize_feature_reduction()
        
        # 生成最終報告
        self.generate_final_report()
        
        print("\n=== 特徵工程管道整合 (修復版) 完成 ===")
        print(f"最終特徵數量: {len(self.final_features)}")
        print(f"目標變數: {self.target_column}")
        print(f"最終數據集欄位數: {len(self.final_features) + 1} (包含目標變數)")
        print("\n主要輸出檔案:")
        print("- train_final.csv (包含特徵和目標變數)")
        print("- test_final.csv (包含特徵和目標變數)")
        print("- feature_engineering_final_report_fixed.txt")
        print("\n您現在可以使用 train_final.csv 來訓練模型了")

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run_pipeline_integration()
