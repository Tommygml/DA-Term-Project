"""
feature_eng_04.py - 特徵工程管道整合 (修正版)
功能：整合三個階段的特徵選擇結果，產生最終特徵集，確保正確包含目標變數
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
        self.target_column = None
        
    def load_all_results(self):
        """載入所有階段的特徵選擇結果"""
        try:
            # 載入清理後的原始數據
            self.train_cleaned = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            self.test_cleaned = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
            # 識別目標變數
            self.target_column = self.train_cleaned.columns[-1]
            print(f"目標變數: {self.target_column}")
            
            # 載入各階段選擇的特徵
            filter_df = pd.read_csv(WORK_DIR / 'filter_selected_features.csv')
            self.filter_features = filter_df['feature'].tolist()
            
            embedded_df = pd.read_csv(WORK_DIR / 'embedded_selected_features.csv')
            self.embedded_features = embedded_df['feature'].tolist()
            
            wrapper_df = pd.read_csv(WORK_DIR / 'wrapper_selected_features.csv')
            self.wrapper_features = wrapper_df['feature'].tolist()
            
            print("成功載入所有階段的結果")
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
        original_features = set(all_features)
        
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
            if feature not in self.final_features:
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
        train_final = self.train_cleaned[self.final_features + [self.target_column]]
        test_final = self.test_cleaned[self.final_features + [self.target_column]]
        
        # 5. 報告最終結果
        print(f"\n最終特徵數量: {len(self.final_features)}")
        print(f"最終數據集欄位數: {train_final.shape[1]} (包含目標變數)")
        print(f"特徵縮減比例: {len(self.final_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%")
        
        # 6. 保存最終數據集
        train_final.to_csv(WORK_DIR / 'train_final.csv', index=False)
        test_final.to_csv(WORK_DIR / 'test_final.csv', index=False)
        print("已儲存最終數據集: train_final.csv, test_final.csv")
        
        # 7. 打印最終數據集的欄位
        print("\n最終數據集欄位:")
        for i, column in enumerate(train_final.columns, 1):
            col_type = "特徵" if column != self.target_column else "目標變數"
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
        
        # 特徵選擇熱力圖
        # 創建特徵選擇矩陣
        all_features = [col for col in self.train_cleaned.columns if col != self.target_column]
        selection_matrix = []
        
        for feature in all_features:
            row = [
                1,  # 原始
                1 if feature in self.filter_features else 0,
                1 if feature in self.embedded_features else 0,
                1 if feature in self.wrapper_features else 0,
                1 if feature in self.final_features else 0
            ]
            selection_matrix.append(row)
        
        selection_df = pd.DataFrame(
            selection_matrix,
            index=all_features,
            columns=['原始', 'Filter', 'Embedded', 'Wrapper', '最終']
        )
        
        # 只顯示前30個特徵的熱力圖（避免圖太大）
        plt.figure(figsize=(10, 14))
        sns.heatmap(selection_df.iloc[:30], 
                   cmap='YlOrRd', cbar_kws={'label': '選中(1) / 未選中(0)'},
                   yticklabels=True, xticklabels=True)
        plt.title('特徵選擇狀態熱力圖（前30個特徵）')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_selection_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特徵保留率圓餅圖
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
        
        # 視覺化衍生特徵的添加情況
        derived_features = [f for f in self.final_features if f not in self.wrapper_features]
        if derived_features:
            wrapper_count = len(self.wrapper_features)
            derived_count = len(derived_features)
            
            plt.figure(figsize=(8, 6))
            plt.bar(['Wrapper特徵', '衍生特徵'], [wrapper_count, derived_count], 
                   color=['royalblue', 'indianred'])
            plt.title('最終特徵構成', fontsize=14)
            plt.ylabel('特徵數量', fontsize=12)
            
            # 添加數值標籤
            for i, count in enumerate([wrapper_count, derived_count]):
                plt.text(i, count + 0.2, str(count), ha='center', fontsize=12)
                
            plt.tight_layout()
            plt.savefig(viz_dir / 'feature_composition.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_feature_importance_summary(self):
        """生成特徵重要性綜合摘要"""
        try:
            # 載入各階段的分數資訊
            filter_scores = pd.read_csv(WORK_DIR / 'filter_feature_scores.csv')
            embedded_scores = pd.read_csv(WORK_DIR / 'embedded_feature_scores.csv')
            wrapper_details = pd.read_csv(WORK_DIR / 'wrapper_rfe_details.csv')
            
            # 整合所有資訊
            summary = []
            for feature in self.final_features:
                summary_dict = {'feature': feature}
                
                # 標記是原始特徵還是衍生特徵
                is_derived = feature not in self.wrapper_features
                summary_dict['is_derived'] = 'Yes' if is_derived else 'No'
                
                # Filter階段分數
                filter_row = filter_scores[filter_scores['feature'] == feature]
                if not filter_row.empty:
                    if 'mutual_info' in filter_row.columns:
                        summary_dict['mi_score'] = filter_row['mutual_info'].values[0]
                
                # Embedded階段分數
                embedded_row = embedded_scores[embedded_scores['feature'] == feature]
                if not embedded_row.empty:
                    if 'combined_score' in embedded_row.columns:
                        summary_dict['combined_score'] = embedded_row['combined_score'].values[0]
                    if 'rf_score' in embedded_row.columns:
                        summary_dict['rf_importance'] = embedded_row['rf_score'].values[0]
                
                # Wrapper階段排名
                wrapper_row = wrapper_details[wrapper_details['feature'] == feature]
                if not wrapper_row.empty:
                    if 'ranking' in wrapper_row.columns:
                        summary_dict['rfe_ranking'] = wrapper_row['ranking'].values[0]
                
                summary.append(summary_dict)
            
            summary_df = pd.DataFrame(summary)
            
            # 如果存在RFE排名，按排名排序
            if 'rfe_ranking' in summary_df.columns:
                summary_df = summary_df.sort_values('rfe_ranking')
            
            summary_df.to_csv(WORK_DIR / 'final_feature_importance_summary.csv', index=False)
            
            print("\n特徵重要性摘要已儲存至: final_feature_importance_summary.csv")
            if not summary_df.empty:
                print("前5個最重要的特徵:")
                print(summary_df.head().to_string())
            
            return summary_df
        
        except Exception as e:
            print(f"生成特徵重要性摘要時發生錯誤: {e}")
            print("將跳過此步驟並繼續執行後續操作")
            return pd.DataFrame({'feature': self.final_features})
    
    def save_pipeline_config(self):
        """儲存特徵工程管道配置"""
        try:
            # 判斷是否有衍生特徵添加
            derived_features = [f for f in self.final_features if f not in self.wrapper_features]
            
            config = {
                'pipeline_stages': {
                    'stage_1': {
                        'name': 'Filter',
                        'methods': ['chi2', 'anova', 'mutual_info'],
                        'input_features': len(self.train_cleaned.columns) - 1,
                        'output_features': len(self.filter_features)
                    },
                    'stage_2': {
                        'name': 'Embedded',
                        'methods': ['lasso', 'random_forest', 'elastic_net'],
                        'input_features': len(self.filter_features),
                        'output_features': len(self.embedded_features)
                    },
                    'stage_3': {
                        'name': 'Wrapper',
                        'methods': ['rfe_cv'],
                        'input_features': len(self.embedded_features),
                        'output_features': len(self.wrapper_features)
                    }
                },
                'final_features': self.final_features,
                'wrapper_features': self.wrapper_features,
                'derived_features': derived_features,
                'target_column': self.target_column,
                'total_reduction': f"{(1 - len(self.final_features)/(len(self.train_cleaned.columns)-1))*100:.1f}%",
                'dataset_info': {
                    'train_rows': len(self.train_cleaned),
                    'test_rows': len(self.test_cleaned),
                    'final_columns': len(self.final_features) + 1
                }
            }
            
            with open(WORK_DIR / 'feature_engineering_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            print("\n特徵工程配置已儲存至: feature_engineering_config.json")
        
        except Exception as e:
            print(f"儲存管道配置時發生錯誤: {e}")
            print("將跳過此步驟並繼續執行後續操作")
    
    def generate_final_report(self):
        """生成最終報告"""
        try:
            report = []
            report.append("=== 特徵工程管道整合報告 ===\n")
            
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
            
            # 各階段移除的特徵
            report.append("\n=== 各階段移除的特徵 ===")
            
            all_features = set(col for col in self.train_cleaned.columns if col != self.target_column)
            filter_removed = all_features - set(self.filter_features)
            embedded_removed = set(self.filter_features) - set(self.embedded_features)
            wrapper_removed = set(self.embedded_features) - set(self.wrapper_features)
            
            report.append(f"Filter階段移除：{len(filter_removed)} 個特徵")
            report.append(f"Embedded階段移除：{len(embedded_removed)} 個特徵")
            report.append(f"Wrapper階段移除：{len(wrapper_removed)} 個特徵")
            
            # 衍生特徵
            derived_features = [f for f in self.final_features if f not in self.wrapper_features]
            if derived_features:
                report.append(f"\n=== 添加的衍生特徵 ({len(derived_features)}) ===")
                for i, feature in enumerate(derived_features, 1):
                    report.append(f"{i}. {feature}")
            
            # 最終特徵列表
            report.append(f"\n=== 最終保留的特徵 ({len(self.final_features)}) ===")
            for i, feature in enumerate(self.final_features, 1):
                feature_type = "衍生特徵" if feature in derived_features else "原始特徵"
                report.append(f"{i}. {feature} ({feature_type})")
            
            # 儲存報告
            with open(WORK_DIR / 'feature_engineering_final_report.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print("\n最終報告已儲存至: feature_engineering_final_report.txt")
        
        except Exception as e:
            print(f"生成最終報告時發生錯誤: {e}")
            print("將跳過此步驟並繼續執行後續操作")
    
    def run_pipeline_integration(self):
        """執行特徵工程管道整合"""
        print("=== 開始特徵工程管道整合 (修正版) ===")
        print("本版本會檢查並正確處理衍生特徵和目標變數\n")
        
        # 載入所有結果
        if not self.load_all_results():
            return
        
        # 分析特徵選擇歷程
        self.analyze_feature_journey()
        
        # 創建最終特徵集
        self.create_final_features()
        
        # 視覺化結果
        self.visualize_feature_reduction()
        
        # 生成特徵重要性摘要
        self.generate_feature_importance_summary()
        
        # 儲存管道配置
        self.save_pipeline_config()
        
        # 生成最終報告
        self.generate_final_report()
        
        print("\n=== 特徵工程管道整合完成 ===")
        print(f"最終特徵數量：{len(self.final_features)}")
        print(f"包含目標變數的最終數據集欄位數: {len(self.final_features) + 1}")
        print("主要輸出檔案：")
        print("- train_final.csv (包含特徵和目標變數)")
        print("- test_final.csv (包含特徵和目標變數)")
        print("- final_feature_importance_summary.csv")
        print("- feature_engineering_config.json")
        print("- feature_engineering_final_report.txt")
        print("\n您現在可以使用 train_final.csv 來訓練模型了")

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run_pipeline_integration()
