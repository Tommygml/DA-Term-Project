"""
feature_eng_04.py - 特徵工程管道整合
功能：整合三個階段的特徵選擇結果，產生最終特徵集
作者：Tommy
日期：2025-05-15
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
        
    def load_all_results(self):
        """載入所有階段的特徵選擇結果"""
        try:
            # 載入清理後的原始數據
            self.train_cleaned = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            self.test_cleaned = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
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
    
    def analyze_feature_journey(self):
        """分析特徵在各階段的選擇情況"""
        print("\n=== 特徵選擇歷程分析 ===")
        
        # 獲取所有特徵（不包含目標變數）
        all_features = [col for col in self.train_cleaned.columns if col != self.train_cleaned.columns[-1]]
        
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
    
    def create_final_features(self):
        """創建最終特徵集"""
        print("\n=== 創建最終特徵集 ===")
        
        # 使用Wrapper階段的特徵作為最終特徵集
        self.final_features = self.wrapper_features.copy()
        
        # 定義原始特徵集
        self.original_features = [col for col in self.train_cleaned.columns if col != self.train_cleaned.columns[-1]]
        
        print(f"最終特徵數量：{len(self.final_features)}")
        print(f"特徵縮減比例：{len(self.final_features)/len(self.original_features)*100:.1f}%")
        
        # 保存最終數據集 - 只保留選中的特徵和目標變量
        target_column = self.train_cleaned.columns[-1]
        train_final = self.train_cleaned[self.final_features + [target_column]]
        test_final = self.test_cleaned[self.final_features + [target_column]]
        
        train_final.to_csv(WORK_DIR / 'train_final.csv', index=False)
        test_final.to_csv(WORK_DIR / 'test_final.csv', index=False)
        
        print("已儲存最終數據集：train_final.csv, test_final.csv")
        
        # 儲存最終特徵集
        final_features_df = pd.DataFrame({
            'feature': self.final_features,
            'selected': 1
        })
        final_features_df.to_csv(WORK_DIR / 'final_selected_features.csv', index=False)
    
    def visualize_feature_reduction(self):
        """視覺化特徵縮減過程"""
        viz_dir = WORK_DIR / 'feature_engineering_results'
        viz_dir.mkdir(exist_ok=True)
        
        # 特徵數量變化圖
        stages = ['原始', 'Filter', 'Embedded', 'Wrapper(最終)']
        feature_counts = [
            len(self.train_cleaned.columns) - 1,  # 減去目標變數
            len(self.filter_features),
            len(self.embedded_features),
            len(self.wrapper_features)
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(stages, feature_counts, color=['blue', 'green', 'orange', 'red'])
        
        # 在條形圖上添加數值
        for bar, count in zip(bars, feature_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('特徵選擇階段')
        plt.ylabel('特徵數量')
        plt.title('特徵工程各階段特徵數量變化')
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
        all_features = [col for col in self.train_cleaned.columns if col != self.train_cleaned.columns[-1]]
        selection_matrix = []
        
        for feature in all_features:
            row = [
                1,  # 原始
                1 if feature in self.filter_features else 0,
                1 if feature in self.embedded_features else 0,
                1 if feature in self.wrapper_features else 0
            ]
            selection_matrix.append(row)
        
        selection_df = pd.DataFrame(
            selection_matrix,
            index=all_features,
            columns=['原始', 'Filter', 'Embedded', 'Wrapper']
        )
        
        # 只顯示前30個特徵的熱力圖（避免圖太大）
        plt.figure(figsize=(8, 12))
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
        
        plt.figure(figsize=(8, 8))
        plt.pie([final_retained, final_removed], 
               labels=[f'保留的特徵\n({final_retained}個)', 
                      f'移除的特徵\n({final_removed}個)'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90)
        plt.title(f'最終特徵保留情況\n總特徵數：{len(all_features)}')
        plt.tight_layout()
        plt.savefig(viz_dir / 'final_feature_retention.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_feature_importance_summary(self):
        """生成特徵重要性綜合摘要"""
        # 載入各階段的分數資訊
        filter_scores = pd.read_csv(WORK_DIR / 'filter_feature_scores.csv')
        embedded_scores = pd.read_csv(WORK_DIR / 'embedded_feature_scores.csv')
        wrapper_details = pd.read_csv(WORK_DIR / 'wrapper_rfe_details.csv')
        
        # 整合所有資訊
        summary = []
        for feature in self.final_features:
            summary_dict = {'feature': feature}
            
            # Filter階段分數
            filter_row = filter_scores[filter_scores['feature'] == feature]
            if not filter_row.empty:
                summary_dict['mi_score'] = filter_row['mutual_info'].values[0]
            
            # Embedded階段分數
            embedded_row = embedded_scores[embedded_scores['feature'] == feature]
            if not embedded_row.empty:
                summary_dict['combined_score'] = embedded_row['combined_score'].values[0]
                summary_dict['rf_importance'] = embedded_row['rf_score'].values[0]
            
            # Wrapper階段排名
            wrapper_row = wrapper_details[wrapper_details['feature'] == feature]
            if not wrapper_row.empty:
                summary_dict['rfe_ranking'] = wrapper_row['ranking'].values[0]
            
            summary.append(summary_dict)
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('rfe_ranking')
        summary_df.to_csv(WORK_DIR / 'final_feature_importance_summary.csv', index=False)
        
        return summary_df
    
    def save_pipeline_config(self):
        """儲存特徵工程管道配置"""
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
            'total_reduction': f"{(1 - len(self.final_features)/(len(self.train_cleaned.columns)-1))*100:.1f}%"
        }
        
        with open(WORK_DIR / 'feature_engineering_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    def generate_final_report(self):
        """生成最終報告"""
        report = []
        report.append("=== 特徵工程管道整合報告 ===\n")
        
        # 概要統計
        report.append("=== 特徵數量變化 ===")
        report.append(f"原始特徵數量：{len(self.train_cleaned.columns) - 1}")
        report.append(f"Filter階段後：{len(self.filter_features)} ({len(self.filter_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
        report.append(f"Embedded階段後：{len(self.embedded_features)} ({len(self.embedded_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
        report.append(f"Wrapper階段後：{len(self.wrapper_features)} ({len(self.wrapper_features)/(len(self.train_cleaned.columns)-1)*100:.1f}%)")
        report.append(f"最終特徵數量：{len(self.final_features)}")
        report.append(f"總體縮減率：{(1 - len(self.final_features)/(len(self.train_cleaned.columns)-1))*100:.1f}%")
        
        # 各階段移除的特徵
        report.append("\n=== 各階段移除的特徵 ===")
        
        all_features = set(col for col in self.train_cleaned.columns if col != self.train_cleaned.columns[-1])
        filter_removed = all_features - set(self.filter_features)
        embedded_removed = set(self.filter_features) - set(self.embedded_features)
        wrapper_removed = set(self.embedded_features) - set(self.wrapper_features)
        
        report.append(f"Filter階段移除：{len(filter_removed)} 個特徵")
        report.append(f"Embedded階段移除：{len(embedded_removed)} 個特徵")
        report.append(f"Wrapper階段移除：{len(wrapper_removed)} 個特徵")
        
        # 最終特徵列表
        report.append("\n=== 最終保留的特徵 ===")
        for i, feature in enumerate(self.final_features, 1):
            report.append(f"{i}. {feature}")
        
        # 儲存報告
        with open(WORK_DIR / 'feature_engineering_final_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\n最終報告已儲存至 feature_engineering_final_report.txt")
    
    def run_pipeline_integration(self):
        """執行特徵工程管道整合"""
        print("=== 開始特徵工程管道整合 ===")
        
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
        importance_summary = self.generate_feature_importance_summary()
        print(f"\n特徵重要性摘要已儲存，前5個最重要的特徵：")
        print(importance_summary.head())
        
        # 儲存管道配置
        self.save_pipeline_config()
        
        # 生成最終報告
        self.generate_final_report()
        
        print("\n=== 特徵工程管道整合完成 ===")
        print(f"最終特徵數量：{len(self.final_features)}")
        print("輸出檔案：")
        print("- train_final.csv")
        print("- test_final.csv")
        print("- final_feature_importance_summary.csv")
        print("- feature_engineering_config.json")
        print("- feature_engineering_final_report.txt")

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run_pipeline_integration()
