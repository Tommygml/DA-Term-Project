"""
feature_eng_03.py - Wrapper階段實施（RFE實施與最優特徵集確定）
功能：使用遞迴特徵消除（RFE）進行特徵選擇，找出最優特徵子集
作者：Tommy
日期：2025-05-15
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

class WrapperFeatureSelector:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.embedded_features = []
        self.rfe_selected_features = []
        self.rfe_results = {}
        
    def load_data(self):
        """載入數據和Embedded階段選擇的特徵"""
        try:
            # 載入清理後的數據
            train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
            # 載入Embedded階段選擇的特徵
            embedded_selected = pd.read_csv(WORK_DIR / 'embedded_selected_features.csv')
            self.embedded_features = embedded_selected['feature'].tolist()
            
            # 分離特徵和目標變數
            target_col = train_df.columns[-1]
            
            # 只使用Embedded階段選擇的特徵
            self.X_train = train_df[self.embedded_features]
            self.y_train = train_df[target_col]
            self.X_test = test_df[self.embedded_features]
            self.y_test = test_df[target_col]
            
            # 編碼目標變數
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            
            print(f"成功載入數據")
            print(f"Embedded階段選擇的特徵數量：{len(self.embedded_features)}")
            print(f"訓練集：{len(self.X_train)} 筆")
            print(f"測試集：{len(self.X_test)} 筆")
            print(f"目標類別：{self.label_encoder.classes_}")
            
            return True
        except Exception as e:
            print(f"載入數據時發生錯誤：{e}")
            return False
    
    def create_custom_scorer(self):
        """創建自定義評分函數"""
        # 使用macro F1分數作為評分標準
        return make_scorer(f1_score, average='macro')
    
    def perform_rfe(self):
        """使用遞歸特徵消除演算法進行特徵選擇"""
        print("\n=== 執行遞歸特徵消除 ===")
        
        # 使用隨機森林作為基礎模型
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 使用交叉驗證的遞歸特徵消除
        self.rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='f1_weighted',
            min_features_to_select=5,
            n_jobs=-1
        )
        
        # 訓練RFE模型
        self.rfecv.fit(self.X_train, self.y_train_encoded)
        
        # 獲取選擇的特徵
        self.rfe_selected_features = [
            feature for feature, selected in zip(self.embedded_features, self.rfecv.support_)
            if selected
        ]
        
        print(f"選擇的特徵數量：{len(self.rfe_selected_features)}")
        
        # 查找sklearn版本並使用正確的屬性
        try:
            # 嘗試新版本的cv_results_屬性
            if hasattr(self.rfecv, 'cv_results_'):
                scores = self.rfecv.cv_results_['mean_test_score']
                best_score = scores[self.rfecv.n_features_ - 1]
                print(f"最佳F1分數：{best_score:.4f}")
            # 嘗試舊版本的grid_scores_屬性
            elif hasattr(self.rfecv, 'grid_scores_'):
                scores = self.rfecv.grid_scores_
                print(f"最佳F1分數：{scores[self.rfecv.n_features_-1]:.4f}")
            else:
                print("無法獲取特徵選擇的分數信息。")
                scores = [0] * len(self.embedded_features)  # 創建空分數列表
        except Exception as e:
            print(f"獲取特徵選擇分數時出錯：{e}")
            print("繼續執行特徵選擇流程。")
            scores = [0] * len(self.embedded_features)  # 創建空分數列表
        
        # 確保n_features和scores長度一致
        n_features = list(range(1, len(scores) + 1))
        
        # 儲存詳細結果
        self.rfe_results = {
            'n_features': n_features,
            'scores': scores,
            'rankings': self.rfecv.ranking_,
            'support': self.rfecv.support_
        }
    
    def analyze_performance_curve(self):
        """分析性能曲線，找出性能與特徵數的平衡點"""
        print("\n=== 分析性能曲線 ===")
        
        scores = self.rfe_results['scores']
        n_features = self.rfe_results['n_features']
        
        # 確保數據長度匹配
        min_len = min(len(n_features), len(scores))
        if min_len < len(n_features) or min_len < len(scores):
            print(f"警告：特徵數量和分數長度不一致，截斷至共同長度 {min_len}")
            n_features = n_features[:min_len]
            scores = scores[:min_len]
        
        # 找出性能變化的拐點
        # 計算性能改善率
        improvements = []
        for i in range(1, len(scores)):
            improvement = (scores[i] - scores[i-1]) / max(scores[i-1], 0.0001) * 100
            improvements.append(improvement)
        
        # 找出改善率小於1%的點作為可能的停止點
        stopping_points = []
        for i, imp in enumerate(improvements):
            if imp < 1.0 and i > min(10, len(improvements) // 3):  # 至少保留一定數量的特徵
                stopping_points.append(i + 1)
        
        optimal_n = None
        if stopping_points:
            optimal_n = stopping_points[0]
            print(f"根據性能改善率分析，建議使用 {optimal_n} 個特徵")
            print(f"此時F1分數為：{scores[optimal_n-1]:.4f}")
            
            if optimal_n < len(scores) and self.rfecv.n_features_ < len(scores):
                print(f"相比最佳分數（{scores[self.rfecv.n_features_-1]:.4f}）"
                      f"僅下降 {(scores[self.rfecv.n_features_-1] - scores[optimal_n-1])*100:.2f}%")
        else:
            print("未找到明顯的性能拐點，建議使用RFECV自動選擇的特徵數量")
            optimal_n = self.rfecv.n_features_
        
        # 儲存平衡點分析結果
        self.rfe_results['balance_analysis'] = {
            'optimal_n': optimal_n if optimal_n is not None else self.rfecv.n_features_,
            'improvements': improvements,
            'stopping_points': stopping_points
        }
    
    def visualize_results(self):
        """視覺化RFE結果"""
        viz_dir = WORK_DIR / 'wrapper_selection_results'
        viz_dir.mkdir(exist_ok=True)
        
        # 性能曲線圖
        plt.figure(figsize=(12, 6))
        
        n_features = self.rfe_results['n_features']
        scores = self.rfe_results['scores']
        
        # 確保數據長度匹配
        min_len = min(len(n_features), len(scores))
        if min_len < len(n_features) or min_len < len(scores):
            print(f"警告：特徵數量和分數長度不一致，截斷至共同長度 {min_len}")
            n_features = n_features[:min_len]
            scores = scores[:min_len]
        
        plt.plot(n_features, scores, 'b-', linewidth=2, marker='o', markersize=5)
        plt.axvline(x=self.rfecv.n_features_, color='r', linestyle='--', 
                   label=f'最優特徵數: {self.rfecv.n_features_}')
        
        # 如果有平衡點分析結果，也標記出來
        if 'balance_analysis' in self.rfe_results:
            optimal_n = self.rfe_results['balance_analysis']['optimal_n']
            if optimal_n != self.rfecv.n_features_:
                plt.axvline(x=optimal_n, color='g', linestyle='--', 
                           label=f'建議特徵數: {optimal_n}')
        
        plt.xlabel('特徵數量')
        plt.ylabel('CV F1分數 (Macro)')
        plt.title('RFE交叉驗證性能曲線')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'rfe_performance_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特徵排名視覺化
        plt.figure(figsize=(10, 12))
        
        # 創建排名數據框
        ranking_df = pd.DataFrame({
            'feature': self.embedded_features,
            'ranking': self.rfe_results['rankings'],
            'selected': self.rfe_results['support']
        })
        ranking_df = ranking_df.sort_values('ranking')
        
        # 顏色映射
        colors = ['green' if selected else 'red' 
                 for selected in ranking_df['selected']]
        
        plt.barh(ranking_df['feature'], ranking_df['ranking'], color=colors)
        plt.xlabel('RFE排名（數字越小越重要）')
        plt.title('特徵RFE排名')
        plt.gca().invert_yaxis()
        
        # 添加圖例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='green', label='選中的特徵'),
            Patch(color='red', label='未選中的特徵')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'rfe_feature_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 改善率曲線
        if 'balance_analysis' in self.rfe_results:
            plt.figure(figsize=(10, 6))
            
            improvements = self.rfe_results['balance_analysis']['improvements']
            plt.plot(n_features[1:], improvements, 'b-', linewidth=2)
            plt.axhline(y=1, color='r', linestyle='--', label='1%改善率閾值')
            
            plt.xlabel('特徵數量')
            plt.ylabel('性能改善率 (%)')
            plt.title('相鄰特徵數的性能改善率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'performance_improvement_rate.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """儲存RFE結果"""
        # 儲存選擇的特徵
        selected_features_df = pd.DataFrame({
            'feature': self.rfe_selected_features,
            'selected': 1
        })
        selected_features_df.to_csv(WORK_DIR / 'wrapper_selected_features.csv', index=False)
        
        # 儲存所有特徵的RFE結果
        rfe_details = []
        for i, feature in enumerate(self.embedded_features):
            detail = {
                'feature': feature,
                'ranking': self.rfe_results['rankings'][i],
                'selected': self.rfe_results['support'][i],
                'selected_at_n': self.rfecv.n_features_ if self.rfe_results['support'][i] else 0
            }
            rfe_details.append(detail)
        
        rfe_df = pd.DataFrame(rfe_details)
        rfe_df = rfe_df.sort_values('ranking')
        rfe_df.to_csv(WORK_DIR / 'wrapper_rfe_details.csv', index=False)
        
        # 儲存性能曲線數據
        perf_curve_df = pd.DataFrame({
            'n_features': self.rfe_results['n_features'],
            'cv_score': self.rfe_results['scores']
        })
        perf_curve_df.to_csv(WORK_DIR / 'wrapper_performance_curve.csv', index=False)
        
        # 獲取分數
        scores = self.rfe_results['scores']
        if len(scores) <= self.rfecv.n_features_:
            print("警告：分數陣列長度不足，無法獲取最佳分數和單特徵分數")
            best_score = scores[-1] if len(scores) > 0 else 0
            single_feature_score = scores[0] if len(scores) > 0 else 0
        else:
            best_score = scores[self.rfecv.n_features_-1]
            single_feature_score = scores[0]
        
        # 生成報告
        report = []
        report.append("=== Wrapper階段特徵選擇報告 ===\n")
        report.append(f"初始特徵數量（Embedded階段）：{len(self.embedded_features)}")
        report.append(f"RFE最優特徵數量：{self.rfecv.n_features_}")
        report.append(f"最終選擇的特徵數量：{len(self.rfe_selected_features)}")
        report.append(f"特徵選擇比例：{len(self.rfe_selected_features)/len(self.embedded_features)*100:.1f}%")
        
        report.append("\n=== 性能指標 ===")
        report.append(f"最佳CV F1分數：{best_score:.4f}")
        report.append(f"單特徵F1分數：{single_feature_score:.4f}")
        
        if best_score > 0 and single_feature_score > 0:
            report.append(f"性能提升：{(best_score - single_feature_score)*100:.2f}%")
        
        if 'balance_analysis' in self.rfe_results:
            optimal_n = self.rfe_results['balance_analysis']['optimal_n']
            if optimal_n != self.rfecv.n_features_ and optimal_n < len(scores):
                report.append(f"\n=== 平衡點分析 ===")
                report.append(f"建議使用特徵數：{optimal_n}")
                report.append(f"建議點F1分數：{scores[optimal_n-1]:.4f}")
                report.append(f"相比最優僅降低：{(best_score - scores[optimal_n-1])*100:.2f}%")
        
        report.append("\n=== 選擇的特徵（按排名）===")
        selected_with_ranking = [
            (f, self.rfe_results['rankings'][self.embedded_features.index(f)])
            for f in self.rfe_selected_features
        ]
        selected_with_ranking.sort(key=lambda x: x[1])
        
        for i, (feature, ranking) in enumerate(selected_with_ranking, 1):
            report.append(f"{i}. {feature} (排名: {ranking})")
        
        with open(WORK_DIR / 'wrapper_selection_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\n結果已儲存：")
        print("- wrapper_selected_features.csv")
        print("- wrapper_rfe_details.csv")
        print("- wrapper_performance_curve.csv")
        print("- wrapper_selection_report.txt")
    
    def run_wrapper_selection(self):
        """執行完整的Wrapper階段特徵選擇"""
        print("=== 開始Wrapper階段特徵選擇 ===")
        
        # 載入數據
        if not self.load_data():
            return
        
        # 執行RFE
        self.perform_rfe()
        
        # 分析性能曲線
        self.analyze_performance_curve()
        
        # 視覺化結果
        self.visualize_results()
        
        # 儲存結果
        self.save_results()
        
        print("\n=== Wrapper階段特徵選擇完成 ===")

if __name__ == "__main__":
    selector = WrapperFeatureSelector()
    selector.run_wrapper_selection()