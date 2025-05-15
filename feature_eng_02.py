"""
feature_eng_02.py - Embedded階段實施（多模型特徵選擇）
功能：使用Lasso、隨機森林和Elastic Net進行特徵選擇
作者：Tommy
日期：2025-05-15
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
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

class EmbeddedFeatureSelector:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.filter_features = []
        self.embedded_scores = {}
        self.selected_features = []
        
    def load_data(self):
        """載入數據和Filter階段選擇的特徵"""
        try:
            # 載入清理後的數據
            train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
            # 載入Filter階段選擇的特徵
            filter_selected = pd.read_csv(WORK_DIR / 'filter_selected_features.csv')
            self.filter_features = filter_selected['feature'].tolist()
            
            # 分離特徵和目標變數
            target_col = train_df.columns[-1]
            
            # 只使用Filter階段選擇的特徵
            self.X_train = train_df[self.filter_features]
            self.y_train = train_df[target_col]
            self.X_test = test_df[self.filter_features]
            self.y_test = test_df[target_col]
            
            # 編碼目標變數
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            
            print(f"成功載入數據")
            print(f"Filter階段選擇的特徵數量：{len(self.filter_features)}")
            print(f"訓練集：{len(self.X_train)} 筆")
            print(f"測試集：{len(self.X_test)} 筆")
            
            return True
        except Exception as e:
            print(f"載入數據時發生錯誤：{e}")
            return False
    
    def lasso_selection(self):
        """使用Lasso回歸進行特徵選擇"""
        print("\n=== Lasso特徵選擇 ===")
        
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        # 對多類別問題使用One-vs-Rest策略
        lasso_scores = np.zeros(len(self.filter_features))
        
        # 為每個類別訓練一個Lasso模型
        for class_idx in range(len(self.label_encoder.classes_)):
            # 創建二元分類目標
            y_binary = (self.y_train_encoded == class_idx).astype(int)
            
            # 使用交叉驗證選擇最佳的alpha
            lasso_cv = LassoCV(cv=5, random_state=42, n_alphas=100)
            lasso_cv.fit(X_train_scaled, y_binary)
            
            # 累加每個特徵的重要性（係數絕對值）
            lasso_scores += np.abs(lasso_cv.coef_)
            
            print(f"類別 {self.label_encoder.classes_[class_idx]} - "
                  f"最佳alpha: {lasso_cv.alpha_:.4f}, "
                  f"非零係數: {np.sum(lasso_cv.coef_ != 0)}")
        
        # 平均化分數
        lasso_scores /= len(self.label_encoder.classes_)
        
        # 儲存結果
        self.embedded_scores['lasso'] = dict(zip(self.filter_features, lasso_scores))
        
        # 顯示前15個最重要的特徵
        sorted_features = sorted(self.embedded_scores['lasso'].items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        print("\nLasso最重要的前15個特徵：")
        for feature, score in sorted_features:
            print(f"{feature}: {score:.4f}")
    
    def random_forest_selection(self):
        """使用隨機森林進行特徵選擇"""
        print("\n=== 隨機森林特徵選擇 ===")
        
        # 訓練隨機森林
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train_encoded)
        
        # 獲取特徵重要性
        importances = rf.feature_importances_
        
        # 儲存結果
        self.embedded_scores['random_forest'] = dict(zip(self.filter_features, importances))
        
        # 顯示前15個最重要的特徵
        sorted_features = sorted(self.embedded_scores['random_forest'].items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        print("\n隨機森林最重要的前15個特徵：")
        for feature, score in sorted_features:
            print(f"{feature}: {score:.4f}")
    
    def elastic_net_selection(self):
        """使用Elastic Net進行特徵選擇"""
        print("\n=== Elastic Net特徵選擇 ===")
        
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        # 對多類別問題使用One-vs-Rest策略
        elastic_scores = np.zeros(len(self.filter_features))
        
        # 為每個類別訓練一個Elastic Net模型
        for class_idx in range(len(self.label_encoder.classes_)):
            # 創建二元分類目標
            y_binary = (self.y_train_encoded == class_idx).astype(int)
            
            # 使用交叉驗證選擇最佳的alpha和l1_ratio
            elastic_cv = ElasticNetCV(
                cv=5, 
                random_state=42, 
                n_alphas=50,
                l1_ratio=[.1, .5, .7, .9, .95, .99]
            )
            elastic_cv.fit(X_train_scaled, y_binary)
            
            # 累加每個特徵的重要性（係數絕對值）
            elastic_scores += np.abs(elastic_cv.coef_)
            
            print(f"類別 {self.label_encoder.classes_[class_idx]} - "
                  f"最佳alpha: {elastic_cv.alpha_:.4f}, "
                  f"最佳l1_ratio: {elastic_cv.l1_ratio_:.2f}, "
                  f"非零係數: {np.sum(elastic_cv.coef_ != 0)}")
        
        # 平均化分數
        elastic_scores /= len(self.label_encoder.classes_)
        
        # 儲存結果
        self.embedded_scores['elastic_net'] = dict(zip(self.filter_features, elastic_scores))
        
        # 顯示前15個最重要的特徵
        sorted_features = sorted(self.embedded_scores['elastic_net'].items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        print("\nElastic Net最重要的前15個特徵：")
        for feature, score in sorted_features:
            print(f"{feature}: {score:.4f}")
    
    def combine_scores(self):
        """綜合三種方法的分數選擇特徵"""
        print("\n=== 綜合特徵選擇 ===")
        
        # 標準化各方法的分數到0-1範圍
        normalized_scores = {}
        
        for method in ['lasso', 'random_forest', 'elastic_net']:
            scores = list(self.embedded_scores[method].values())
            min_score = min(scores)
            max_score = max(scores)
            
            normalized_scores[method] = {}
            for feature, score in self.embedded_scores[method].items():
                if max_score > min_score:
                    normalized_scores[method][feature] = (score - min_score) / (max_score - min_score)
                else:
                    normalized_scores[method][feature] = 0
        
        # 計算綜合分數（加權平均）
        weights = {
            'lasso': 0.3,
            'random_forest': 0.4,
            'elastic_net': 0.3
        }
        
        combined_scores = {}
        for feature in self.filter_features:
            combined_scores[feature] = sum(
                normalized_scores[method][feature] * weights[method]
                for method in weights.keys()
            )
        
        # 選擇特徵
        # 方法1：選擇綜合分數排名前N個或累積重要性達到80%
        sorted_features = sorted(combined_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        total_score = sum(combined_scores.values())
        cumulative_score = 0
        
        self.selected_features = []
        for feature, score in sorted_features:
            self.selected_features.append(feature)
            cumulative_score += score
            
            # 至少選擇20個特徵，或累積分數達到80%
            if len(self.selected_features) >= 20 and cumulative_score >= 0.8 * total_score:
                break
        
        # 方法2：非零係數共識
        # 檢查在Lasso和Elastic Net中都有非零係數的特徵
        lasso_nonzero = {f for f, s in self.embedded_scores['lasso'].items() if s > 0}
        elastic_nonzero = {f for f, s in self.embedded_scores['elastic_net'].items() if s > 0}
        consensus_features = lasso_nonzero.intersection(elastic_nonzero)
        
        # 合併兩種方法的結果
        self.selected_features = list(set(self.selected_features) | consensus_features)
        
        print(f"綜合分數選擇的特徵：{len(sorted_features[:len(self.selected_features)])} 個")
        print(f"非零係數共識特徵：{len(consensus_features)} 個")
        print(f"最終選擇的特徵：{len(self.selected_features)} 個")
        
        # 儲存綜合分數
        self.embedded_scores['combined'] = combined_scores
    
    def visualize_results(self):
        """視覺化結果"""
        viz_dir = WORK_DIR / 'embedded_selection_results'
        viz_dir.mkdir(exist_ok=True)
        
        # 各方法的特徵重要性對比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = ['lasso', 'random_forest', 'elastic_net', 'combined']
        titles = ['Lasso係數', '隨機森林重要性', 'Elastic Net係數', '綜合分數']
        
        for idx, (method, title, ax) in enumerate(zip(methods, titles, axes.flat)):
            # 選擇前20個特徵顯示
            sorted_features = sorted(self.embedded_scores[method].items(), 
                                   key=lambda x: x[1], reverse=True)[:20]
            features, scores = zip(*sorted_features)
            
            ax.barh(features, scores)
            ax.set_xlabel('分數')
            ax.set_title(title)
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'embedded_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 方法之間的分數相關性熱力圖
        plt.figure(figsize=(10, 8))
        
        # 創建分數矩陣
        score_matrix = []
        for feature in self.filter_features:
            scores = [
                self.embedded_scores['lasso'][feature],
                self.embedded_scores['random_forest'][feature],
                self.embedded_scores['elastic_net'][feature]
            ]
            score_matrix.append(scores)
        
        score_df = pd.DataFrame(
            score_matrix,
            columns=['Lasso', 'Random Forest', 'Elastic Net'],
            index=self.filter_features
        )
        
        # 計算相關係數
        corr_matrix = score_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title('不同方法特徵分數的相關性')
        plt.savefig(viz_dir / 'method_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """儲存結果"""
        # 儲存選擇的特徵
        selected_features_df = pd.DataFrame({
            'feature': self.selected_features,
            'selected': 1
        })
        selected_features_df.to_csv(WORK_DIR / 'embedded_selected_features.csv', index=False)
        
        # 儲存所有特徵的分數
        all_scores = []
        for feature in self.filter_features:
            score_dict = {
                'feature': feature,
                'lasso_score': self.embedded_scores['lasso'][feature],
                'rf_score': self.embedded_scores['random_forest'][feature],
                'elastic_score': self.embedded_scores['elastic_net'][feature],
                'combined_score': self.embedded_scores['combined'][feature],
                'selected': feature in self.selected_features
            }
            all_scores.append(score_dict)
        
        scores_df = pd.DataFrame(all_scores)
        scores_df = scores_df.sort_values('combined_score', ascending=False)
        scores_df.to_csv(WORK_DIR / 'embedded_feature_scores.csv', index=False)
        
        # 生成報告
        report = []
        report.append("=== Embedded階段特徵選擇報告 ===\n")
        report.append(f"初始特徵數量（Filter階段）：{len(self.filter_features)}")
        report.append(f"選擇的特徵數量：{len(self.selected_features)}")
        report.append(f"特徵選擇比例：{len(self.selected_features)/len(self.filter_features)*100:.1f}%")
        
        report.append("\n=== 各方法選擇摘要 ===")
        lasso_nonzero = sum(1 for s in self.embedded_scores['lasso'].values() if s > 0)
        elastic_nonzero = sum(1 for s in self.embedded_scores['elastic_net'].values() if s > 0)
        rf_top20 = len([s for s in self.embedded_scores['random_forest'].values() if s > np.percentile(list(self.embedded_scores['random_forest'].values()), 80)])
        
        report.append(f"Lasso非零係數特徵：{lasso_nonzero} 個")
        report.append(f"Elastic Net非零係數特徵：{elastic_nonzero} 個")
        report.append(f"隨機森林前20%重要特徵：{rf_top20} 個")
        
        report.append("\n=== 選擇的特徵（按綜合分數排序）===")
        sorted_selected = sorted(
            [(f, self.embedded_scores['combined'][f]) for f in self.selected_features],
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, score) in enumerate(sorted_selected, 1):
            report.append(f"{i}. {feature}: {score:.4f}")
        
        with open(WORK_DIR / 'embedded_selection_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\n結果已儲存：")
        print("- embedded_selected_features.csv")
        print("- embedded_feature_scores.csv")
        print("- embedded_selection_report.txt")
    
    def run_embedded_selection(self):
        """執行完整的Embedded階段特徵選擇"""
        print("=== 開始Embedded階段特徵選擇 ===")
        
        # 載入數據
        if not self.load_data():
            return
        
        # 執行三種特徵選擇方法
        self.lasso_selection()
        self.random_forest_selection()
        self.elastic_net_selection()
        
        # 綜合分數並選擇特徵
        self.combine_scores()
        
        # 視覺化結果
        self.visualize_results()
        
        # 儲存結果
        self.save_results()
        
        print("\n=== Embedded階段特徵選擇完成 ===")

if __name__ == "__main__":
    selector = EmbeddedFeatureSelector()
    selector.run_embedded_selection()
