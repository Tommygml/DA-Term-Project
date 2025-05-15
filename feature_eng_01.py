"""
feature_eng_01.py - Filter階段實施（統計檢定）
功能：使用統計方法進行特徵篩選
作者：Tommy
日期：2025-05-15
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
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

class FilterFeatureSelector:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_scores = {}
        self.selected_features = []
        
    def load_data(self):
        """載入清理後的數據"""
        try:
            self.train_df = pd.read_csv(WORK_DIR / 'train_cleaned.csv')
            self.test_df = pd.read_csv(WORK_DIR / 'test_cleaned.csv')
            
            # 分離特徵和目標變數
            target_col = self.train_df.columns[-1]
            self.X_train = self.train_df.drop(columns=[target_col])
            self.y_train = self.train_df[target_col]
            self.X_test = self.test_df.drop(columns=[target_col])
            self.y_test = self.test_df[target_col]
            
            # 編碼目標變數
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            
            print(f"成功載入數據")
            print(f"訓練集：{len(self.X_train)} 筆，{len(self.X_train.columns)} 個特徵")
            print(f"測試集：{len(self.X_test)} 筆，{len(self.X_test.columns)} 個特徵")
            print(f"目標變數類別：{self.label_encoder.classes_}")
            
            return True
        except Exception as e:
            print(f"載入數據時發生錯誤：{e}")
            return False
    
    def identify_feature_types(self):
        """識別特徵類型"""
        self.numeric_features = []
        self.categorical_features = []
        
        for col in self.X_train.columns:
            if self.X_train[col].dtype in ['int64', 'float64']:
                # 檢查是否為二元或多類別特徵
                unique_values = self.X_train[col].nunique()
                if unique_values <= 10:
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"\n數值型特徵：{len(self.numeric_features)} 個")
        print(f"類別型特徵：{len(self.categorical_features)} 個")
    
    def chi_square_test(self):
        """卡方檢定（用於類別特徵）"""
        print("\n=== 執行卡方檢定 ===")
        chi2_scores = {}
        
        for feature in self.categorical_features:
            # 確保特徵為非負整數
            X_feature = self.X_train[[feature]].copy()
            if X_feature[feature].min() < 0:
                X_feature[feature] = X_feature[feature] - X_feature[feature].min()
            
            try:
                chi2_stat, p_value = chi2(X_feature, self.y_train_encoded)
                chi2_scores[feature] = {
                    'chi2_statistic': chi2_stat[0],
                    'p_value': p_value[0]
                }
            except Exception as e:
                print(f"特徵 {feature} 的卡方檢定失敗：{e}")
                chi2_scores[feature] = {
                    'chi2_statistic': 0,
                    'p_value': 1
                }
        
        # 儲存結果
        self.feature_scores['chi2'] = chi2_scores
        
        # 顯示前10個最顯著的特徵
        sorted_features = sorted(chi2_scores.items(), 
                               key=lambda x: x[1]['p_value'])[:10]
        print("\n卡方檢定最顯著的前10個特徵：")
        for feature, scores in sorted_features:
            print(f"{feature}: χ²={scores['chi2_statistic']:.4f}, "
                  f"p-value={scores['p_value']:.4e}")
    
    def anova_test(self):
        """ANOVA檢定（用於數值特徵）"""
        print("\n=== 執行ANOVA檢定 ===")
        anova_scores = {}
        
        for feature in self.numeric_features:
            X_feature = self.X_train[[feature]]
            try:
                f_stat, p_value = f_classif(X_feature, self.y_train_encoded)
                anova_scores[feature] = {
                    'f_statistic': f_stat[0],
                    'p_value': p_value[0]
                }
            except Exception as e:
                print(f"特徵 {feature} 的ANOVA檢定失敗：{e}")
                anova_scores[feature] = {
                    'f_statistic': 0,
                    'p_value': 1
                }
        
        # 儲存結果
        self.feature_scores['anova'] = anova_scores
        
        # 顯示前10個最顯著的特徵
        sorted_features = sorted(anova_scores.items(), 
                               key=lambda x: x[1]['p_value'])[:10]
        print("\n ANOVA檢定最顯著的前10個特徵：")
        for feature, scores in sorted_features:
            print(f"{feature}: F={scores['f_statistic']:.4f}, "
                  f"p-value={scores['p_value']:.4e}")
    
    def correlation_analysis(self):
        """相關性分析（用於識別高度相關的特徵）"""
        print("\n=== 執行相關性分析 ===")
        
        # 計算數值特徵的相關係數矩陣
        numeric_df = self.X_train[self.numeric_features]
        corr_matrix = numeric_df.corr().abs()
        
        # 找出高度相關的特徵對
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.7:  # 閾值設為0.7
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        self.feature_scores['correlation'] = high_corr_pairs
        
        print(f"發現 {len(high_corr_pairs)} 對高度相關的特徵（相關係數 > 0.7）：")
        for feat1, feat2, corr in high_corr_pairs[:5]:  # 只顯示前5對
            print(f"{feat1} - {feat2}: {corr:.3f}")
    
    def mutual_information(self):
        """互信息計算（適用於所有特徵）"""
        print("\n=== 計算互信息 ===")
        mi_scores = {}
        
        # 對所有特徵計算互信息
        all_features = self.X_train.columns.tolist()
        mi_values = mutual_info_classif(self.X_train, self.y_train_encoded,
                                       discrete_features='auto', 
                                       random_state=42)
        
        for feature, mi_score in zip(all_features, mi_values):
            mi_scores[feature] = mi_score
        
        # 儲存結果
        self.feature_scores['mutual_info'] = mi_scores
        
        # 顯示前15個最高互信息的特徵
        sorted_features = sorted(mi_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        print("\n互信息最高的前15個特徵：")
        for feature, mi_score in sorted_features:
            print(f"{feature}: {mi_score:.4f}")
    
    def select_features(self, p_value_threshold=0.05, mi_percentile=75):
        """根據統計檢定結果選擇特徵"""
        print("\n=== 特徵選擇 ===")
        
        selected_by_chi2 = []
        selected_by_anova = []
        selected_by_mi = []
        
        # 根據卡方檢定選擇特徵
        if 'chi2' in self.feature_scores:
            for feature, scores in self.feature_scores['chi2'].items():
                if scores['p_value'] < p_value_threshold:
                    selected_by_chi2.append(feature)
        
        # 根據ANOVA選擇特徵
        if 'anova' in self.feature_scores:
            for feature, scores in self.feature_scores['anova'].items():
                if scores['p_value'] < p_value_threshold:
                    selected_by_anova.append(feature)
        
        # 根據互信息選擇特徵（前75%）
        if 'mutual_info' in self.feature_scores:
            mi_threshold = np.percentile(
                list(self.feature_scores['mutual_info'].values()), 
                mi_percentile
            )
            for feature, mi_score in self.feature_scores['mutual_info'].items():
                if mi_score >= mi_threshold:
                    selected_by_mi.append(feature)
        
        # 合併所有選擇的特徵
        self.selected_features = list(set(
            selected_by_chi2 + selected_by_anova + selected_by_mi
        ))
        
        print(f"卡方檢定選擇的特徵：{len(selected_by_chi2)} 個")
        print(f"ANOVA選擇的特徵：{len(selected_by_anova)} 個")
        print(f"互信息選擇的特徵：{len(selected_by_mi)} 個")
        print(f"總共選擇的特徵：{len(self.selected_features)} 個")
        
        # 處理高度相關的特徵
        if 'correlation' in self.feature_scores:
            features_to_remove = set()
            for feat1, feat2, corr in self.feature_scores['correlation']:
                # 如果兩個特徵都被選中，移除互信息較低的那個
                if feat1 in self.selected_features and feat2 in self.selected_features:
                    mi_1 = self.feature_scores['mutual_info'].get(feat1, 0)
                    mi_2 = self.feature_scores['mutual_info'].get(feat2, 0)
                    if mi_1 < mi_2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
            
            self.selected_features = [f for f in self.selected_features 
                                     if f not in features_to_remove]
            print(f"移除高度相關特徵後：{len(self.selected_features)} 個")
    
    def visualize_results(self):
        """視覺化篩選結果"""
        viz_dir = WORK_DIR / 'feature_selection_results'
        viz_dir.mkdir(exist_ok=True)
        
        # 互信息分數視覺化
        if 'mutual_info' in self.feature_scores:
            plt.figure(figsize=(12, 8))
            mi_scores = self.feature_scores['mutual_info']
            sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:20]
            features, scores = zip(*sorted_mi)
            
            plt.barh(features, scores)
            plt.xlabel('互信息分數')
            plt.title('前20個特徵的互信息分數')
            plt.tight_layout()
            plt.savefig(viz_dir / 'mutual_information_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # p值分布視覺化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 卡方檢定p值
        if 'chi2' in self.feature_scores:
            p_values = [scores['p_value'] for scores in self.feature_scores['chi2'].values()]
            ax1.hist(p_values, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
            ax1.set_xlabel('p-value')
            ax1.set_ylabel('頻率')
            ax1.set_title('卡方檢定 p值分布')
            ax1.legend()
        
        # ANOVA p值
        if 'anova' in self.feature_scores:
            p_values = [scores['p_value'] for scores in self.feature_scores['anova'].values()]
            ax2.hist(p_values, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
            ax2.set_xlabel('p-value')
            ax2.set_ylabel('頻率')
            ax2.set_title('ANOVA檢定 p值分布')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'p_value_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """儲存篩選結果"""
        # 儲存選擇的特徵
        selected_features_df = pd.DataFrame({
            'feature': self.selected_features,
            'selected': 1
        })
        selected_features_df.to_csv(WORK_DIR / 'filter_selected_features.csv', index=False)
        
        # 儲存詳細的分數資訊
        all_scores = []
        
        # 整理所有特徵的分數
        all_features = self.X_train.columns.tolist()
        for feature in all_features:
            score_dict = {'feature': feature}
            
            # 卡方檢定分數
            if feature in self.feature_scores.get('chi2', {}):
                score_dict['chi2_statistic'] = self.feature_scores['chi2'][feature]['chi2_statistic']
                score_dict['chi2_p_value'] = self.feature_scores['chi2'][feature]['p_value']
            
            # ANOVA分數
            if feature in self.feature_scores.get('anova', {}):
                score_dict['anova_f_statistic'] = self.feature_scores['anova'][feature]['f_statistic']
                score_dict['anova_p_value'] = self.feature_scores['anova'][feature]['p_value']
            
            # 互信息分數
            if feature in self.feature_scores.get('mutual_info', {}):
                score_dict['mutual_info'] = self.feature_scores['mutual_info'][feature]
            
            # 是否被選中
            score_dict['selected'] = feature in self.selected_features
            
            all_scores.append(score_dict)
        
        scores_df = pd.DataFrame(all_scores)
        scores_df.to_csv(WORK_DIR / 'filter_feature_scores.csv', index=False)
        
        # 生成報告
        report = []
        report.append("=== Filter階段特徵選擇報告 ===\n")
        report.append(f"原始特徵數量：{len(all_features)}")
        report.append(f"選擇的特徵數量：{len(self.selected_features)}")
        report.append(f"特徵選擇比例：{len(self.selected_features)/len(all_features)*100:.1f}%")
        
        report.append("\n=== 統計檢定結果摘要 ===")
        report.append(f"卡方檢定顯著特徵：{len([f for f, s in self.feature_scores.get('chi2', {}).items() if s['p_value'] < 0.05])} 個")
        report.append(f"ANOVA檢定顯著特徵：{len([f for f, s in self.feature_scores.get('anova', {}).items() if s['p_value'] < 0.05])} 個")
        report.append(f"高度相關特徵對：{len(self.feature_scores.get('correlation', []))} 對")
        
        report.append("\n=== 選擇的特徵列表 ===")
        for i, feature in enumerate(self.selected_features, 1):
            report.append(f"{i}. {feature}")
        
        with open(WORK_DIR / 'filter_selection_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\n結果已儲存：")
        print("- filter_selected_features.csv")
        print("- filter_feature_scores.csv")
        print("- filter_selection_report.txt")
    
    def run_filter_selection(self):
        """執行完整的Filter階段特徵選擇"""
        print("=== 開始Filter階段特徵選擇 ===")
        
        # 載入數據
        if not self.load_data():
            return
        
        # 識別特徵類型
        self.identify_feature_types()
        
        # 執行統計檢定
        self.chi_square_test()
        self.anova_test()
        self.correlation_analysis()
        self.mutual_information()
        
        # 選擇特徵
        self.select_features()
        
        # 視覺化結果
        self.visualize_results()
        
        # 儲存結果
        self.save_results()
        
        print("\n=== Filter階段特徵選擇完成 ===")

if __name__ == "__main__":
    selector = FilterFeatureSelector()
    selector.run_filter_selection()
