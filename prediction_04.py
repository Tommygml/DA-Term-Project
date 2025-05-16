"""
prediction_04.py - 隨機森林模型實作
功能：使用隨機森林進行學生輟學預測
作者：Tommy
日期：2025-05-17
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.model_name = "RandomForest"
        self.classes = None
        self.feature_names = None
        self.is_multiclass = False
        
    def load_data(self):
        """載入最終的訓練集和測試集"""
        try:
            # 優先嘗試載入修復後的數據集
            try:
                train_df = pd.read_csv(WORK_DIR / 'train_final_fixed.csv')
                test_df = pd.read_csv(WORK_DIR / 'test_final_fixed.csv')
                print("已載入修復後的數據集")
            except:
                # 如果找不到修復後的數據集，則嘗試載入原始最終數據集
                train_df = pd.read_csv(WORK_DIR / 'train_final.csv')
                test_df = pd.read_csv(WORK_DIR / 'test_final.csv')
                print("已載入原始最終數據集")
            
            # 檢查是否存在名為"Target"的列，如果沒有，則假設最後一列是目標變數
            if 'Target' in train_df.columns:
                target_col = 'Target'
            else:
                target_col = train_df.columns[-1]
                print(f"找不到'Target'列，使用最後一列 '{target_col}' 作為目標變數")
            
            # 分離特徵和目標變數
            self.X_train = train_df.drop(columns=[target_col])
            self.y_train = train_df[target_col]
            self.X_test = test_df.drop(columns=[target_col])
            self.y_test = test_df[target_col]
            
            # 保存特徵名稱
            self.feature_names = self.X_train.columns.tolist()
            
            # 檢查目標變數類型
            unique_values = np.unique(self.y_train)
            print(f"目標變數的唯一值數量：{len(unique_values)}")
            
            # 判斷是二分類還是多分類
            if len(unique_values) > 2:
                self.is_multiclass = True
                print("檢測到多分類問題")
            else:
                print("檢測到二分類問題")
            
            # 編碼目標變數（如果是字符串類型）
            if self.y_train.dtype == 'object':
                self.label_encoder = LabelEncoder()
                self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
                self.y_test_encoded = self.label_encoder.transform(self.y_test)
                self.classes = self.label_encoder.classes_
                print(f"目標類別（編碼後）：{list(zip(self.classes, range(len(self.classes))))}")
            else:
                # 如果目標已經是數值，則不需要額外編碼
                self.y_train_encoded = self.y_train.values
                self.y_test_encoded = self.y_test.values
                self.classes = unique_values
                print(f"目標類別（數值）：{self.classes}")
            
            print(f"成功載入資料集")
            print(f"特徵數量：{len(self.feature_names)}")
            print(f"訓練集大小：{len(self.X_train)}")
            print(f"測試集大小：{len(self.X_test)}")
            
            return True
        except Exception as e:
            print(f"載入資料時發生錯誤：{e}")
            return False
    
    def preprocess_data(self):
        """資料預處理"""
        # 標準化特徵
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 將標準化的數據轉回DataFrame以保留特徵名稱（用於特徵重要性分析）
        self.X_train_scaled_df = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        self.X_test_scaled_df = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)
        
        print("數據預處理完成：已執行標準化")
    
    def train_model(self):
        """訓練隨機森林模型"""
        print("\n=== 訓練隨機森林模型 ===")
        
        # 設定模型參數
        self.model = RandomForestClassifier(
            n_estimators=100,         # 樹的數量
            max_depth=None,           # 樹的最大深度，None表示無限制
            min_samples_split=2,      # 內部節點需要的最小樣本數
            min_samples_leaf=1,       # 葉節點需要的最小樣本數
            max_features='sqrt',      # 每次分裂時考慮的特徵數量
            bootstrap=True,           # 使用bootstrap樣本
            oob_score=True,           # 計算Out-of-Bag分數
            random_state=42,          # 隨機種子
            n_jobs=-1,                # 使用所有CPU核心
            class_weight='balanced'   # 處理類別不平衡
        )
        
        # 訓練模型
        self.model.fit(self.X_train_scaled, self.y_train_encoded)
        
        # 輸出Out-of-Bag分數
        oob_score = self.model.oob_score_
        print(f"模型訓練完成，Out-of-Bag分數：{oob_score:.4f}")
    
    def evaluate_model(self):
        """評估模型性能"""
        print("\n=== 模型評估 ===")
        
        # 在測試集上進行預測
        self.y_pred_encoded = self.model.predict(self.X_test_scaled)
        self.prediction_proba = self.model.predict_proba(self.X_test_scaled)
        
        # 如果需要將預測結果轉換回原始標籤
        if hasattr(self, 'label_encoder'):
            self.predictions = self.label_encoder.inverse_transform(self.y_pred_encoded)
        else:
            self.predictions = self.y_pred_encoded
        
        # 計算準確率
        accuracy = accuracy_score(self.y_test, self.predictions)
        print(f"準確率：{accuracy:.4f}")
        
        # 計算 F1 分數（使用 macro 平均以處理類別不平衡）
        f1 = f1_score(self.y_test, self.predictions, average='macro')
        print(f"F1分數（macro）：{f1:.4f}")
        
        # 類別報告
        class_report = classification_report(self.y_test, self.predictions)
        print("\n分類報告：")
        print(class_report)
        
        # 混淆矩陣
        cm = confusion_matrix(self.y_test, self.predictions)
        print("\n混淆矩陣：")
        print(cm)
        
        # 儲存評估結果
        self.evaluation_results = {
            'accuracy': accuracy,
            'f1_macro': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': self.predictions,
            'prediction_proba': self.prediction_proba,
            'oob_score': self.model.oob_score_
        }
    
    def analyze_feature_importance(self):
        """分析和視覺化隨機森林特徵重要性"""
        print("\n=== 分析特徵重要性 ===")
        
        # 獲取基於平均不純度減少的特徵重要性
        importance_mdi = self.model.feature_importances_
        
        # 結合特徵名稱和重要性分數
        feature_importance_mdi = dict(zip(self.feature_names, importance_mdi))
        
        # 按重要性排序
        sorted_importance_mdi = {k: v for k, v in sorted(
            feature_importance_mdi.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        # 顯示最重要的前10個特徵
        print("基於平均不純度減少的特徵重要性（MDI）：")
        for i, (feature, importance) in enumerate(list(sorted_importance_mdi.items())[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        # 計算基於置換的特徵重要性（計算時間可能較長）
        print("\n計算基於置換的特徵重要性（可能需要一些時間）...")
        try:
            # 使用較小的重複次數和樣本來加速計算
            perm_importance = permutation_importance(
                self.model, self.X_test_scaled, self.y_test_encoded,
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            # 結合特徵名稱和重要性分數
            feature_importance_perm = dict(zip(
                self.feature_names, 
                perm_importance.importances_mean
            ))
            
            # 按重要性排序
            sorted_importance_perm = {k: v for k, v in sorted(
                feature_importance_perm.items(), 
                key=lambda item: item[1], 
                reverse=True
            )}
            
            # 顯示最重要的前10個特徵
            print("\n基於置換的特徵重要性：")
            for i, (feature, importance) in enumerate(list(sorted_importance_perm.items())[:10], 1):
                std = perm_importance.importances_std[self.feature_names.index(feature)]
                print(f"{i}. {feature}: {importance:.4f} ± {std:.4f}")
                
            # 保存特徵重要性
            self.feature_importance = {
                'mdi': sorted_importance_mdi,
                'permutation': sorted_importance_perm
            }
        except Exception as e:
            print(f"計算置換重要性時發生錯誤：{e}")
            print("僅使用MDI重要性")
            self.feature_importance = {
                'mdi': sorted_importance_mdi
            }
    
    def analyze_trees(self):
        """分析隨機森林中的樹"""
        print("\n=== 分析隨機森林樹結構 ===")
        
        # 獲取所有樹的深度
        tree_depths = [estimator.get_depth() for estimator in self.model.estimators_]
        
        # 獲取所有樹的節點數量
        tree_nodes = [estimator.tree_.node_count for estimator in self.model.estimators_]
        
        # 計算統計資訊
        avg_depth = np.mean(tree_depths)
        max_depth = np.max(tree_depths)
        avg_nodes = np.mean(tree_nodes)
        max_nodes = np.max(tree_nodes)
        
        print(f"樹的平均深度：{avg_depth:.2f}")
        print(f"樹的最大深度：{max_depth}")
        print(f"樹的平均節點數：{avg_nodes:.2f}")
        print(f"樹的最大節點數：{max_nodes}")
        
        # 保存樹的統計資訊
        self.tree_stats = {
            'depths': tree_depths,
            'nodes': tree_nodes,
            'avg_depth': avg_depth,
            'max_depth': max_depth,
            'avg_nodes': avg_nodes,
            'max_nodes': max_nodes
        }
    
    def visualize_results(self):
        """視覺化模型結果"""
        result_dir = WORK_DIR / 'model_results' / self.model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩陣
        self.plot_confusion_matrix(result_dir)
        
        # 特徵重要性
        self.plot_feature_importance(result_dir)
        
        # 樹的統計資訊
        self.plot_tree_statistics(result_dir)
        
        # 決策邊界（適用於2D特徵）
        self.plot_decision_boundary(result_dir)
        
        # 錯誤分析
        self.plot_error_analysis(result_dir)
        
        # 二分類問題的ROC和PR曲線
        if not self.is_multiclass:
            self.plot_binary_curves(result_dir)
        else:
            # 多分類問題的各類別ROC曲線
            self.plot_multiclass_curves(result_dir)
    
    def plot_confusion_matrix(self, result_dir):
        """繪製混淆矩陣"""
        plt.figure(figsize=(10, 8))
        cm = self.evaluation_results['confusion_matrix']
        
        # 計算歸一化的混淆矩陣
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 繪製熱力圖
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('混淆矩陣')
        plt.xlabel('預測類別')
        plt.ylabel('實際類別')
        plt.tight_layout()
        plt.savefig(result_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 繪製歸一化的混淆矩陣
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('歸一化混淆矩陣')
        plt.xlabel('預測類別')
        plt.ylabel('實際類別')
        plt.tight_layout()
        plt.savefig(result_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, result_dir):
        """繪製特徵重要性圖"""
        # MDI特徵重要性
        if 'mdi' in self.feature_importance:
            sorted_features = list(self.feature_importance['mdi'].items())[:15]
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(features, importances, color='darkgreen')
            plt.title('特徵重要性 (基於平均不純度減少)', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.gca().invert_yaxis()  # 反轉Y軸，使最重要的特徵在上方
            plt.tight_layout()
            plt.savefig(result_dir / 'feature_importance_mdi.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 置換特徵重要性
        if 'permutation' in self.feature_importance:
            sorted_features = list(self.feature_importance['permutation'].items())[:15]
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(features, importances, color='darkblue')
            plt.title('特徵重要性 (基於置換)', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(result_dir / 'feature_importance_permutation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 比較兩種特徵重要性（如果都有計算）
        if 'mdi' in self.feature_importance and 'permutation' in self.feature_importance:
            # 選取兩種方法中都有的前10個特徵
            top_features = set()
            for feature, _ in list(self.feature_importance['mdi'].items())[:10]:
                top_features.add(feature)
            for feature, _ in list(self.feature_importance['permutation'].items())[:10]:
                top_features.add(feature)
            
            top_features = list(top_features)[:10]  # 限制為最多10個特徵
            
            # 獲取這些特徵在兩種方法中的重要性
            mdi_values = [self.feature_importance['mdi'].get(f, 0) for f in top_features]
            perm_values = [self.feature_importance['permutation'].get(f, 0) for f in top_features]
            
            # 歸一化重要性值，使其可比
            mdi_values = [v / sum(mdi_values) for v in mdi_values]
            perm_values = [v / sum(perm_values) for v in perm_values]
            
            # 創建比較圖
            plt.figure(figsize=(12, 8))
            x = np.arange(len(top_features))
            width = 0.35
            
            plt.bar(x - width/2, mdi_values, width, label='MDI', color='darkgreen')
            plt.bar(x + width/2, perm_values, width, label='Permutation', color='darkblue')
            
            plt.xlabel('特徵')
            plt.ylabel('歸一化重要性')
            plt.title('特徵重要性比較 (MDI vs. Permutation)')
            plt.xticks(x, top_features, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(result_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_tree_statistics(self, result_dir):
        """繪製樹的統計資訊"""
        if hasattr(self, 'tree_stats'):
            # 繪製樹深度分布
            plt.figure(figsize=(10, 6))
            plt.hist(self.tree_stats['depths'], bins=20, color='blue', alpha=0.7, edgecolor='black')
            plt.axvline(self.tree_stats['avg_depth'], color='red', linestyle='--', 
                      label=f'平均深度: {self.tree_stats["avg_depth"]:.2f}')
            plt.xlabel('樹的深度')
            plt.ylabel('樹的數量')
            plt.title('隨機森林中樹的深度分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(result_dir / 'tree_depth_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 繪製樹節點數分布
            plt.figure(figsize=(10, 6))
            plt.hist(self.tree_stats['nodes'], bins=20, color='green', alpha=0.7, edgecolor='black')
            plt.axvline(self.tree_stats['avg_nodes'], color='red', linestyle='--', 
                      label=f'平均節點數: {self.tree_stats["avg_nodes"]:.2f}')
            plt.xlabel('樹的節點數')
            plt.ylabel('樹的數量')
            plt.title('隨機森林中樹的節點數分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(result_dir / 'tree_nodes_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_decision_boundary(self, result_dir):
        """繪製決策邊界（僅當特徵數量為2時）"""
        # 只有在特徵數量為2時才繪製決策邊界
        if len(self.feature_names) == 2:
            try:
                # 創建網格點
                h = 0.02  # 網格步長
                x_min, x_max = self.X_test_scaled_df.iloc[:, 0].min() - 1, self.X_test_scaled_df.iloc[:, 0].max() + 1
                y_min, y_max = self.X_test_scaled_df.iloc[:, 1].min() - 1, self.X_test_scaled_df.iloc[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                # 在網格點上進行預測
                Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # 繪製決策邊界和散點圖
                plt.figure(figsize=(10, 8))
                plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
                
                # 繪製測試點
                for i, cls in enumerate(np.unique(self.y_test_encoded)):
                    idx = np.where(self.y_test_encoded == cls)
                    plt.scatter(self.X_test_scaled_df.iloc[idx, 0], self.X_test_scaled_df.iloc[idx, 1], 
                               label=f'Class {self.classes[i]}', s=60, alpha=0.9, edgecolor='k')
                
                plt.xlabel(self.feature_names[0], fontsize=12)
                plt.ylabel(self.feature_names[1], fontsize=12)
                plt.title('隨機森林決策邊界', fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.savefig(result_dir / 'decision_boundary.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"繪製決策邊界時發生錯誤：{e}")
    
    def plot_error_analysis(self, result_dir):
        """繪製錯誤分析圖"""
        try:
            # 獲取正確和錯誤的預測
            correct = self.y_test_encoded == self.y_pred_encoded
            incorrect = ~correct
            
            # 如果只有一個特徵，就繪製單變量分析
            if len(self.feature_names) == 1:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.X_test_scaled_df.iloc[correct, 0], [0] * sum(correct), 
                          label='正確預測', alpha=0.6, s=50, c='green')
                plt.scatter(self.X_test_scaled_df.iloc[incorrect, 0], [0] * sum(incorrect), 
                          label='錯誤預測', alpha=0.6, s=50, c='red')
                plt.xlabel(self.feature_names[0], fontsize=12)
                plt.yticks([])
                plt.title('單變量錯誤分析', fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.savefig(result_dir / 'error_analysis_1d.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 如果有兩個特徵，繪製二維錯誤分析
            elif len(self.feature_names) == 2:
                plt.figure(figsize=(10, 8))
                plt.scatter(self.X_test_scaled_df.iloc[correct, 0], self.X_test_scaled_df.iloc[correct, 1], 
                          label='正確預測', alpha=0.6, s=50, c='green')
                plt.scatter(self.X_test_scaled_df.iloc[incorrect, 0], self.X_test_scaled_df.iloc[incorrect, 1], 
                          label='錯誤預測', alpha=0.6, s=50, c='red')
                plt.xlabel(self.feature_names[0], fontsize=12)
                plt.ylabel(self.feature_names[1], fontsize=12)
                plt.title('二維錯誤分析', fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.savefig(result_dir / 'error_analysis_2d.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 如果有超過兩個特徵，繪製前兩個主要特徵的錯誤分析
            else:
                # 以MDI特徵重要性為基礎選擇前兩個特徵
                if hasattr(self, 'feature_importance') and 'mdi' in self.feature_importance:
                    top_features = list(self.feature_importance['mdi'].keys())[:2]
                    feature_idx = [self.feature_names.index(f) for f in top_features]
                    
                    plt.figure(figsize=(10, 8))
                    plt.scatter(self.X_test_scaled_df.iloc[correct, feature_idx[0]], 
                              self.X_test_scaled_df.iloc[correct, feature_idx[1]], 
                              label='正確預測', alpha=0.6, s=50, c='green')
                    plt.scatter(self.X_test_scaled_df.iloc[incorrect, feature_idx[0]], 
                              self.X_test_scaled_df.iloc[incorrect, feature_idx[1]], 
                              label='錯誤預測', alpha=0.6, s=50, c='red')
                    plt.xlabel(top_features[0], fontsize=12)
                    plt.ylabel(top_features[1], fontsize=12)
                    plt.title('二維錯誤分析（基於前兩個重要特徵）', fontsize=14)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(result_dir / 'error_analysis_top_features.png', dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            print(f"繪製錯誤分析圖時發生錯誤：{e}")
    
    def plot_binary_curves(self, result_dir):
        """繪製二分類評估曲線 (ROC和PR)"""
        try:
            # 獲取預測概率（確保這是類別1的概率）
            if self.prediction_proba.ndim > 1:
                y_score = self.prediction_proba[:, 1]
            else:
                y_score = self.prediction_proba
            
            # 計算ROC曲線
            fpr, tpr, _ = roc_curve(self.y_test_encoded, y_score)
            roc_auc = auc(fpr, tpr)
            
            # 計算PR曲線
            precision, recall, _ = precision_recall_curve(self.y_test_encoded, y_score)
            avg_precision = average_precision_score(self.y_test_encoded, y_score)
            
            # 繪製ROC曲線
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC曲線')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # 繪製PR曲線
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color='blue', lw=2, 
                   label=f'PR曲線 (AP = {avg_precision:.3f})')
            plt.fill_between(recall, precision, alpha=0.2, color='blue')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall曲線')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(result_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存ROC和PR的數據
            curve_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'precision': np.interp(fpr, recall[::-1], precision[::-1])  # 將precision對應到相同的fpr值
            })
            curve_data.to_csv(result_dir / 'roc_pr_data.csv', index=False)
            
            # 保存AUC和AP的數據
            metrics_data = pd.DataFrame({
                'metric': ['ROC AUC', 'Average Precision'],
                'value': [roc_auc, avg_precision]
            })
            metrics_data.to_csv(result_dir / 'roc_pr_metrics.csv', index=False)
        
        except Exception as e:
            print(f"繪製二分類評估曲線時發生錯誤: {e}")
            print("跳過ROC和PR曲線繪製")
    
    def plot_multiclass_curves(self, result_dir):
        """繪製多分類評估曲線 (每類別的ROC)"""
        try:
            plt.figure(figsize=(12, 10))
            
            # 為每個類別計算ROC曲線
            for i, class_name in enumerate(self.classes):
                # 將問題轉換為one-vs-rest(二分類)
                y_true_binary = (self.y_test_encoded == i).astype(int)
                y_score = self.prediction_proba[:, i]
                
                # 計算ROC曲線
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                # 繪製ROC曲線
                plt.plot(fpr, tpr, lw=2, 
                        label=f'類別 {class_name} (AUC = {roc_auc:.3f})')
            
            # 添加隨機猜測基準線
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('多類別ROC曲線 (one-vs-rest)', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(result_dir / 'multiclass_roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 計算並保存每個類別的AUC值
            auc_values = []
            for i, class_name in enumerate(self.classes):
                y_true_binary = (self.y_test_encoded == i).astype(int)
                y_score = self.prediction_proba[:, i]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)
                    auc_values.append({
                        'class': class_name,
                        'auc': roc_auc
                    })
                except:
                    print(f"計算類別 {class_name} 的AUC時發生錯誤")
            
            # 保存AUC數據
            if auc_values:
                auc_df = pd.DataFrame(auc_values)
                auc_df.to_csv(result_dir / 'multiclass_auc_values.csv', index=False)
        
        except Exception as e:
            print(f"繪製多分類評估曲線時發生錯誤: {e}")
            print("跳過多分類ROC曲線繪製")
    
    def save_model_and_results(self):
        """儲存模型和評估結果"""
        # 儲存模型
        model_path = WORK_DIR / 'saved_models' / f'{self.model_name}.pkl'
        model_path.parent.mkdir(exist_ok=True)
        
        # 儲存模型
        joblib.dump(self.model, model_path)
        
        # 儲存標準化器和其他元數據
        metadata_path = WORK_DIR / 'saved_models' / f'{self.model_name}_metadata.pkl'
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_multiclass': self.is_multiclass
        }
        
        if hasattr(self, 'label_encoder'):
            metadata['label_encoder'] = self.label_encoder
        
        joblib.dump(metadata, metadata_path)
        
        # 儲存評估結果
        results_path = WORK_DIR / 'model_results' / self.model_name / 'evaluation_results.csv'
        results_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 創建結果字典
        results = {
            'model_name': [self.model_name],
            'accuracy': [self.evaluation_results['accuracy']],
            'f1_macro': [self.evaluation_results['f1_macro']],
            'oob_score': [self.evaluation_results['oob_score']]
        }
        
        # 為每個類別加入召回率和精確度
        for i, cls in enumerate(self.classes):
            if self.is_multiclass:
                # 使用sklearn的classification_report函數獲取每個類別的指標
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, self.predictions, average=None, labels=[cls]
                )
                
                results[f'precision_{cls}'] = [precision[0]]
                results[f'recall_{cls}'] = [recall[0]]
                results[f'f1_{cls}'] = [f1[0]]
            else:
                # 二分類問題
                cls_idx = 1  # 通常關注正類
                from sklearn.metrics import precision_score, recall_score
                precision = precision_score(self.y_test, self.predictions)
                recall = recall_score(self.y_test, self.predictions)
                
                results[f'precision'] = [precision]
                results[f'recall'] = [recall]
        
        # 儲存為CSV
        pd.DataFrame(results).to_csv(results_path, index=False)
        
        # 儲存特徵重要性
        if hasattr(self, 'feature_importance'):
            # MDI特徵重要性
            if 'mdi' in self.feature_importance:
                mdi_df = pd.DataFrame([
                    {'feature': feature, 'importance': importance, 'method': 'MDI'}
                    for feature, importance in self.feature_importance['mdi'].items()
                ])
                
                # 置換特徵重要性
                if 'permutation' in self.feature_importance:
                    perm_df = pd.DataFrame([
                        {'feature': feature, 'importance': importance, 'method': 'Permutation'}
                        for feature, importance in self.feature_importance['permutation'].items()
                    ])
                    # 合併兩種特徵重要性
                    importance_df = pd.concat([mdi_df, perm_df], ignore_index=True)
                else:
                    importance_df = mdi_df
                
                importance_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv', index=False)
        
        # 儲存樹的統計資訊
        if hasattr(self, 'tree_stats'):
            tree_stats_df = pd.DataFrame({
                'statistic': ['avg_depth', 'max_depth', 'avg_nodes', 'max_nodes'],
                'value': [
                    self.tree_stats['avg_depth'],
                    self.tree_stats['max_depth'],
                    self.tree_stats['avg_nodes'],
                    self.tree_stats['max_nodes']
                ]
            })
            tree_stats_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'tree_statistics.csv', index=False)
        
        # 儲存預測結果
        predictions_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions
        })
        
        # 添加各類別的預測機率
        if self.is_multiclass:
            for i, cls in enumerate(self.classes):
                predictions_df[f'prob_{cls}'] = self.prediction_proba[:, i]
        else:
            if self.prediction_proba.ndim > 1:
                # 如果有兩列，第一列是負類，第二列是正類
                predictions_df['probability'] = self.prediction_proba[:, 1]
            else:
                # 如果只有一列，直接是正類概率
                predictions_df['probability'] = self.prediction_proba
        
        predictions_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'predictions.csv', index=False)
        
        print("\n模型和結果已儲存:")
        print(f"- 模型: {model_path}")
        print(f"- 元數據: {metadata_path}")
        print(f"- 評估結果: {results_path}")
        print(f"- 特徵重要性: {WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv'}")
        print(f"- 預測結果: {WORK_DIR / 'model_results' / self.model_name / 'predictions.csv'}")
    
    def generate_report(self):
        """生成模型報告"""
        report = []
        report.append(f"=== 隨機森林模型報告 ===\n")
        
        # 基本資訊
        report.append("模型資訊:")
        report.append(f"- 模型名稱: {self.model_name}")
        report.append(f"- 問題類型: {'多分類' if self.is_multiclass else '二分類'}")
        report.append(f"- 訓練集大小: {len(self.X_train)}")
        report.append(f"- 測試集大小: {len(self.X_test)}")
        report.append(f"- 特徵數量: {len(self.feature_names)}")
        report.append(f"- 目標類別: {', '.join(map(str, self.classes))}")
        
        # 模型參數
        report.append("\n模型參數:")
        params = self.model.get_params()
        report.append(f"- 樹的數量: {params['n_estimators']}")
        report.append(f"- 最大深度: {params['max_depth']}")
        report.append(f"- 最小分裂樣本數: {params['min_samples_split']}")
        report.append(f"- 最小葉節點樣本數: {params['min_samples_leaf']}")
        report.append(f"- 特徵抽樣方式: {params['max_features']}")
        
        # 樹的統計資訊
        if hasattr(self, 'tree_stats'):
            report.append("\n樹的統計資訊:")
            report.append(f"- 平均深度: {self.tree_stats['avg_depth']:.2f}")
            report.append(f"- 最大深度: {self.tree_stats['max_depth']}")
            report.append(f"- 平均節點數: {self.tree_stats['avg_nodes']:.2f}")
            report.append(f"- 最大節點數: {self.tree_stats['max_nodes']}")
        
        # 性能指標
        report.append("\n性能指標:")
        report.append(f"- 準確率: {self.evaluation_results['accuracy']:.4f}")
        report.append(f"- F1分數(macro): {self.evaluation_results['f1_macro']:.4f}")
        report.append(f"- OOB分數: {self.evaluation_results['oob_score']:.4f}")
        
        # 混淆矩陣
        report.append("\n混淆矩陣:")
        cm = self.evaluation_results['confusion_matrix']
        cm_string = []
        cm_string.append("預測 \\ 實際 | " + " | ".join(map(str, self.classes)))
        cm_string.append("-" * 50)
        for i, row in enumerate(cm):
            cm_string.append(f"{self.classes[i]}     | " + " | ".join(str(cell).rjust(6) for cell in row))
        report.extend(cm_string)
        
        # 特徵重要性
        if hasattr(self, 'feature_importance') and 'mdi' in self.feature_importance:
            report.append("\n特徵重要性 (前10，基於MDI):")
            sorted_features = list(self.feature_importance['mdi'].items())[:10]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                report.append(f"{i}. {feature}: {importance:.4f}")
        
        # 儲存報告
        report_path = WORK_DIR / 'model_results' / self.model_name / 'model_report.txt'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n模型報告已儲存: {report_path}")
    
    def run_random_forest_pipeline(self):
        """執行完整的隨機森林模型建立流程"""
        print("=== 開始建立隨機森林模型 ===")
        
        # 載入資料
        if not self.load_data():
            return
        
        # 預處理
        self.preprocess_data()
        
        # 訓練模型
        self.train_model()
        
        # 評估模型
        self.evaluate_model()
        
        # 分析特徵重要性
        self.analyze_feature_importance()
        
        # 分析樹結構
        self.analyze_trees()
        
        # 視覺化
        self.visualize_results()
        
        # 儲存模型和結果
        self.save_model_and_results()
        
        # 生成報告
        self.generate_report()
        
        print("\n=== 隨機森林模型建立完成 ===")

if __name__ == "__main__":
    # 運行隨機森林模型
    rf_model = RandomForestModel()
    rf_model.run_random_forest_pipeline()
