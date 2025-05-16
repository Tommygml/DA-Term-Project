"""
prediction_05.py - 支持向量機模型實作
功能：使用SVM進行學生輟學預測
作者：Tommy
日期：2025-05-17
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
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

class SVMModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.model_name = "SVM"
        self.classes = None
        self.feature_names = None
        self.is_multiclass = False
        self.best_params = None
        
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
    
    def find_best_parameters(self):
        """使用網格搜索找到最佳參數"""
        print("\n=== 執行網格搜索以找到最佳參數 ===")
        
        # 使用較小的子集進行參數調優以節省時間
        if len(self.X_train) > 1000:
            # 隨機取樣20%的數據進行參數調優
            sample_size = int(len(self.X_train) * 0.2)
            idx = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train_scaled[idx]
            y_sample = self.y_train_encoded[idx]
            print(f"使用 {sample_size} 筆樣本資料進行參數調優")
        else:
            X_sample = self.X_train_scaled
            y_sample = self.y_train_encoded
            print(f"使用全部 {len(self.X_train)} 筆訓練資料進行參數調優")
        
        # 定義參數網格
        param_grid = {
            'C': [0.1, 1, 10, 100],  # 正則化參數
            'gamma': ['scale', 'auto', 0.1, 0.01],  # 核函數係數
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # 核函數類型
        }
        
        if self.is_multiclass:
            # 多分類問題的額外參數
            param_grid['decision_function_shape'] = ['ovo', 'ovr']
        
        # 使用網格搜索找到最佳參數
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42, class_weight='balanced'),
            param_grid=param_grid,
            cv=5,  # 5折交叉驗證
            scoring='f1_macro',  # 使用F1分數作為評分標準
            n_jobs=-1,  # 使用所有CPU核心
            verbose=1
        )
        
        # 執行網格搜索
        grid_search.fit(X_sample, y_sample)
        
        # 獲取最佳參數
        self.best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"最佳參數：{self.best_params}")
        print(f"最佳交叉驗證分數：{best_score:.4f}")
        print(f"網格搜索完成，將使用最佳參數訓練模型")
        
        # 保存網格搜索結果
        cv_results = pd.DataFrame(grid_search.cv_results_)
        result_dir = WORK_DIR / 'model_results' / self.model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(result_dir / 'grid_search_results.csv', index=False)
        
        return self.best_params
    
    def train_model(self):
        """訓練SVM模型"""
        print("\n=== 訓練SVM模型 ===")
        
        # 找到最佳參數
        if self.best_params is None:
            self.find_best_parameters()
        
        # 使用最佳參數創建SVM模型
        self.model = SVC(
            C=self.best_params['C'],
            gamma=self.best_params['gamma'],
            kernel=self.best_params['kernel'],
            probability=True,  # 啟用概率輸出
            random_state=42,
            class_weight='balanced',  # 處理類別不平衡
            **({} if not self.is_multiclass else 
               {'decision_function_shape': self.best_params.get('decision_function_shape', 'ovr')})
        )
        
        # 訓練模型
        self.model.fit(self.X_train_scaled, self.y_train_encoded)
        
        # 取得支持向量的數量
        n_support = self.model.n_support_
        total_sv = sum(n_support)
        
        print(f"模型訓練完成")
        print(f"支持向量的數量：{total_sv} ({total_sv/len(self.X_train)*100:.2f}% 的訓練樣本)")
        
        # 對於每個類別，顯示支持向量的數量
        if len(n_support) > 1:
            for i, (cls, count) in enumerate(zip(self.classes, n_support)):
                print(f"類別 {cls} 的支持向量數量：{count}")
    
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
            'support_vectors': self.model.n_support_
        }
    
    def analyze_feature_importance(self):
        """分析特徵重要性（使用置換重要性，因為SVM沒有內建特徵重要性）"""
        print("\n=== 分析特徵重要性（使用置換重要性）===")
        
        try:
            # 計算置換特徵重要性
            perm_importance = permutation_importance(
                self.model, self.X_test_scaled, self.y_test_encoded,
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # 結合特徵名稱和重要性分數
            feature_importance = dict(zip(
                self.feature_names, 
                perm_importance.importances_mean
            ))
            
            # 按重要性排序
            self.feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            )}
            
            # 顯示最重要的前10個特徵
            print("\n基於置換的特徵重要性：")
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10], 1):
                std = perm_importance.importances_std[self.feature_names.index(feature)]
                print(f"{i}. {feature}: {importance:.4f} ± {std:.4f}")
                
        except Exception as e:
            print(f"計算特徵重要性時發生錯誤：{e}")
            print("跳過特徵重要性分析")
            self.feature_importance = {}
    
    def analyze_support_vectors(self):
        """分析支持向量的分布"""
        print("\n=== 分析支持向量 ===")
        
        if hasattr(self.model, 'support_'):
            # 獲取支持向量的索引
            support_indices = self.model.support_
            n_support = self.model.n_support_
            
            # 獲取支持向量
            support_vectors = self.model.support_vectors_
            
            # 計算所有訓練樣本的決策函數值
            if self.is_multiclass:
                # 多分類問題下，決策函數返回每個樣本對每個類別的距離
                decision_function = self.model.decision_function(self.X_train_scaled)
                if decision_function.ndim > 1:
                    # 選擇最大距離（最有信心的預測）
                    confidence = np.max(decision_function, axis=1)
                else:
                    confidence = decision_function
            else:
                # 二分類問題下，決策函數返回每個樣本到超平面的距離
                confidence = self.model.decision_function(self.X_train_scaled)
            
            # 計算支持向量的比例
            sv_ratio = len(support_indices) / len(self.X_train)
            
            print(f"支持向量數量：{len(support_indices)} ({sv_ratio*100:.2f}% 的訓練樣本)")
            print(f"支持向量的平均距離：{np.mean(np.abs(confidence[support_indices])):.4f}")
            print(f"非支持向量的平均距離：{np.mean(np.abs(confidence[~np.isin(range(len(self.X_train)), support_indices)])):.4f}")
            
            # 保存支持向量分析結果
            self.support_vector_analysis = {
                'support_indices': support_indices,
                'n_support': n_support,
                'sv_ratio': sv_ratio,
                'confidence': confidence
            }
        else:
            print("無法獲取支持向量，跳過支持向量分析")
    
    def visualize_results(self):
        """視覺化模型結果"""
        result_dir = WORK_DIR / 'model_results' / self.model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩陣
        self.plot_confusion_matrix(result_dir)
        
        # 特徵重要性
        if hasattr(self, 'feature_importance') and self.feature_importance:
            self.plot_feature_importance(result_dir)
        
        # 支持向量分析
        if hasattr(self, 'support_vector_analysis'):
            self.plot_support_vector_analysis(result_dir)
        
        # 決策邊界（適用於2D特徵）
        self.plot_decision_boundary(result_dir)
        
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
        # 取前15個最重要的特徵
        sorted_features = list(self.feature_importance.items())[:15]
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(features, importances, color='navy')
        plt.title('特徵重要性 (基於置換)', fontsize=14)
        plt.xlabel('重要性', fontsize=12)
        plt.gca().invert_yaxis()  # 反轉Y軸，使最重要的特徵在上方
        plt.tight_layout()
        plt.savefig(result_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_support_vector_analysis(self, result_dir):
        """繪製支持向量分析圖"""
        if 'confidence' in self.support_vector_analysis:
            # 獲取支持向量和非支持向量的決策函數值
            support_indices = self.support_vector_analysis['support_indices']
            confidence = self.support_vector_analysis['confidence']
            
            # 將樣本分為支持向量和非支持向量
            sv_confidence = confidence[support_indices]
            nonsv_confidence = confidence[~np.isin(range(len(self.X_train)), support_indices)]
            
            # 繪製決策函數值分布
            plt.figure(figsize=(10, 6))
            
            plt.hist(sv_confidence, bins=30, alpha=0.5, label='支持向量', color='red')
            plt.hist(nonsv_confidence, bins=30, alpha=0.5, label='非支持向量', color='blue')
            
            plt.xlabel('決策函數值（絕對值）')
            plt.ylabel('樣本數量')
            plt.title('支持向量和非支持向量的決策函數分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(result_dir / 'support_vector_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 支持向量的數量條形圖
            if len(self.support_vector_analysis['n_support']) > 1:
                plt.figure(figsize=(8, 6))
                plt.bar(range(len(self.classes)), self.support_vector_analysis['n_support'], color='green')
                plt.xticks(range(len(self.classes)), self.classes)
                plt.xlabel('類別')
                plt.ylabel('支持向量數量')
                plt.title('各類別的支持向量數量')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(result_dir / 'support_vector_count.png', dpi=300, bbox_inches='tight')
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
                plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
                
                # 繪製測試點
                for i, cls in enumerate(np.unique(self.y_test_encoded)):
                    idx = np.where(self.y_test_encoded == cls)
                    plt.scatter(self.X_test_scaled_df.iloc[idx, 0], self.X_test_scaled_df.iloc[idx, 1], 
                               label=f'Class {self.classes[i]}', s=60, alpha=0.9, edgecolor='k')
                
                # 繪製支持向量
                support_vectors = self.model.support_vectors_
                plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                          s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
                
                plt.xlabel(self.feature_names[0], fontsize=12)
                plt.ylabel(self.feature_names[1], fontsize=12)
                plt.title('SVM決策邊界和支持向量', fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.savefig(result_dir / 'decision_boundary.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"繪製決策邊界時發生錯誤：{e}")
    
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
            'is_multiclass': self.is_multiclass,
            'best_params': self.best_params
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
            'f1_macro': [self.evaluation_results['f1_macro']]
        }
        
        # 加入支持向量數量
        if 'support_vectors' in self.evaluation_results:
            total_sv = sum(self.evaluation_results['support_vectors'])
            results['total_support_vectors'] = [total_sv]
            results['support_vector_ratio'] = [total_sv / len(self.X_train)]
        
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
        if hasattr(self, 'feature_importance') and self.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in self.feature_importance.items()
            ])
            importance_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv', index=False)
        
        # 儲存支持向量分析
        if hasattr(self, 'support_vector_analysis'):
            sv_stats_df = pd.DataFrame({
                'class': list(self.classes),
                'support_vectors': list(self.support_vector_analysis['n_support'])
            })
            sv_stats_df['ratio'] = sv_stats_df['support_vectors'] / sv_stats_df['support_vectors'].sum()
            sv_stats_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'support_vector_stats.csv', index=False)
        
        # 儲存網格搜索結果
        if hasattr(self, 'best_params'):
            params_df = pd.DataFrame({
                'parameter': list(self.best_params.keys()),
                'best_value': list(self.best_params.values())
            })
            params_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'best_parameters.csv', index=False)
        
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
        if hasattr(self, 'feature_importance') and self.feature_importance:
            print(f"- 特徵重要性: {WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv'}")
        print(f"- 預測結果: {WORK_DIR / 'model_results' / self.model_name / 'predictions.csv'}")
    
    def generate_report(self):
        """生成模型報告"""
        report = []
        report.append(f"=== SVM模型報告 ===\n")
        
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
        report.append(f"- 核函數: {params['kernel']}")
        report.append(f"- C值: {params['C']}")
        report.append(f"- Gamma: {params['gamma']}")
        if self.is_multiclass:
            report.append(f"- 決策函數形狀: {params.get('decision_function_shape', 'ovr')}")
        
        # 支持向量資訊
        total_sv = sum(self.model.n_support_)
        report.append("\n支持向量資訊:")
        report.append(f"- 總支持向量數量: {total_sv} ({total_sv/len(self.X_train)*100:.2f}% 的訓練樣本)")
        for i, (cls, count) in enumerate(zip(self.classes, self.model.n_support_)):
            report.append(f"- 類別 {cls} 的支持向量數量: {count} ({count/total_sv*100:.2f}%)")
        
        # 性能指標
        report.append("\n性能指標:")
        report.append(f"- 準確率: {self.evaluation_results['accuracy']:.4f}")
        report.append(f"- F1分數(macro): {self.evaluation_results['f1_macro']:.4f}")
        
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
        if hasattr(self, 'feature_importance') and self.feature_importance:
            report.append("\n特徵重要性 (前10，基於置換):")
            sorted_features = list(self.feature_importance.items())[:10]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                report.append(f"{i}. {feature}: {importance:.4f}")
        
        # 儲存報告
        report_path = WORK_DIR / 'model_results' / self.model_name / 'model_report.txt'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n模型報告已儲存: {report_path}")
    
    def run_svm_pipeline(self):
        """執行完整的SVM模型建立流程"""
        print("=== 開始建立SVM模型 ===")
        
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
        
        # 分析支持向量
        self.analyze_support_vectors()
        
        # 視覺化
        self.visualize_results()
        
        # 儲存模型和結果
        self.save_model_and_results()
        
        # 生成報告
        self.generate_report()
        
        print("\n=== SVM模型建立完成 ===")

if __name__ == "__main__":
    # 運行SVM模型
    svm_model = SVMModel()
    svm_model.run_svm_pipeline()
