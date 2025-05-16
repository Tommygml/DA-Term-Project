"""
prediction_06.py - 多層感知器(MLP)模型實作
功能：使用神經網絡進行學生輟學預測
作者：Tommy
日期：2025-05-17
適用環境: Google Colab
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# 設定中文顯示
# 修改字型設定
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'DejaVu Sans', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定Google Drive工作目錄
BASE_DIR = Path("/content/drive/MyDrive/DA Term Project")

class MLPModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.model_name = "MLP"
        self.classes = None
        self.feature_names = None
        self.is_multiclass = False
        self.best_params = None
        self.train_loss_curve = None
        
    def create_directories(self):
        """創建必要的目錄結構"""
        # 創建保存模型的目錄
        model_dir = BASE_DIR / 'saved_models'
        model_dir.mkdir(exist_ok=True)
        
        # 創建保存結果的目錄
        result_dir = BASE_DIR / 'model_results' / self.model_name
        result_dir.mkdir(exist_ok=True, parents=True)
        
        # 創建可視化結果的目錄
        viz_dir = BASE_DIR / 'visualizations' / self.model_name
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        print("目錄結構已創建")
        
    def load_data(self):
        """載入最終的訓練集和測試集"""
        try:
            # 優先嘗試載入修復後的數據集
            try:
                train_df = pd.read_csv(BASE_DIR / 'train_final_fixed.csv')
                test_df = pd.read_csv(BASE_DIR / 'test_final_fixed.csv')
                print("已載入修復後的數據集")
            except:
                # 如果找不到修復後的數據集，則嘗試載入原始最終數據集
                train_df = pd.read_csv(BASE_DIR / 'train_final.csv')
                test_df = pd.read_csv(BASE_DIR / 'test_final.csv')
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
        
        # 將標準化的數據轉回DataFrame以保留特徵名稱
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
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # 隱藏層結構
            'activation': ['relu', 'tanh'],  # 激活函數
            'solver': ['adam'],  # 優化器
            'alpha': [0.0001, 0.001, 0.01],  # L2正則化參數
            'learning_rate_init': [0.001, 0.01],  # 初始學習率
            'batch_size': [32, 64, 128]  # 批次大小
        }
        
        # 使用網格搜索找到最佳參數
        grid_search = GridSearchCV(
            MLPClassifier(max_iter=100, early_stopping=True, random_state=42),
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
        result_dir = BASE_DIR / 'model_results' / self.model_name
        cv_results.to_csv(result_dir / 'grid_search_results.csv', index=False)
        
        return self.best_params
    
    def train_model(self):
        """訓練MLP模型"""
        print("\n=== 訓練MLP模型 ===")
        
        # 找到最佳參數
        if self.best_params is None:
            self.find_best_parameters()
        
        # 使用最佳參數創建MLP模型
        self.model = MLPClassifier(
            hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
            activation=self.best_params['activation'],
            solver=self.best_params['solver'],
            alpha=self.best_params['alpha'],
            learning_rate_init=self.best_params['learning_rate_init'],
            batch_size=self.best_params['batch_size'],
            max_iter=1000,  # 最大迭代次數
            early_stopping=True,  # 啟用早停
            validation_fraction=0.1,  # 用於早停的驗證集比例
            n_iter_no_change=10,  # 沒有改善時的迭代次數
            random_state=42,
            verbose=True  # 顯示訓練過程
        )
        
        # 訓練模型
        self.model.fit(self.X_train_scaled, self.y_train_encoded)
        
        # 保存損失曲線
        self.train_loss_curve = self.model.loss_curve_
        
        print(f"模型訓練完成，迭代次數：{self.model.n_iter_}")
        # 修改後的代碼
        if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None:
            print(f"最佳驗證損失：{self.model.best_loss_:.6f}")
        else:
            print("最佳驗證損失：不適用")
        
        # 輸出網絡架構信息
        hidden_layer_sizes = self.model.hidden_layer_sizes
        print(f"網絡架構：輸入層({len(self.feature_names)}) - ", end="")
        for i, size in enumerate(hidden_layer_sizes):
            print(f"隱藏層{i+1}({size}) - ", end="")
        print(f"輸出層({len(self.classes)})")
    
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
            'prediction_proba': self.prediction_proba
        }
    
    def analyze_neural_network(self):
        """分析神經網絡結構和權重"""
        print("\n=== 分析神經網絡結構和權重 ===")
        
        # 獲取網絡權重
        self.weights = self.model.coefs_
        self.biases = self.model.intercepts_
        
        # 層數統計
        n_layers = len(self.weights) + 1  # 加上輸入層
        layer_sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
        
        print(f"網絡總層數：{n_layers} (輸入層 + {n_layers-2} 隱藏層 + 輸出層)")
        print(f"各層神經元數量：{layer_sizes}")
        
        # 權重統計
        total_weights = sum(w.size for w in self.weights)
        total_biases = sum(b.size for b in self.biases)
        total_params = total_weights + total_biases
        
        print(f"總參數數量：{total_params}")
        print(f"權重數量：{total_weights}")
        print(f"偏置數量：{total_biases}")
        
        # 計算第一層的權重統計（可用於特徵重要性）
        first_layer_weights = self.weights[0]  # 第一層權重
        weight_importance = np.abs(first_layer_weights).mean(axis=1)  # 每個輸入特徵的平均權重
        
        # 結合特徵名稱和權重重要性
        self.feature_importance = dict(zip(self.feature_names, weight_importance))
        
        # 按重要性排序
        self.feature_importance = {k: v for k, v in sorted(
            self.feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        # 顯示最重要的前10個特徵
        print("\n基於第一層權重的特徵重要性：")
        for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10], 1):
            print(f"{i}. {feature}: {importance:.4f}")
    
    def visualize_results(self):
        """視覺化模型結果"""
        result_dir = BASE_DIR / 'model_results' / self.model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        viz_dir = BASE_DIR / 'visualizations' / self.model_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩陣
        self.plot_confusion_matrix(viz_dir)
        
        # 損失曲線
        self.plot_loss_curve(viz_dir)
        
        # 特徵重要性
        self.plot_feature_importance(viz_dir)
        
        # 網絡結構可視化
        self.plot_network_structure(viz_dir)
        
        # 二分類問題的ROC和PR曲線
        if not self.is_multiclass:
            self.plot_binary_curves(viz_dir)
        else:
            # 多分類問題的各類別ROC曲線
            self.plot_multiclass_curves(viz_dir)
    
    def plot_confusion_matrix(self, viz_dir):
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
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(viz_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curve(self, viz_dir):
        """繪製損失曲線"""
        if hasattr(self, 'train_loss_curve') and self.train_loss_curve is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_loss_curve, color='blue', linewidth=2)
            plt.title('訓練損失曲線')
            plt.xlabel('迭代次數')
            plt.ylabel('損失')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_feature_importance(self, viz_dir):
        """繪製特徵重要性圖"""
        if hasattr(self, 'feature_importance') and self.feature_importance:
            # 取前15個最重要的特徵
            sorted_features = list(self.feature_importance.items())[:15]
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(features, importances, color='purple')
            plt.title('特徵重要性 (基於第一層權重)', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.gca().invert_yaxis()  # 反轉Y軸，使最重要的特徵在上方
            plt.tight_layout()
            plt.savefig(viz_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_network_structure(self, viz_dir):
        """繪製網絡結構圖"""
        try:
            # 創建水平網絡圖
            layer_sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
            max_size = max(layer_sizes)
            
            fig_width = 4 + 2 * len(layer_sizes)
            fig_height = 4 + max_size * 0.3
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # 繪製各層
            layer_names = ["輸入層"] + [f"隱藏層 {i+1}" for i in range(len(layer_sizes)-2)] + ["輸出層"]
            
            for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
                # 層的x坐標
                x = i
                
                # 繪製層標籤
                ax.text(x, max_size + 1, name, ha='center', va='center', fontsize=12)
                
                # 繪製節點
                for j in range(size):
                    y = max_size - (max_size / size) * j - (max_size / size) / 2
                    
                    # 節點
                    circle = plt.Circle((x, y), 0.2, fill=True, color='skyblue', alpha=0.8, linewidth=1, edgecolor='black')
                    ax.add_patch(circle)
                    
                    # 節點標籤
                    if i == 0:  # 輸入層顯示特徵名稱
                        if len(self.feature_names) > size:
                            label = f"特徵 {j+1}"
                        else:
                            label = self.feature_names[j]
                        ax.text(x - 1.5, y, label, ha='right', va='center', fontsize=8)
                    elif i == len(layer_sizes) - 1:  # 輸出層顯示類別名稱
                        if j < len(self.classes):
                            ax.text(x + 1, y, str(self.classes[j]), ha='left', va='center', fontsize=10)
                
                # 如果不是最後一層，繪製連接線
                if i < len(layer_sizes) - 1:
                    next_size = layer_sizes[i+1]
                    
                    # 只為部分連接繪製線條以避免過度擁擠
                    max_lines = 30
                    step_size = 1  # 預設步長為1
                    if size * next_size > max_lines:
                        step_size = max(1, int(size * next_size / max_lines))
                    #else:
                    #    step_line = 1
                        
                    #for j in range(0, size, step_line):
                    for j in range(0, size, step_size):
                        y1 = max_size - (max_size / size) * j - (max_size / size) / 2
                        
                        #for k in range(0, next_size, step_line):
                        #for k in range(0, next_size, step_size):
                        for k in range(0, next_size, step_size):
                            y2 = max_size - (max_size / next_size) * k - (max_size / next_size) / 2
                            ax.plot([x, x+1], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
            
            ax.set_xlim(-2, len(layer_sizes) + 1)
            ax.set_ylim(-1, max_size + 2)
            ax.axis('off')
            plt.title('神經網絡結構', fontsize=14)
            plt.tight_layout()
            plt.savefig(viz_dir / 'network_structure.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"繪製網絡結構圖時發生錯誤：{e}")
    
    def plot_binary_curves(self, viz_dir):
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
            plt.savefig(viz_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存ROC和PR的數據
            curve_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'precision': np.interp(fpr, recall[::-1], precision[::-1])  # 將precision對應到相同的fpr值
            })
            curve_data.to_csv(viz_dir / 'roc_pr_data.csv', index=False)
            
            # 保存AUC和AP的數據
            metrics_data = pd.DataFrame({
                'metric': ['ROC AUC', 'Average Precision'],
                'value': [roc_auc, avg_precision]
            })
            metrics_data.to_csv(viz_dir / 'roc_pr_metrics.csv', index=False)
        
        except Exception as e:
            print(f"繪製二分類評估曲線時發生錯誤: {e}")
            print("跳過ROC和PR曲線繪製")
    
    def plot_multiclass_curves(self, viz_dir):
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
            plt.savefig(viz_dir / 'multiclass_roc_curves.png', dpi=300, bbox_inches='tight')
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
                auc_df.to_csv(viz_dir / 'multiclass_auc_values.csv', index=False)
        
        except Exception as e:
            print(f"繪製多分類評估曲線時發生錯誤: {e}")
            print("跳過多分類ROC曲線繪製")
    
    def save_model_and_results(self):
        """儲存模型和評估結果"""
        # 儲存模型
        model_path = BASE_DIR / 'saved_models' / f'{self.model_name}.pkl'
        model_path.parent.mkdir(exist_ok=True)
        
        # 儲存模型
        joblib.dump(self.model, model_path)
        
        # 儲存標準化器和其他元數據
        metadata_path = BASE_DIR / 'saved_models' / f'{self.model_name}_metadata.pkl'
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_multiclass': self.is_multiclass,
            'best_params': self.best_params,
            'network_architecture': self.model.hidden_layer_sizes
        }
        
        if hasattr(self, 'label_encoder'):
            metadata['label_encoder'] = self.label_encoder
        
        joblib.dump(metadata, metadata_path)
        
        # 儲存評估結果
        results_path = BASE_DIR / 'model_results' / self.model_name / 'evaluation_results.csv'
        results_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 創建結果字典
        results = {
            'model_name': [self.model_name],
            'accuracy': [self.evaluation_results['accuracy']],
            'f1_macro': [self.evaluation_results['f1_macro']]
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
        if hasattr(self, 'feature_importance') and self.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in self.feature_importance.items()
            ])
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df.to_csv(BASE_DIR / 'model_results' / self.model_name / 'feature_importance.csv', index=False)
        
        # 儲存網絡參數統計
        if hasattr(self, 'weights'):
            layer_sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
            total_weights = sum(w.size for w in self.weights)
            total_biases = sum(b.size for b in self.biases)
            
            network_stats = pd.DataFrame({
                'layer': list(range(len(layer_sizes))),
                'size': layer_sizes,
                'parameters': [self.weights[i-1].size + self.biases[i-1].size if i > 0 else 0 for i in range(len(layer_sizes))]
            })
            
            network_stats.to_csv(BASE_DIR / 'model_results' / self.model_name / 'network_statistics.csv', index=False)
            
            # 儲存權重範圍
            weight_stats = []
            for i, w in enumerate(self.weights):
                weight_stats.append({
                    'layer': i+1,
                    'min_weight': np.min(w),
                    'max_weight': np.max(w),
                    'mean_weight': np.mean(w),
                    'std_weight': np.std(w)
                })
            
            pd.DataFrame(weight_stats).to_csv(BASE_DIR / 'model_results' / self.model_name / 'weight_statistics.csv', index=False)
        
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
        
        predictions_df.to_csv(BASE_DIR / 'model_results' / self.model_name / 'predictions.csv', index=False)
        
        print("\n模型和結果已儲存:")
        print(f"- 模型: {model_path}")
        print(f"- 元數據: {metadata_path}")
        print(f"- 評估結果: {results_path}")
        if hasattr(self, 'feature_importance') and self.feature_importance:
            print(f"- 特徵重要性: {BASE_DIR / 'model_results' / self.model_name / 'feature_importance.csv'}")
        print(f"- 預測結果: {BASE_DIR / 'model_results' / self.model_name / 'predictions.csv'}")
    
    def generate_report(self):
        """生成模型報告"""
        report = []
        report.append(f"=== 多層感知器 (MLP) 模型報告 ===\n")
        
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
        report.append(f"- 隱藏層配置: {self.model.hidden_layer_sizes}")
        report.append(f"- 激活函數: {self.model.activation}")
        report.append(f"- 優化器: {self.model.solver}")
        report.append(f"- L2正則化參數 (alpha): {self.model.alpha}")
        report.append(f"- 學習率: {self.model.learning_rate_init}")
        report.append(f"- 批次大小: {self.model.batch_size}")
        report.append(f"- 迭代次數: {self.model.n_iter_}")
        
        # 網絡參數統計
        if hasattr(self, 'weights'):
            layer_sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
            total_weights = sum(w.size for w in self.weights)
            total_biases = sum(b.size for b in self.biases)
            total_params = total_weights + total_biases
            
            report.append("\n網絡參數統計:")
            report.append(f"- 總參數數量: {total_params}")
            report.append(f"- 權重數量: {total_weights}")
            report.append(f"- 偏置數量: {total_biases}")
            
            report.append("\n網絡架構:")
            report.append(f"- 輸入層: {layer_sizes[0]} 個神經元")
            for i, size in enumerate(layer_sizes[1:-1], 1):
                report.append(f"- 隱藏層 {i}: {size} 個神經元")
            report.append(f"- 輸出層: {layer_sizes[-1]} 個神經元")
        
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
            report.append("\n特徵重要性 (前10，基於第一層權重):")
            sorted_features = list(self.feature_importance.items())[:10]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                report.append(f"{i}. {feature}: {importance:.4f}")
        
        # 儲存報告
        report_path = BASE_DIR / 'model_results' / self.model_name / 'model_report.txt'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n模型報告已儲存: {report_path}")
    
    def run_mlp_pipeline(self):
        """執行完整的MLP模型建立流程"""
        print("=== 開始建立多層感知器 (MLP) 模型 ===")
        
        # 創建必要目錄
        self.create_directories()
        
        # 載入資料
        if not self.load_data():
            return
        
        # 預處理
        self.preprocess_data()
        
        # 訓練模型
        self.train_model()
        
        # 評估模型
        self.evaluate_model()
        
        # 分析神經網絡
        self.analyze_neural_network()
        
        # 視覺化
        self.visualize_results()
        
        # 儲存模型和結果
        self.save_model_and_results()
        
        # 生成報告
        self.generate_report()
        
        print("\n=== 多層感知器 (MLP) 模型建立完成 ===")

if __name__ == "__main__":
    # 執行MLP模型
    mlp_model = MLPModel()
    mlp_model.run_mlp_pipeline()
