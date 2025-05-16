"""
prediction_02.py - XGBoost模型實作
功能：使用XGBoost進行學生輟學預測
作者：Tommy
日期：2025-05-17
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.model_name = "XGBoost"
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
        
        print("數據預處理完成：已執行標準化")
    
    def train_model(self):
        """訓練XGBoost模型"""
        print("\n=== 訓練XGBoost模型 ===")
        
        # 設定模型參數
        params = {
            'objective': 'multi:softprob' if self.is_multiclass else 'binary:logistic',
            'eval_metric': 'mlogloss' if self.is_multiclass else 'logloss',
            'eta': 0.1,                # 學習率
            'max_depth': 6,            # 樹的最大深度
            'min_child_weight': 1,     # 最小子權重，用於避免過擬合
            'subsample': 0.8,          # 樣本抽樣比例
            'colsample_bytree': 0.8,   # 特徵抽樣比例
            'gamma': 0,                # 樹節點分裂的最小損失減少
            'alpha': 0,                # L1正則化項
            'lambda': 1,               # L2正則化項
            'num_class': len(self.classes) if self.is_multiclass else 2,  # 類別數量
            'seed': 42                 # 隨機種子
        }
        
        # 轉換為XGBoost的DMatrix格式
        dtrain = xgb.DMatrix(self.X_train_scaled, label=self.y_train_encoded,
                          feature_names=self.feature_names)
        dtest = xgb.DMatrix(self.X_test_scaled, label=self.y_test_encoded,
                         feature_names=self.feature_names)
        
        # 設定評估集
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        
        # 訓練模型
        num_round = 100  # 迭代次數
        early_stopping = 10  # 提前停止的迭代次數
        
        self.model = xgb.train(params, dtrain, num_round, evallist,
                            early_stopping_rounds=early_stopping, verbose_eval=10)
        
        print(f"模型訓練完成，最佳迭代次數：{self.model.best_iteration+1}")
    
    def evaluate_model(self):
        """評估模型性能"""
        print("\n=== 模型評估 ===")
        
        # 轉換測試集為DMatrix
        dtest = xgb.DMatrix(self.X_test_scaled, feature_names=self.feature_names)
        
        # 在測試集上進行預測
        self.prediction_proba = self.model.predict(dtest)
        
        # 對於多分類問題，需要找出每個樣本的最高概率類別
        if self.is_multiclass:
            self.predictions = np.argmax(self.prediction_proba, axis=1)
            
            # 如果目標是字符串類型，需要將預測轉換回原始標籤
            if hasattr(self, 'label_encoder'):
                self.predictions = self.label_encoder.inverse_transform(self.predictions)
        else:
            # 二分類問題，使用0.5作為閾值
            self.predictions = (self.prediction_proba > 0.5).astype(int)
            
            # 如果目標是字符串類型，需要將預測轉換回原始標籤
            if hasattr(self, 'label_encoder'):
                self.predictions = self.label_encoder.inverse_transform(self.predictions)
        
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
        
        # 儲存評估結果
        self.evaluation_results = {
            'accuracy': accuracy,
            'f1_macro': f1,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(self.y_test, self.predictions),
            'predictions': self.predictions,
            'prediction_proba': self.prediction_proba
        }
    
    def analyze_feature_importance(self):
        """分析和視覺化XGBoost特徵重要性"""
        print("\n=== 分析特徵重要性 ===")
        
        # 獲取特徵重要性（基於權重）
        importance_weight = self.model.get_score(importance_type='weight')
        
        # 獲取特徵重要性（基於覆蓋度）
        importance_cover = self.model.get_score(importance_type='cover')
        
        # 獲取特徵重要性（基於增益）
        importance_gain = self.model.get_score(importance_type='gain')
        
        # 合併所有特徵重要性
        self.feature_importance = {}
        
        # 確保所有特徵都有重要性分數
        for feature in self.feature_names:
            self.feature_importance[feature] = {
                'weight': importance_weight.get(feature, 0),
                'cover': importance_cover.get(feature, 0),
                'gain': importance_gain.get(feature, 0)
            }
        
        # 按增益排序
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1]['gain'],
            reverse=True
        )
        
        # 顯示最重要的前10個特徵
        print("最重要的特徵（按增益排序）：")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i}. {feature}")
            print(f"   - 權重: {importance['weight']:.4f}")
            print(f"   - 覆蓋度: {importance['cover']:.4f}")
            print(f"   - 增益: {importance['gain']:.4f}")
    
    def visualize_results(self):
        """視覺化模型結果"""
        result_dir = WORK_DIR / 'model_results' / self.model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 混淆矩陣
        plt.figure(figsize=(10, 8))
        cm = self.evaluation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('混淆矩陣')
        plt.xlabel('預測類別')
        plt.ylabel('實際類別')
        plt.tight_layout()
        plt.savefig(result_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特徵重要性視覺化 (基於權重和增益)
        self.plot_feature_importance(result_dir)
        
        # 學習曲線 (通過XGBoost的cv功能重新計算)
        self.plot_learning_curve(result_dir)
        
        # 二分類問題的ROC曲線和PR曲線
        if not self.is_multiclass:
            self.plot_binary_curves(result_dir)
        else:
            # 多分類問題的各類別ROC曲線
            self.plot_multiclass_curves(result_dir)
    
    def plot_feature_importance(self, result_dir):
        """繪製特徵重要性圖"""
        # 按增益排序所有特徵
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1]['gain'],
            reverse=True
        )
        
        # 選取前15個特徵
        top_features = [f[0] for f in sorted_features[:15]]
        importance_gain = [f[1]['gain'] for f in sorted_features[:15]]
        importance_weight = [f[1]['weight'] for f in sorted_features[:15]]
        
        # 增益的特徵重要性圖
        plt.figure(figsize=(10, 8))
        plt.barh(top_features, importance_gain, color='darkblue')
        plt.title('XGBoost特徵重要性 (基於增益)', fontsize=14)
        plt.xlabel('增益', fontsize=12)
        plt.gca().invert_yaxis()  # 反轉Y軸，使最重要的特徵在上方
        plt.tight_layout()
        plt.savefig(result_dir / 'feature_importance_gain.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 權重的特徵重要性圖
        plt.figure(figsize=(10, 8))
        plt.barh(top_features, importance_weight, color='navy')
        plt.title('XGBoost特徵重要性 (基於權重)', fontsize=14)
        plt.xlabel('權重', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(result_dir / 'feature_importance_weight.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curve(self, result_dir):
        """重新計算並繪製學習曲線"""
        try:
            # 設定模型參數
            params = {
                'objective': 'multi:softprob' if self.is_multiclass else 'binary:logistic',
                'eval_metric': 'mlogloss' if self.is_multiclass else 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            if self.is_multiclass:
                params['num_class'] = len(self.classes)
            
            # 轉換為XGBoost的DMatrix格式
            dtrain = xgb.DMatrix(self.X_train_scaled, label=self.y_train_encoded,
                              feature_names=self.feature_names)
            
            # 執行交叉驗證
            cv_results = xgb.cv(
                params, dtrain, num_boost_round=200,
                nfold=5, stratified=True,
                early_stopping_rounds=20, verbose_eval=None,
                seed=42
            )
            
            # 繪製學習曲線
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(cv_results['train-mlogloss-mean' if self.is_multiclass else 'train-logloss-mean'],
                    label='訓練損失')
            plt.plot(cv_results['test-mlogloss-mean' if self.is_multiclass else 'test-logloss-mean'],
                    label='驗證損失')
            plt.xlabel('迭代次數')
            plt.ylabel('損失')
            plt.title('XGBoost學習曲線 (損失)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(cv_results['train-mlogloss-std' if self.is_multiclass else 'train-logloss-std'],
                    label='訓練標準差')
            plt.plot(cv_results['test-mlogloss-std' if self.is_multiclass else 'test-logloss-std'],
                    label='驗證標準差')
            plt.xlabel('迭代次數')
            plt.ylabel('標準差')
            plt.title('XGBoost學習曲線 (標準差)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(result_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 最佳迭代次數
            best_iteration = cv_results.shape[0]
            metric_name = 'test-mlogloss-mean' if self.is_multiclass else 'test-logloss-mean'
            best_score = cv_results[metric_name].iloc[-1]
            
            # 保存CV結果
            cv_summary = f"交叉驗證最佳迭代次數: {best_iteration}\n"
            cv_summary += f"交叉驗證最佳分數: {best_score:.4f}\n"
            
            with open(result_dir / 'cv_results.txt', 'w', encoding='utf-8') as f:
                f.write(cv_summary)
                f.write("\n詳細CV結果:\n")
                f.write(cv_results.to_string())
        
        except Exception as e:
            print(f"繪製學習曲線時發生錯誤: {e}")
            print("跳過學習曲線繪製")
    
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
        model_path = WORK_DIR / 'saved_models' / f'{self.model_name}.json'
        model_path.parent.mkdir(exist_ok=True)
        
        # 儲存XGBoost模型
        self.model.save_model(str(model_path))
        
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
        importance_df = pd.DataFrame([
            {
                'feature': feature,
                'weight': scores['weight'],
                'cover': scores['cover'],
                'gain': scores['gain']
            }
            for feature, scores in self.feature_importance.items()
        ])
        importance_df = importance_df.sort_values('gain', ascending=False)
        importance_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv', index=False)
        
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
        report.append(f"=== XGBoost模型報告 ===\n")
        
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
        for param, value in self.model.attributes().items():
            report.append(f"- {param}: {value}")
        
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
        report.append("\n特徵重要性 (前10，按增益排序):")
        # 按增益排序
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1]['gain'],
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            report.append(f"{i}. {feature}: 增益={importance['gain']:.4f}, 權重={importance['weight']:.4f}")
        
        # 儲存報告
        report_path = WORK_DIR / 'model_results' / self.model_name / 'model_report.txt'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n模型報告已儲存: {report_path}")
    
    def run_xgboost_pipeline(self):
        """執行完整的XGBoost模型建立流程"""
        print("=== 開始建立XGBoost模型 ===")
        
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
        
        # 視覺化
        self.visualize_results()
        
        # 儲存模型和結果
        self.save_model_and_results()
        
        # 生成報告
        self.generate_report()
        
        print("\n=== XGBoost模型建立完成 ===")

if __name__ == "__main__":
    # 運行XGBoost模型
    xgb_model = XGBoostModel()
    xgb_model.run_xgboost_pipeline()
