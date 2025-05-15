"""
prediction_01.py - 邏輯回歸模型實作
功能：使用邏輯回歸進行學生輟學預測（基準模型）
作者：Tommy
日期：2025-05-16
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.model_name = "LogisticRegression"
        self.classes = None
        self.feature_names = None
        
    def load_data(self):
        """載入最終的訓練集和測試集"""
        try:
            # 載入處理後的資料集
            train_df = pd.read_csv(WORK_DIR / 'train_final.csv')
            test_df = pd.read_csv(WORK_DIR / 'test_final.csv')
            
            # 分離特徵和目標變數
            target_col = train_df.columns[-1]
            self.X_train = train_df.drop(columns=[target_col])
            self.y_train = train_df[target_col]
            self.X_test = test_df.drop(columns=[target_col])
            self.y_test = test_df[target_col]
            
            # 保存特徵名稱和類別
            self.feature_names = self.X_train.columns.tolist()
            self.classes = np.unique(self.y_train)
            
            print(f"成功載入資料集")
            print(f"特徵數量：{len(self.feature_names)}")
            print(f"訓練集大小：{len(self.X_train)}")
            print(f"測試集大小：{len(self.X_test)}")
            print(f"目標類別：{self.classes}")
            
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
        """訓練邏輯回歸模型"""
        print("\n=== 訓練邏輯回歸模型 ===")
        
        # 設定模型參數
        self.model = LogisticRegression(
            C=1.0,                # 正則化強度的倒數
            penalty='l2',         # 使用L2正則化
            solver='lbfgs',       # 優化算法
            max_iter=1000,        # 最大迭代次數
            multi_class='multinomial',  # 多類別策略
            class_weight='balanced',    # 處理類別不平衡
            random_state=42,      # 隨機種子
            n_jobs=-1             # 使用所有CPU核心
        )
        
        # 使用標準化後的資料進行訓練
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("模型訓練完成")
    
    def evaluate_model(self):
        """評估模型性能"""
        print("\n=== 模型評估 ===")
        
        # 在測試集上進行預測
        self.predictions = self.model.predict(self.X_test_scaled)
        self.prediction_proba = self.model.predict_proba(self.X_test_scaled)
        
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
    
    def analyze_coefficients(self):
        """分析和視覺化邏輯回歸係數"""
        print("\n=== 分析模型係數 ===")
        
        # 獲取邏輯回歸係數
        if self.model.coef_.shape[0] > 1:  # 多類別
            # 為每個類別都保存係數
            coefficients = {}
            for i, cls in enumerate(self.model.classes_):
                coefficients[cls] = self.model.coef_[i]
                
            # 使用絕對值平均來獲得總體重要性
            avg_abs_coef = np.abs(self.model.coef_).mean(axis=0)
            feature_importance = dict(zip(self.feature_names, avg_abs_coef))
        else:  # 二元分類
            feature_importance = dict(zip(self.feature_names, self.model.coef_[0]))
            coefficients = {self.model.classes_[1]: self.model.coef_[0]}
        
        # 按絕對值大小排序
        self.feature_importance = {k: v for k, v in sorted(
            feature_importance.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )}
        
        # 顯示最重要的前15個特徵
        print("最重要的15個特徵（按係數絕對值排序）：")
        for i, (feature, coef) in enumerate(list(self.feature_importance.items())[:15], 1):
            print(f"{i}. {feature}: {coef:.4f}")
        
        # 儲存係數信息
        self.coefficients = coefficients
    
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
        
        # 繪製特徵重要性圖
        plt.figure(figsize=(12, 10))
        features = list(self.feature_importance.keys())[:15]  # 取前15個特徵
        importances = [abs(self.feature_importance[f]) for f in features]  # 使用絕對值
        
        bars = plt.barh(features, importances)
        
        # 根據係數正負設定不同顏色
        for i, feature in enumerate(features):
            coef_value = self.feature_importance[feature]
            if coef_value > 0:
                bars[i].set_color('blue')
            else:
                bars[i].set_color('red')
        
        plt.xlabel('係數絕對值')
        plt.title('邏輯回歸模型特徵重要性（前15個特徵）')
        plt.gca().invert_yaxis()  # 反轉Y軸，使最重要的特徵在上方
        
        # 添加圖例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='blue', label='正係數（增加輟學/畢業機率）'),
            Patch(color='red', label='負係數（降低輟學/畢業機率）')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(result_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 繪製多類別的係數熱力圖
        if len(self.classes) > 2:
            plt.figure(figsize=(14, 10))
            coef_matrix = []
            
            top_features = list(self.feature_importance.keys())[:10]
            for cls in self.model.classes_:
                cls_coefs = []
                for feat in top_features:
                    idx = self.feature_names.index(feat)
                    cls_idx = np.where(self.model.classes_ == cls)[0][0]
                    cls_coefs.append(self.model.coef_[cls_idx][idx])
                coef_matrix.append(cls_coefs)
            
            coef_df = pd.DataFrame(coef_matrix, 
                                 index=self.model.classes_,
                                 columns=top_features)
            
            sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', linewidths=0.5)
            plt.title('各類別的前10個特徵係數')
            plt.tight_layout()
            plt.savefig(result_dir / 'class_coefficients.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_model_and_results(self):
        """儲存模型和評估結果"""
        # 儲存模型
        model_path = WORK_DIR / 'saved_models' / f'{self.model_name}.pkl'
        model_path.parent.mkdir(exist_ok=True)
        
        # 儲存模型和標準化器
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes
        }
        joblib.dump(model_data, model_path)
        
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
        for cls in self.classes:
            cls_idx = np.where(self.classes == cls)[0][0]
            y_true = (self.y_test == cls).astype(int)
            y_pred = (self.predictions == cls).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            results[f'precision_{cls}'] = [precision]
            results[f'recall_{cls}'] = [recall]
        
        # 儲存為CSV
        pd.DataFrame(results).to_csv(results_path, index=False)
        
        # 儲存特徵重要性
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': [abs(self.feature_importance.get(feat, 0)) for feat in self.feature_names],
            'coefficient': [self.feature_importance.get(feat, 0) for feat in self.feature_names]
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'feature_importance.csv', index=False)
        
        # 儲存預測結果
        predictions_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions
        })
        
        # 添加各類別的預測機率
        for i, cls in enumerate(self.model.classes_):
            predictions_df[f'prob_{cls}'] = self.prediction_proba[:, i]
        
        predictions_df.to_csv(WORK_DIR / 'model_results' / self.model_name / 'predictions.csv', index=False)
        
        print("\n模型和結果已儲存:")
        print(f"- 模型: {model_path}")
        print(f"- 評估結果: {results_path}")
    
    def generate_report(self):
        """生成模型報告"""
        report = []
        report.append(f"=== 邏輯回歸模型報告 ===\n")
        
        # 基本資訊
        report.append("模型資訊:")
        report.append(f"- 模型名稱: {self.model_name}")
        report.append(f"- 訓練集大小: {len(self.X_train)}")
        report.append(f"- 測試集大小: {len(self.X_test)}")
        report.append(f"- 特徵數量: {len(self.feature_names)}")
        report.append(f"- 目標類別: {', '.join(self.classes)}")
        
        # 模型參數
        report.append("\n模型參數:")
        for param, value in self.model.get_params().items():
            report.append(f"- {param}: {value}")
        
        # 性能指標
        report.append("\n性能指標:")
        report.append(f"- 準確率: {self.evaluation_results['accuracy']:.4f}")
        report.append(f"- F1分數(macro): {self.evaluation_results['f1_macro']:.4f}")
        
        # 混淆矩陣
        report.append("\n混淆矩陣:")
        cm = self.evaluation_results['confusion_matrix']
        cm_string = []
        cm_string.append("預測 \\ 實際 | " + " | ".join(self.classes))
        cm_string.append("-" * 50)
        for i, row in enumerate(cm):
            cm_string.append(f"{self.classes[i]}     | " + " | ".join(str(cell).rjust(6) for cell in row))
        report.extend(cm_string)
        
        # 特徵重要性
        report.append("\n特徵重要性 (前15):")
        for i, (feature, coef) in enumerate(list(self.feature_importance.items())[:15], 1):
            report.append(f"{i}. {feature}: {coef:.4f}")
        
        # 儲存報告
        report_path = WORK_DIR / 'model_results' / self.model_name / 'model_report.txt'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n模型報告已儲存: {report_path}")
    
    def run_logistic_regression_pipeline(self):
        """執行完整的邏輯回歸模型建立流程"""
        print("=== 開始建立邏輯回歸模型 ===")
        
        # 載入資料
        if not self.load_data():
            return
        
        # 預處理
        self.preprocess_data()
        
        # 訓練模型
        self.train_model()
        
        # 評估模型
        self.evaluate_model()
        
        # 分析係數
        self.analyze_coefficients()
        
        # 視覺化
        self.visualize_results()
        
        # 儲存模型和結果
        self.save_model_and_results()
        
        # 生成報告
        self.generate_report()
        
        print("\n=== 邏輯回歸模型建立完成 ===")

if __name__ == "__main__":
    # 引入精確度和召回率計算
    from sklearn.metrics import precision_score, recall_score
    
    # 運行邏輯回歸模型
    lr_model = LogisticRegressionModel()
    lr_model.run_logistic_regression_pipeline()
