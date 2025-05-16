"""
DA_term_02.py - 模型評估框架
功能：評估比較多個機器學習模型，找出最佳模型
作者：Tommy
日期：2025-05-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定工作目錄
WORK_DIR = Path(r"c:\Tommy\Python\DA Homework\Term Project")

class ModelEvaluator:
    def __init__(self):
        # 已完成的模型列表 (不包含 prediction_03.py)
        self.model_names = [
            "LogisticRegression",
            "XGBoost",
            "RandomForest",
            "SVM",
            "MLP"
        ]
        
        # 對應的文件名
        self.model_filenames = {
            "LogisticRegression": "LogisticRegression",
            "XGBoost": "XGBoost",
            "RandomForest": "RandomForest",
            "SVM": "SVM",
            "MLP": "MLP"
        }
        
        self.model_results = {}
        self.best_model = None
        self.best_metrics = None
        self.target_classes = None
        self.is_multiclass = None
    
    def load_evaluation_results(self):
        """載入所有模型的評估結果"""
        print("=== 載入模型評估結果 ===")
        
        # 存儲所有模型的評估指標
        all_metrics = []
        
        # 逐一載入每個模型的評估結果
        for model_name in self.model_names:
            try:
                # 構建評估結果文件路徑
                result_path = WORK_DIR / "model_results" / self.model_filenames[model_name] / "evaluation_results.csv"
                
                # 讀取評估結果
                results_df = pd.read_csv(result_path)
                print(f"已載入 {model_name} 的評估結果")
                
                # 儲存模型的預測結果
                try:
                    pred_path = WORK_DIR / "model_results" / self.model_filenames[model_name] / "predictions.csv"
                    preds_df = pd.read_csv(pred_path)
                    
                    # 從預測結果判斷類別
                    if self.target_classes is None and 'actual' in preds_df.columns:
                        self.target_classes = sorted(preds_df['actual'].unique())
                        self.is_multiclass = len(self.target_classes) > 2
                        print(f"檢測到目標類別: {self.target_classes}")
                        print(f"多分類問題: {'是' if self.is_multiclass else '否'}")
                    
                    self.model_results[model_name] = {
                        'metrics': results_df.iloc[0].to_dict(),
                        'predictions': preds_df
                    }
                except Exception as e:
                    print(f"無法載入 {model_name} 的預測結果: {e}")
                    self.model_results[model_name] = {
                        'metrics': results_df.iloc[0].to_dict()
                    }
                
                # 提取關鍵指標並添加到比較列表
                metrics = results_df.iloc[0].to_dict()
                metrics['model_name'] = model_name
                all_metrics.append(metrics)
            
            except Exception as e:
                print(f"載入 {model_name} 的評估結果時發生錯誤: {e}")
        
        # 合併所有模型的指標
        self.comparison_df = pd.DataFrame(all_metrics)
        
        # 檢查是否成功載入數據
        if len(self.comparison_df) > 0:
            print(f"成功載入 {len(self.comparison_df)} 個模型的評估結果")
            return True
        else:
            print("沒有成功載入任何模型的評估結果")
            return False
    
    def find_best_model(self):
        """找出表現最佳的模型"""
        print("\n=== 尋找最佳模型 ===")
        
        if len(self.comparison_df) == 0:
            print("沒有模型可以比較")
            return None
        
        # 使用 F1 分數 (宏平均) 作為主要評估指標
        if 'f1_macro' in self.comparison_df.columns:
            # 按 F1 分數排序
            sorted_df = self.comparison_df.sort_values('f1_macro', ascending=False)
            self.best_model = sorted_df.iloc[0]['model_name']
            self.best_metrics = sorted_df.iloc[0].to_dict()
            
            print(f"依據 F1 分數 (宏平均)，最佳模型是: {self.best_model}")
            print(f"F1 分數 (宏平均): {self.best_metrics['f1_macro']:.4f}")
            print(f"準確率: {self.best_metrics['accuracy']:.4f}")
            
            return self.best_model
        
        # 如果沒有 F1 分數，則使用準確率
        elif 'accuracy' in self.comparison_df.columns:
            # 按準確率排序
            sorted_df = self.comparison_df.sort_values('accuracy', ascending=False)
            self.best_model = sorted_df.iloc[0]['model_name']
            self.best_metrics = sorted_df.iloc[0].to_dict()
            
            print(f"依據準確率，最佳模型是: {self.best_model}")
            print(f"準確率: {self.best_metrics['accuracy']:.4f}")
            
            return self.best_model
        
        else:
            print("找不到可用於比較的評估指標")
            return None
    
    def compare_models(self):
        """比較不同模型的性能"""
        print("\n=== 模型性能比較 ===")
        
        # 顯示所有模型的主要指標
        print("\n主要評估指標比較:")
        
        # 選擇要顯示的指標
        display_columns = ['model_name', 'accuracy', 'f1_macro']
        
        # 添加類別特定的指標（如果是多分類）
        if self.is_multiclass and self.target_classes is not None:
            for cls in self.target_classes:
                if f'f1_{cls}' in self.comparison_df.columns:
                    display_columns.append(f'f1_{cls}')
        
        # 按 F1 分數排序
        if 'f1_macro' in self.comparison_df.columns:
            sorted_df = self.comparison_df.sort_values('f1_macro', ascending=False)
        else:
            sorted_df = self.comparison_df.sort_values('accuracy', ascending=False)
        
        # 顯示排序後的指標
        display_df = sorted_df[display_columns].copy()
        
        # 將數值列格式化為小數點後四位
        for col in display_df.columns:
            if col != 'model_name' and display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        # 模型之間的性能差異
        if len(sorted_df) > 1:
            best_f1 = float(sorted_df.iloc[0]['f1_macro'])
            second_f1 = float(sorted_df.iloc[1]['f1_macro'])
            
            print(f"\n最佳模型 ({sorted_df.iloc[0]['model_name']}) 的 F1 分數比第二名 ({sorted_df.iloc[1]['model_name']}) 高 {(best_f1-second_f1)/second_f1*100:.2f}%")
        
        # 保存比較結果
        result_dir = WORK_DIR / "model_comparison"
        result_dir.mkdir(exist_ok=True)
        self.comparison_df.to_csv(result_dir / "model_metrics_comparison.csv", index=False)
        print(f"\n比較結果已保存至 {result_dir / 'model_metrics_comparison.csv'}")
    
    def visualize_comparison(self):
        """可視化不同模型的比較結果"""
        print("\n=== 生成比較可視化 ===")
        
        # 創建結果目錄
        result_dir = WORK_DIR / "model_comparison"
        result_dir.mkdir(exist_ok=True)
        
        # 確保有數據可視化
        if len(self.comparison_df) == 0:
            print("沒有數據可視化")
            return
        
        # 1. 模型性能條形圖
        self.plot_performance_bars(result_dir)
        
        # 2. 性能雷達圖（多指標比較）
        self.plot_radar_chart(result_dir)
        
        # 3. 混淆矩陣比較（如果有預測結果）
        self.plot_confusion_matrices(result_dir)
        
        # 4. ROC 曲線比較（如果有預測概率）
        self.plot_roc_comparison(result_dir)
        
        print(f"所有可視化圖表已保存至 {result_dir}")
    
    def plot_performance_bars(self, result_dir):
        """繪製模型性能條形圖"""
        try:
            # 準備數據
            models = self.comparison_df['model_name'].tolist()
            
            # 主要指標比較
            if 'f1_macro' in self.comparison_df.columns and 'accuracy' in self.comparison_df.columns:
                f1_scores = self.comparison_df['f1_macro'].astype(float).tolist()
                accuracies = self.comparison_df['accuracy'].astype(float).tolist()
                
                # 按 F1 分數排序
                sorted_indices = np.argsort(f1_scores)[::-1]
                models = [models[i] for i in sorted_indices]
                f1_scores = [f1_scores[i] for i in sorted_indices]
                accuracies = [accuracies[i] for i in sorted_indices]
                
                # 繪製條形圖
                plt.figure(figsize=(12, 6))
                
                x = np.arange(len(models))
                width = 0.35
                
                plt.bar(x - width/2, f1_scores, width, label='F1 分數 (宏平均)', color='royalblue')
                plt.bar(x + width/2, accuracies, width, label='準確率', color='forestgreen')
                
                plt.xlabel('模型', fontsize=12)
                plt.ylabel('分數', fontsize=12)
                plt.title('模型性能比較 (F1 分數和準確率)', fontsize=14)
                plt.xticks(x, models, rotation=45, ha='right')
                plt.ylim(0, 1.1)
                
                # 添加數值標籤
                for i, v in enumerate(f1_scores):
                    plt.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
                
                for i, v in enumerate(accuracies):
                    plt.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(result_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("已生成模型性能條形圖")
            
            # 如果是多分類問題，還可以繪製每個類別的 F1 分數
            if self.is_multiclass and self.target_classes is not None:
                class_metrics = {}
                
                # 收集每個類別的 F1 分數
                for cls in self.target_classes:
                    col_name = f'f1_{cls}'
                    if col_name in self.comparison_df.columns:
                        class_metrics[cls] = self.comparison_df[col_name].astype(float).tolist()
                
                if class_metrics:
                    plt.figure(figsize=(14, 7))
                    
                    x = np.arange(len(models))
                    width = 0.8 / len(class_metrics)
                    
                    # 每個類別一個條
                    for i, (cls, scores) in enumerate(class_metrics.items()):
                        offset = (i - len(class_metrics)/2 + 0.5) * width
                        bars = plt.bar(x + offset, [scores[j] for j in sorted_indices], width, 
                                      label=f'類別 {cls}')
                    
                    plt.xlabel('模型', fontsize=12)
                    plt.ylabel('F1 分數', fontsize=12)
                    plt.title('各模型在不同類別上的 F1 分數', fontsize=14)
                    plt.xticks(x, models, rotation=45, ha='right')
                    plt.ylim(0, 1.1)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(result_dir / "model_class_f1_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print("已生成各類別 F1 分數比較圖")
        
        except Exception as e:
            print(f"繪製性能條形圖時發生錯誤: {e}")
    
    def plot_radar_chart(self, result_dir):
        """繪製性能雷達圖"""
        try:
            # 確保有足夠的指標和模型
            if len(self.comparison_df) < 2:
                print("模型數量不足，無法繪製雷達圖")
                return
            
            # 選擇要顯示的指標
            metrics = ['accuracy', 'f1_macro']
            
            # 添加每個類別的精確率和召回率
            if self.is_multiclass and self.target_classes is not None:
                for cls in self.target_classes:
                    prec_col = f'precision_{cls}'
                    rec_col = f'recall_{cls}'
                    if prec_col in self.comparison_df.columns:
                        metrics.append(prec_col)
                    if rec_col in self.comparison_df.columns:
                        metrics.append(rec_col)
            
            # 選擇顯示的模型（最多5個）
            if 'f1_macro' in self.comparison_df.columns:
                top_models = self.comparison_df.sort_values('f1_macro', ascending=False).head(5)
            else:
                top_models = self.comparison_df.sort_values('accuracy', ascending=False).head(5)
            
            # 準備雷達圖數據
            metrics_data = []
            for _, row in top_models.iterrows():
                model_data = [row['model_name']]
                for metric in metrics:
                    if metric in row and pd.notna(row[metric]):
                        model_data.append(float(row[metric]))
                    else:
                        model_data.append(0)
                metrics_data.append(model_data)
            
            # 設置雷達圖
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 閉合雷達圖
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # 添加每個模型的數據
            for model_data in metrics_data:
                values = model_data[1:]
                values += values[:1]  # 閉合雷達圖
                ax.plot(angles, values, 'o-', linewidth=2, label=model_data[0])
                ax.fill(angles, values, alpha=0.1)
            
            # 設置雷達圖標籤
            metric_labels = [m.replace('_', ' ').title() for m in metrics]
            ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
            
            # 設置雷達圖刻度
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            
            # 添加圖例和標題
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('模型性能雷達圖', fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(result_dir / "model_radar_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("已生成模型性能雷達圖")
        
        except Exception as e:
            print(f"繪製雷達圖時發生錯誤: {e}")
    
    def plot_confusion_matrices(self, result_dir):
        """繪製各模型的混淆矩陣比較"""
        try:
            # 檢查是否所有模型都有預測結果
            models_with_predictions = [model for model in self.model_names 
                                     if model in self.model_results and 
                                     'predictions' in self.model_results[model]]
            
            if not models_with_predictions:
                print("沒有可用的預測結果來繪製混淆矩陣")
                return
            
            # 選擇頂部的模型進行比較（最多4個）
            if 'f1_macro' in self.comparison_df.columns:
                top_models = self.comparison_df.sort_values('f1_macro', ascending=False).head(4)['model_name'].tolist()
            else:
                top_models = self.comparison_df.sort_values('accuracy', ascending=False).head(4)['model_name'].tolist()
            
            # 篩選出有預測結果的頂部模型
            top_models_with_predictions = [model for model in top_models if model in models_with_predictions]
            
            if not top_models_with_predictions:
                print("頂部模型沒有可用的預測結果")
                return
            
            # 計算每個模型的混淆矩陣
            num_models = len(top_models_with_predictions)
            num_cols = min(2, num_models)
            num_rows = (num_models + num_cols - 1) // num_cols
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10 * num_rows / 2))
            if num_models == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, model_name in enumerate(top_models_with_predictions):
                if i >= len(axes):
                    break
                    
                predictions_df = self.model_results[model_name]['predictions']
                if 'actual' in predictions_df.columns and 'predicted' in predictions_df.columns:
                    cm = confusion_matrix(predictions_df['actual'], predictions_df['predicted'])
                    
                    # 繪製混淆矩陣
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                              xticklabels=self.target_classes if self.target_classes is not None else 'auto',
                              yticklabels=self.target_classes if self.target_classes is not None else 'auto')
                    
                    axes[i].set_title(f'{model_name} 混淆矩陣')
                    axes[i].set_xlabel('預測類別')
                    axes[i].set_ylabel('實際類別')
            
            # 隱藏多餘的子圖
            for i in range(num_models, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(result_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("已生成混淆矩陣比較圖")
        
        except Exception as e:
            print(f"繪製混淆矩陣比較時發生錯誤: {e}")
    
    def plot_roc_comparison(self, result_dir):
        """繪製ROC曲線比較（針對二分類問題）"""
        try:
            # 只對二分類問題進行ROC比較
            if self.is_multiclass or not self.target_classes or len(self.target_classes) != 2:
                print("不是二分類問題，跳過ROC曲線比較")
                return
            
            # 檢查哪些模型有預測概率
            models_with_proba = []
            
            for model_name in self.model_names:
                if (model_name in self.model_results and 
                    'predictions' in self.model_results[model_name]):
                    
                    predictions_df = self.model_results[model_name]['predictions']
                    
                    # 檢查是否有概率列
                    if 'probability' in predictions_df.columns or 'prob_1' in predictions_df.columns:
                        models_with_proba.append(model_name)
            
            if not models_with_proba:
                print("沒有模型包含預測概率，跳過ROC曲線比較")
                return
            
            # 繪製ROC曲線比較圖
            plt.figure(figsize=(10, 8))
            
            for model_name in models_with_proba:
                predictions_df = self.model_results[model_name]['predictions']
                
                # 獲取預測概率
                if 'probability' in predictions_df.columns:
                    y_score = predictions_df['probability']
                elif 'prob_1' in predictions_df.columns:
                    y_score = predictions_df['prob_1']
                else:
                    continue
                
                # 計算ROC曲線
                y_true = predictions_df['actual'].astype(int)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                # 繪製ROC曲線
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # 添加隨機猜測基準線
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('各模型ROC曲線比較', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(result_dir / "roc_curves_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("已生成ROC曲線比較圖")
        
        except Exception as e:
            print(f"繪製ROC曲線比較時發生錯誤: {e}")
    
    def analyze_features(self):
        """分析特徵重要性排名"""
        print("\n=== 分析特徵重要性 ===")
        
        feature_importance_data = {}
        
        # 收集各模型的特徵重要性
        for model_name in self.model_names:
            try:
                importance_path = WORK_DIR / "model_results" / self.model_filenames[model_name] / "feature_importance.csv"
                if importance_path.exists():
                    importance_df = pd.read_csv(importance_path)
                    
                    # 確保有特徵和重要性列
                    if 'feature' in importance_df.columns:
                        # 找出重要性列（可能有不同的命名）
                        importance_col = None
                        for col in ['importance', 'coefficient', 'gain', 'weight']:
                            if col in importance_df.columns:
                                importance_col = col
                                break
                        
                        if importance_col:
                            # 排序特徵
                            importance_df = importance_df.sort_values(importance_col, ascending=False)
                            
                            # 保存特徵重要性
                            feature_importance_data[model_name] = importance_df
                            print(f"已載入 {model_name} 的特徵重要性: {len(importance_df)} 個特徵")
            except Exception as e:
                print(f"載入 {model_name} 的特徵重要性時發生錯誤: {e}")
        
        if not feature_importance_data:
            print("沒有找到任何模型的特徵重要性資料")
            return
        
        # 分析最佳模型的特徵重要性
        if self.best_model and self.best_model in feature_importance_data:
            best_model_importance = feature_importance_data[self.best_model]
            
            # 顯示最佳模型的前10個重要特徵
            importance_col = [col for col in best_model_importance.columns 
                             if col in ['importance', 'coefficient', 'gain', 'weight']][0]
            
            print(f"\n最佳模型 ({self.best_model}) 的前10個重要特徵:")
            top_features = best_model_importance.head(10)
            print(top_features[['feature', importance_col]].to_string(index=False))
            
            # 保存最佳模型的特徵重要性
            result_dir = WORK_DIR / "model_comparison"
            best_model_importance.to_csv(result_dir / "best_model_feature_importance.csv", index=False)
            
            # 繪製最佳模型的特徵重要性圖
            self.plot_feature_importance(result_dir, best_model_importance, self.best_model)
            
            print(f"最佳模型的特徵重要性已保存至 {result_dir / 'best_model_feature_importance.csv'}")
        
        # 比較不同模型的特徵重要性排名
        self.compare_feature_rankings(feature_importance_data)
    
    def plot_feature_importance(self, result_dir, importance_df, model_name):
        """繪製特徵重要性圖"""
        try:
            # 找出重要性列
            importance_col = [col for col in importance_df.columns 
                             if col in ['importance', 'coefficient', 'gain', 'weight']][0]
            
            # 選取前15個特徵
            top_features = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            
            # 判斷係數正負（如果是係數的話）
            if importance_col == 'coefficient':
                colors = ['blue' if c >= 0 else 'red' for c in top_features[importance_col]]
                plt.barh(top_features['feature'], top_features[importance_col].abs(), color=colors)
                
                # 添加圖例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(color='blue', label='正係數（正相關）'),
                    Patch(color='red', label='負係數（負相關）')
                ]
                plt.legend(handles=legend_elements, loc='lower right')
            else:
                plt.barh(top_features['feature'], top_features[importance_col], color='darkblue')
            
            plt.title(f'{model_name} 特徵重要性', fontsize=14)
            plt.xlabel(importance_col.title(), fontsize=12)
            plt.gca().invert_yaxis()  # 讓最重要的特徵在上方
            plt.tight_layout()
            
            plt.savefig(result_dir / f"{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已生成 {model_name} 的特徵重要性圖")
        except Exception as e:
            print(f"繪製特徵重要性圖時發生錯誤: {e}")
    
    def compare_feature_rankings(self, feature_importance_data):
        """比較不同模型的特徵重要性排名"""
        print("\n=== 比較不同模型的特徵重要性排名 ===")
        
        # 至少需要兩個模型有特徵重要性數據
        if len(feature_importance_data) < 2:
            print("模型數量不足，無法進行特徵重要性排名比較")
            return
        
        # 獲取所有模型的前10特徵
        top_features = {}
        for model_name, importance_df in feature_importance_data.items():
            # 獲取特徵列
            feature_col = 'feature'
            
            # 找出重要性列
            importance_col = [col for col in importance_df.columns 
                             if col in ['importance', 'coefficient', 'gain', 'weight']][0]
            
            # 排序並獲取前10
            top10 = importance_df.sort_values(importance_col, ascending=False).head(10)[feature_col].tolist()
            top_features[model_name] = top10
        
        # 找出在多個模型中共同出現的特徵
        all_top_features = []
        for features in top_features.values():
            all_top_features.extend(features)
        
        # 計算每個特徵出現的次數
        feature_counts = pd.Series(all_top_features).value_counts()
        common_features = feature_counts[feature_counts > 1].index.tolist()
        
        if not common_features:
            print("沒有在多個模型中共同出現的重要特徵")
            return
        
        print(f"在多個模型中共同出現的重要特徵: {len(common_features)}")
        
        # 創建特徵排名比較表
        feature_rankings = pd.DataFrame(index=common_features)
        
        for model_name, importance_df in feature_importance_data.items():
            # 獲取所有特徵的排名
            feature_col = 'feature'
            
            # 找出重要性列
            importance_col = [col for col in importance_df.columns 
                             if col in ['importance', 'coefficient', 'gain', 'weight']][0]
            
            # 獲取排名
            importance_df['rank'] = importance_df[importance_col].rank(ascending=False)
            # 檢查是否有重複的特徵名稱
            if importance_df[feature_col].duplicated().any():
                print(f"警告: {model_name} 模型中存在重複的特徵名稱，將保留排名較高的")
                # 為每個特徵保留排名最高的記錄
                importance_df = importance_df.sort_values('rank').drop_duplicates(subset=[feature_col], keep='first')

            # 將排名轉換為字典，以便後續賦值
            rankings_dict = dict(zip(importance_df[feature_col], importance_df['rank']))

            # 對每個共同特徵賦值
            for feature in feature_rankings.index:
                if feature in rankings_dict:
                    feature_rankings.loc[feature, model_name] = rankings_dict[feature]
                else:
                    # 對於不在該模型中的特徵，可以將其設置為 NaN
                    feature_rankings.loc[feature, model_name] = np.nan

        
        # 只保留共同特徵
        #feature_rankings = feature_rankings.loc[common_features]
        
        # 計算平均排名並排序
        feature_rankings['平均排名'] = feature_rankings.mean(axis=1)
        feature_rankings = feature_rankings.sort_values('平均排名')
        
        print("\n共同重要特徵的排名比較 (數字越小表示重要性越高):")
        print(feature_rankings.to_string())
        
        # 保存共同特徵排名
        result_dir = WORK_DIR / "model_comparison"
        feature_rankings.to_csv(result_dir / "common_feature_rankings.csv")
        
        # 視覺化共同特徵在不同模型中的排名
        self.visualize_feature_rankings(result_dir, feature_rankings)
        
        print(f"共同特徵排名已保存至 {result_dir / 'common_feature_rankings.csv'}")
    
    def visualize_feature_rankings(self, result_dir, feature_rankings):
        """視覺化特徵排名比較"""
        try:
            # 只顯示前10個共同特徵
            top_common = feature_rankings.head(10)
            
            # 轉置數據以便繪圖
            plot_data = top_common.drop(columns=['平均排名']).transpose()
            
            plt.figure(figsize=(14, 8))
            
            # 使用不同的標記和顏色
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            colors = plt.cm.tab10.colors
            
            # 繪製每個模型的排名
            for i, feature in enumerate(plot_data.columns):
                plt.plot(plot_data.index, plot_data[feature], 
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       label=feature)
            
            plt.title('各模型中共同重要特徵的排名比較', fontsize=14)
            plt.xlabel('模型', fontsize=12)
            plt.ylabel('排名 (數字越小表示重要性越高)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='特徵', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # 反轉Y軸，使排名靠前的特徵在上方
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(result_dir / "feature_ranking_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("已生成特徵排名比較圖")
        except Exception as e:
            print(f"視覺化特徵排名時發生錯誤: {e}")
    
    def generate_final_report(self):
        """生成最終比較報告"""
        print("\n=== 生成最終比較報告 ===")
        
        report = []
        report.append("=== 機器學習模型比較報告 ===\n")
        
        # 添加比較的模型
        report.append("比較的模型:")
        for model_name in self.model_names:
            if model_name in self.model_results:
                report.append(f"- {model_name}")
        
        # 添加問題類型
        report.append(f"\n問題類型: {'多分類' if self.is_multiclass else '二分類'}")
        if self.target_classes is not None:
            report.append(f"目標類別: {', '.join(map(str, self.target_classes))}")
        
        # 添加最佳模型
        if self.best_model:
            report.append(f"\n最佳模型: {self.best_model}")
            report.append(f"評估指標:")
            
            if self.best_metrics:
                for metric, value in self.best_metrics.items():
                    if metric != 'model_name' and not pd.isna(value):
                        try:
                            value_float = float(value)
                            report.append(f"- {metric}: {value_float:.4f}")
                        except:
                            report.append(f"- {metric}: {value}")
        
        # 添加比較結果
        if len(self.comparison_df) > 0:
            report.append("\n模型比較結果:")
            
            # 選擇要顯示的指標
            display_columns = ['model_name', 'accuracy', 'f1_macro']
            
            # 添加類別特定的指標（如果是多分類）
            if self.is_multiclass and self.target_classes is not None:
                for cls in self.target_classes:
                    if f'f1_{cls}' in self.comparison_df.columns:
                        display_columns.append(f'f1_{cls}')
            
            # 按 F1 分數排序
            if 'f1_macro' in self.comparison_df.columns:
                sorted_df = self.comparison_df.sort_values('f1_macro', ascending=False)
            else:
                sorted_df = self.comparison_df.sort_values('accuracy', ascending=False)
            
            # 格式化數據
            formatted_df = sorted_df[display_columns].copy()
            for col in formatted_df.columns:
                if col != 'model_name' and formatted_df[col].dtype in ['float64', 'float32']:
                    formatted_df[col] = formatted_df[col].map(lambda x: f"{float(x):.4f}")
            
            # 添加到報告
            report.append(formatted_df.to_string(index=False))
        
        # 添加特徵分析
        if self.best_model and 'best_model_feature_importance.csv' in [f.name for f in (WORK_DIR / "model_comparison").glob('*.csv')]:
            try:
                importance_df = pd.read_csv(WORK_DIR / "model_comparison" / "best_model_feature_importance.csv")
                
                # 找出重要性列
                importance_col = [col for col in importance_df.columns 
                                if col in ['importance', 'coefficient', 'gain', 'weight']][0]
                
                report.append(f"\n最佳模型 ({self.best_model}) 的前10個重要特徵:")
                top_features = importance_df.head(10)[['feature', importance_col]]
                
                # 格式化浮點數
                top_features[importance_col] = top_features[importance_col].map(lambda x: f"{float(x):.4f}")
                
                report.append(top_features.to_string(index=False))
            except Exception as e:
                print(f"添加特徵重要性到報告時發生錯誤: {e}")
        
        # 添加共同特徵分析
        if 'common_feature_rankings.csv' in [f.name for f in (WORK_DIR / "model_comparison").glob('*.csv')]:
            try:
                rankings_df = pd.read_csv(WORK_DIR / "model_comparison" / "common_feature_rankings.csv")
                
                report.append("\n多個模型共同重要的特徵 (按平均排名):")
                report.append(rankings_df.head(5).to_string())
            except Exception as e:
                print(f"添加共同特徵分析到報告時發生錯誤: {e}")
        
        # 添加總結
        report.append("\n總結:")
        if self.best_model:
            report.append(f"1. {self.best_model} 在本次比較中表現最佳")
            
            # 如果有多個模型，添加性能差異
            if len(self.comparison_df) > 1:
                sorted_df = self.comparison_df.sort_values('f1_macro', ascending=False)
                best_f1 = float(sorted_df.iloc[0]['f1_macro'])
                second_f1 = float(sorted_df.iloc[1]['f1_macro'])
                
                report.append(f"2. 最佳模型的F1分數比第二名 ({sorted_df.iloc[1]['model_name']}) 高 {(best_f1-second_f1)/second_f1*100:.2f}%")
        
        # 添加建議
        report.append("\n建議:")
        report.append("1. 在生產環境中使用最佳模型")
        report.append("2. 考慮進一步優化最佳模型的超參數")
        report.append("3. 探索模型集成的可能性，結合多個模型的優勢")
        
        # 保存報告
        report_path = WORK_DIR / "model_comparison" / "final_comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"最終比較報告已保存至 {report_path}")
        
        # 同時保存為 Markdown 格式
        md_report = []
        md_report.append("# 機器學習模型比較報告\n")
        
        # 添加比較的模型
        md_report.append("## 比較的模型")
        for model_name in self.model_names:
            if model_name in self.model_results:
                md_report.append(f"- {model_name}")
        
        # 添加問題類型
        md_report.append(f"\n## 問題類型")
        md_report.append(f"- {'多分類問題' if self.is_multiclass else '二分類問題'}")
        if self.target_classes is not None:
            md_report.append(f"- 目標類別: {', '.join(map(str, self.target_classes))}")
        
        # 添加最佳模型
        if self.best_model:
            md_report.append(f"\n## 最佳模型")
            md_report.append(f"**{self.best_model}**")
            md_report.append(f"\n### 評估指標")
            
            if self.best_metrics:
                for metric, value in self.best_metrics.items():
                    if metric != 'model_name' and not pd.isna(value):
                        try:
                            value_float = float(value)
                            md_report.append(f"- **{metric}**: {value_float:.4f}")
                        except:
                            md_report.append(f"- **{metric}**: {value}")
        
        # 添加比較結果
        if len(self.comparison_df) > 0:
            md_report.append("\n## 模型比較結果")
            
            # 格式化為 Markdown 表格
            md_table = formatted_df.to_markdown(index=False)
            md_report.append(md_table)
        
        # 添加特徵重要性
        if self.best_model and 'best_model_feature_importance.csv' in [f.name for f in (WORK_DIR / "model_comparison").glob('*.csv')]:
            md_report.append(f"\n## 最佳模型的特徵重要性")
            
            # 格式化為 Markdown 表格
            try:
                top_features_md = top_features.to_markdown(index=False)
                md_report.append(top_features_md)
            except:
                md_report.append("*無法生成特徵重要性表格*")
        
        # 添加圖片引用
        md_report.append("\n## 可視化圖表")
        
        images = [
            ("model_performance_comparison.png", "模型性能比較"),
            ("model_radar_chart.png", "模型性能雷達圖"),
            ("confusion_matrices_comparison.png", "混淆矩陣比較"),
            ("roc_curves_comparison.png", "ROC曲線比較"),
            (f"{self.best_model}_feature_importance.png", "最佳模型特徵重要性"),
            ("feature_ranking_comparison.png", "特徵排名比較")
        ]
        
        for img_file, img_title in images:
            if (WORK_DIR / "model_comparison" / img_file).exists():
                md_report.append(f"\n### {img_title}")
                md_report.append(f"![{img_title}]({img_file})")
        
        # 添加總結
        md_report.append("\n## 總結")
        if self.best_model:
            md_report.append(f"1. **{self.best_model}** 在本次比較中表現最佳")
            
            # 如果有多個模型，添加性能差異
            if len(self.comparison_df) > 1:
                sorted_df = self.comparison_df.sort_values('f1_macro', ascending=False)
                best_f1 = float(sorted_df.iloc[0]['f1_macro'])
                second_f1 = float(sorted_df.iloc[1]['f1_macro'])
                
                md_report.append(f"2. 最佳模型的F1分數比第二名 (**{sorted_df.iloc[1]['model_name']}**) 高 **{(best_f1-second_f1)/second_f1*100:.2f}%**")
        
        # 添加建議
        md_report.append("\n## 建議")
        md_report.append("1. 在生產環境中使用最佳模型")
        md_report.append("2. 考慮進一步優化最佳模型的超參數")
        md_report.append("3. 探索模型集成的可能性，結合多個模型的優勢")
        
        # 保存 Markdown 報告
        md_report_path = WORK_DIR / "model_comparison" / "final_comparison_report.md"
        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_report))
        
        print(f"Markdown 格式報告已保存至 {md_report_path}")
    
    def run_evaluation(self):
        """執行完整的模型評估流程"""
        print("=== 開始模型評估與比較 ===")
        
        # 載入評估結果
        if not self.load_evaluation_results():
            print("無法載入評估結果，退出評估")
            return
        
        # 找出最佳模型
        self.find_best_model()
        
        # 比較模型性能
        self.compare_models()
        
        # 視覺化比較結果
        self.visualize_comparison()
        
        # 分析特徵重要性
        self.analyze_features()
        
        # 生成最終比較報告
        self.generate_final_report()
        
        print("\n=== 模型評估與比較完成 ===")
        
        # 顯示最終結論
        if self.best_model:
            print(f"\n最終結論: {self.best_model} 是表現最佳的模型")
            print(f"F1分數 (宏平均): {self.best_metrics.get('f1_macro', '未知')}")
            print(f"準確率: {self.best_metrics.get('accuracy', '未知')}")
            print(f"\n詳細報告已保存至 {WORK_DIR / 'model_comparison' / 'final_comparison_report.txt'}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
