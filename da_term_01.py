"""
DA_term_01.py - 數據預處理與探索分析
功能：對訓練集和測試集進行數據清理、轉換和探索性分析
作者：Tommy
日期：2025-05-15
"""

import pandas as pd
import numpy as np
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

class DataPreprocessor:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.train_cleaned = None
        self.test_cleaned = None
        self.column_names = None
        
    def load_data(self):
        """載入訓練集和測試集"""
        try:
            self.train_df = pd.read_csv(WORK_DIR / 'data_train.csv', sep=';')
            self.test_df = pd.read_csv(WORK_DIR / 'data_test.csv', sep=';')
            print(f"成功載入訓練集：{len(self.train_df)} 筆")
            print(f"成功載入測試集：{len(self.test_df)} 筆")
            
            # 儲存欄位名稱
            self.column_names = self.train_df.columns.tolist()
            return True
        except Exception as e:
            print(f"載入數據時發生錯誤：{e}")
            return False
    
    def check_data_quality(self, df, dataset_name):
        """檢查數據質量"""
        print(f"\n=== {dataset_name} 數據質量檢查 ===")
        
        # 檢查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n缺失值統計：")
            print(missing_values[missing_values > 0])
        else:
            print("\n沒有缺失值")
        
        # 檢查重複值
        duplicates = df.duplicated().sum()
        print(f"\n重複記錄數：{duplicates}")
        
        # 檢查數據類型
        print("\n數據類型：")
        print(df.dtypes.value_counts())
        
        return missing_values, duplicates
    
    def explore_features(self, df, dataset_name):
        """探索特徵分布"""
        print(f"\n=== {dataset_name} 特徵探索 ===")
        
        # 數值型特徵統計
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n數值型特徵數量：{len(numeric_cols)}")
        
        # 基本統計描述
        desc_stats = df[numeric_cols].describe()
        print("\n數值型特徵統計描述：")
        print(desc_stats)
        
        # 類別型特徵
        categorical_cols = df.select_dtypes(include=['object']).columns
        # 檢查可能是類別型的數值欄位（唯一值較少）
        for col in numeric_cols:
            if df[col].nunique() < 10:
                categorical_cols = categorical_cols.union([col])
        
        print(f"\n類別型特徵數量：{len(categorical_cols)}")
        
        # 目標變數分布
        target_col = df.columns[-1]
        print(f"\n目標變數 '{target_col}' 分布：")
        print(df[target_col].value_counts())
        
        return numeric_cols, categorical_cols
    
    def visualize_distributions(self, df, numeric_cols, categorical_cols, dataset_name):
        """視覺化特徵分布"""
        # 創建圖表目錄
        viz_dir = WORK_DIR / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 數值特徵分布圖（選擇前9個）
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.ravel()
            
            for idx, col in enumerate(numeric_cols[:9]):
                axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col} 分布')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('頻率')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'{dataset_name}_numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 相關性熱力圖
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title(f'{dataset_name} 數值特徵相關性熱力圖')
            plt.savefig(viz_dir / f'{dataset_name}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 目標變數分布圖
        target_col = df.columns[-1]
        plt.figure(figsize=(8, 6))
        df[target_col].value_counts().plot(kind='bar')
        plt.title(f'{dataset_name} 目標變數分布')
        plt.xlabel(target_col)
        plt.ylabel('數量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / f'{dataset_name}_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def handle_outliers(self, df, numeric_cols):
        """處理異常值（使用IQR方法）"""
        df_clean = df.copy()
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_info[col] = len(outliers)
            
            # 將異常值限制在邊界內（capping）
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean, outlier_info
    
    def normalize_float_values(self, df, column, threshold=0.0001):
        """
        標準化浮點數值，將接近某些值的數據調整為標準值
        
        參數:
        df: DataFrame - 包含要處理的數據
        column: str - 要處理的列名
        threshold: float - 判斷兩個浮點數是否接近的閾值
        
        返回:
        DataFrame - 處理後的數據框
        """
        # 複製數據框以避免修改原始數據
        df_copy = df.copy()
        
        # 針對接近1.0的值進行標準化
        mask_near_one = abs(df_copy[column] - 1.0) < threshold
        df_copy.loc[mask_near_one, column] = 1.0
        
        # 可以添加更多的標準化規則，例如接近0.0的值
        mask_near_zero = abs(df_copy[column]) < threshold
        df_copy.loc[mask_near_zero, column] = 0.0
        
        # 四捨五入到6位小數，確保浮點數精度一致
        df_copy[column] = df_copy[column].round(6)
        
        return df_copy
    
    def feature_engineering_basic(self, df):
        """基礎特徵工程"""
        df_engineered = df.copy()
        
        # 根據文檔說明，可能需要處理的特徵
        # 例如：將某些二進制特徵轉換為更有意義的類別
        
        # 創建一些基本的衍生特徵
        # 例如：第一學期和第二學期的總體表現
        if 'Curricular units 1st sem (approved)' in df.columns:
            # 學期表現比率
            df_engineered['First_Sem_Success_Rate'] = (
                df_engineered['Curricular units 1st sem (approved)'] / 
                (df_engineered['Curricular units 1st sem (enrolled)'] + 1e-5)
            )
        
        if 'Curricular units 2nd sem (approved)' in df.columns:
            df_engineered['Second_Sem_Success_Rate'] = (
                df_engineered['Curricular units 2nd sem (approved)'] / 
                (df_engineered['Curricular units 2nd sem (enrolled)'] + 1e-5)
            )
        
        # 標準化浮點數值，確保訓練集和測試集一致
        if 'First_Sem_Success_Rate' in df_engineered.columns:
            df_engineered = self.normalize_float_values(df_engineered, 'First_Sem_Success_Rate')
        
        if 'Second_Sem_Success_Rate' in df_engineered.columns:
            df_engineered = self.normalize_float_values(df_engineered, 'Second_Sem_Success_Rate')
            
            # 記錄處理後的唯一值，用於調試
            unique_values = sorted(df_engineered['Second_Sem_Success_Rate'].unique())
            print(f"Second_Sem_Success_Rate 唯一值（處理後）: {len(unique_values)}個")
            print(unique_values)
        
        return df_engineered
    
    def save_cleaned_data(self):
        """儲存清理後的數據"""
        self.train_cleaned.to_csv(WORK_DIR / 'train_cleaned.csv', index=False)
        self.test_cleaned.to_csv(WORK_DIR / 'test_cleaned.csv', index=False)
        print("\n已儲存清理後的數據：")
        print("- train_cleaned.csv")
        print("- test_cleaned.csv")
    
    def generate_eda_report(self):
        """生成EDA報告"""
        report = []
        report.append("=== 數據預處理與探索分析報告 ===\n")
        report.append(f"訓練集大小：{len(self.train_df)} 筆")
        report.append(f"測試集大小：{len(self.test_df)} 筆")
        report.append(f"特徵數量：{len(self.column_names) - 1}")
        report.append(f"目標變數：{self.column_names[-1]}")
        
        # 添加更多統計資訊
        report.append("\n=== 數據處理摘要 ===")
        report.append(f"處理後訓練集大小：{len(self.train_cleaned)} 筆")
        report.append(f"處理後測試集大小：{len(self.test_cleaned)} 筆")
        
        # 儲存報告
        with open(WORK_DIR / 'eda_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\nEDA報告已儲存至 eda_report.txt")
    
    def run_preprocessing(self):
        """執行完整的預處理流程"""
        print("=== 開始數據預處理與探索分析 ===")
        
        # 載入數據
        if not self.load_data():
            return
        
        # 對訓練集進行處理
        print("\n處理訓練集...")
        train_missing, train_duplicates = self.check_data_quality(self.train_df, "訓練集")
        train_numeric, train_categorical = self.explore_features(self.train_df, "訓練集")
        self.visualize_distributions(self.train_df, train_numeric, train_categorical, "訓練集")
        
        # 處理異常值
        self.train_cleaned, train_outliers = self.handle_outliers(self.train_df, train_numeric)
        
        # 基礎特徵工程
        self.train_cleaned = self.feature_engineering_basic(self.train_cleaned)
        
        # 對測試集進行處理
        print("\n處理測試集...")
        test_missing, test_duplicates = self.check_data_quality(self.test_df, "測試集")
        test_numeric, test_categorical = self.explore_features(self.test_df, "測試集")
        self.visualize_distributions(self.test_df, test_numeric, test_categorical, "測試集")
        
        # 處理異常值
        self.test_cleaned, test_outliers = self.handle_outliers(self.test_df, test_numeric)
        
        # 基礎特徵工程
        self.test_cleaned = self.feature_engineering_basic(self.test_cleaned)
        
        # 儲存清理後的數據
        self.save_cleaned_data()
        
        # 生成報告
        self.generate_eda_report()
        
        print("\n=== 數據預處理與探索分析完成 ===")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_preprocessing()
