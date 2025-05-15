import pandas as pd
raw_data = pd.read_csv("data.csv")  # 原始數據
cleaned_data = pd.read_csv("train_cleaned.csv")  # 清洗後的數據
print(f"原始數據特徵數：{len(raw_data.columns) - 1}")
print(f"清洗後數據特徵數：{len(cleaned_data.columns) - 1}")