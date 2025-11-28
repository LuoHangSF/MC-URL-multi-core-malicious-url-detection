import pandas as pd
import feature_extraction

def extract_fe():
    # 1. 读取原始 fishtank.csv
    df = pd.read_csv("data/data.csv")  # 路径按实际情况调整
    # 2. 特征提取
    features = feature_extraction.pandas_get_features(df)
    # 3. 标记为恶意
    features['label'] = 1
    # 4. 保存为 fishtank_features.csv
    features.to_csv("data_features.csv", index=False, encoding='utf-8')
    print("已生成 data_features.csv")


def extract_fe2():
    # 1. 读取原始 fishtank.csv
    df = pd.read_csv("data/data.csv")  # 路径按实际情况调整
    # 2. 特征提取
    features = feature_extraction.pandas_get_features(df)
    # 4. 保存为 fishtank_features.csv
    features.to_csv("data_features.csv", index=False, encoding='utf-8')
    print("已生成 data-special_features.csv")
    
if __name__ == "__main__":
    extract_fe2()
    # # 1. 读取原始 fishtank.csv
    # df = pd.read_csv("data/test_data/fishtank.csv")  # 路径按实际情况调整
    # # 2. 特征提取
    # features = feature_extraction.pandas_get_features(df)
    # # 3. 标记为恶意
    # features['label'] = 1
    # # 4. 保存为 fishtank_features.csv
    # features.to_csv("fishtank_features.csv", index=False, encoding='utf-8')
    # print("已生成 fishtank_features.csv")