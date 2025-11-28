import pandas as pd
import feature_extraction

def extract_fe_train():
    # 1. 读取原始 fishtank.csv
    df = pd.read_csv("data/merged_train.csv")  # 路径按实际情况调整
    # 2. 特征提取
    features = feature_extraction.pandas_get_features(df)
    # 4. 保存为 fishtank_features.csv
    features.to_csv("data/train_features.csv", index=False, encoding='utf-8')
    print("已生成训练集 train_features.csv")


def extract_fe_validation():
    # 1. 读取原始 fishtank.csv
    df = pd.read_csv("data/merged_cv.csv")  # 路径按实际情况调整
    # 2. 特征提取
    features = feature_extraction.pandas_get_features(df)
    # 4. 保存为 fishtank_features.csv
    features.to_csv("data/cv_features.csv", index=False, encoding='utf-8')
    print("已生成验证集 cv_features")
    
def extract_fe_test():
    # 1. 读取原始 fishtank.csv
    df = pd.read_csv("data/merged_test.csv")  # 路径按实际情况调整
    # 2. 特征提取
    features = feature_extraction.pandas_get_features(df)
    # 4. 保存为 fishtank_features.csv
    features.to_csv("data/test_features.csv", index=False, encoding='utf-8')
    print("已生成测试集 test_features.csv")
        
if __name__ == "__main__":
    extract_fe_train()
    extract_fe_validation()
    extract_fe_test()