"""
code is heavily borrowed from:
{
Author: Toky
Description: 基于机器学习的恶意URL检测（终端交互版，返回正确返回的可信度数值）
Website: https://wiki.y1ng.org/0x4_%E5%AE%89%E5%85%A8%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90/4xH_%E5%9F%BA%E4%BA%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%81%B6%E6%84%8FURL%E6%A3%80%E6%B5%8B/
}
"""
# import pickle
import dill as pickle
import os
import sys
import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 全局变量存储模型和向量器
model = None
vectorizer = None
test_score = None  # 存储测试分数

# 定义字符串标签到数字的映射
LABEL_MAPPING = {
    'good': 0,  # 正常URL
    'bad': 1,  # 恶意URL
    'malicious': 1,  # 恶意URL的另一种可能表示
    'benign': 0  # 正常URL的另一种可能表示
}


def csv_data_read(csv_file_path):
    """读取CSV文件中的URL和标签"""
    try:
        df_csv = pd.read_csv(csv_file_path)
        urls = []
        labels = []
        for index, row in df_csv.iterrows():
            urls.append(row["URL"])
            labels.append(row["label"])
        return urls, labels
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        sys.exit(1)


def url_tokenize(url):
    """对URL进行清洗和分词"""
    web_url = url.lower()
    dot_slash = []
    slash = str(web_url).split('/')
    for i in slash:
        r1 = str(i).split('-')
        token_slash = []
        for j in range(0, len(r1)):
            r2 = str(r1[j]).split('.')
            token_slash = token_slash + r2
        dot_slash = dot_slash + r1 + token_slash
    urltoken_list = list(set(dot_slash))
    white_words = ["com", "http:", "https:", ""]
    for white_word in white_words:
        if white_word in urltoken_list:
            urltoken_list.remove(white_word)
    return urltoken_list


def dump_model_object(file_path, model_object):
    """保存模型到本地文件"""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model_object, f)
        print(f"模型已保存至: {file_path}")
    except Exception as e:
        print(f"保存模型失败: {str(e)}")


def load_model_object(file_path):
    """从本地文件加载模型"""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None


def predict_url(url, model, vectorizer):
    """预测URL的可信度，确保返回可信度数值"""
    try:
        # 准备预测数据
        X_predict = [str(url)]
        # 转换为特征向量
        url_vector = vectorizer.transform(X_predict)
        # 进行预测
        y_predict = model.predict(url_vector)
        # 获取预测概率（两个类别的概率）
        probabilities = model.predict_proba(url_vector)[0]
        
        # ==============================================================
        proba = model.predict_proba(url_vector)
        # print(f"good概率: {proba[0]:.4f}, bad概率: {proba[1]:.4f}")
        # print(proba)
        # ==============================================================
        return proba
        
        # 处理预测结果，支持字符串和数字类型
        if isinstance(y_predict, (list, np.ndarray)):
            predict_label = y_predict[0]  # 取第一个元素
        else:
            predict_label = y_predict

        # 将预测标签转换为整数
        if isinstance(predict_label, str):
            # 处理字符串标签
            predict_label_lower = predict_label.lower()
            if predict_label_lower in LABEL_MAPPING:
                predict_class = LABEL_MAPPING[predict_label_lower]
                print(f"预测标签: {predict_label} -> 转换为类别: {predict_class}")
            else:
                print(f"未知的预测标签: {predict_label}，使用概率最高的类别")
                predict_class = np.argmax(probabilities)  # 使用概率最高的类别
        else:
            # 处理数字标签
            try:
                predict_class = int(predict_label)
            except:
                print(f"无法将预测结果转换为整数: {predict_label}，使用概率最高的类别")
                predict_class = np.argmax(probabilities)

        # 检查索引是否有效
        if 0 <= predict_class < len(probabilities):
            confidence = probabilities[predict_class]
            # 确保返回的是浮点数
            return float(confidence)
        else:
            print(f"预测类别索引无效: {predict_class}，有效范围为0到{len(probabilities) - 1}")
            # 返回最大概率作为备选方案
            return proba

    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        return None


def train_or_load_model():
    """训练模型或加载已存在的模型"""
    global test_score
    model_dir = "./model/TF-IDF"
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vector.pkl")

    # 确保模型目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 检查模型是否已存在
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        # print("加载已存在的模型...")
        model = load_model_object(model_path)
        vectorizer = load_model_object(vectorizer_path)

        if not model or not vectorizer:
            # print("模型加载失败，将重新训练...")
            return train_new_model(model_path, vectorizer_path)

        # # ====================================================================
        # # 计算模型分数
        # print("验证模型性能...")
        # grep_csv_file_path = "./data/merged_test.csv"
        # grey_urls, y = csv_data_read(grep_csv_file_path)

        # try:
        #     x = vectorizer.transform(grey_urls)
        # except:
        #     print("向量器需要重新拟合...")
        #     vectorizer.fit(grey_urls)
        #     x = vectorizer.transform(grey_urls)

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # test_score = model.score(x_test, y_test)
        # print(f"模型测试拟合分数: {test_score:.4f}")
        
        # # 模型预测
        # y_pred = model.predict(x_test)
        # y_test = [0 if l == 'good' else 1 for l in y_test]
        # y_pred = [0 if l == 'good' else 1 for l in y_pred]
        # # 计算各项指标
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)

        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1-score: {f1:.4f}")
        # # ====================================================================

        return model, vectorizer
    else:
        return train_new_model(model_path, vectorizer_path)


def train_new_model(model_path, vectorizer_path):
    """训练新模型"""
    global test_score
    print("开始训练新模型...")

    # 加载数据集
    grep_csv_file_path = "./data/merged_train.csv"
    grey_urls, y = csv_data_read(grep_csv_file_path)

    # 特征提取
    url_vectorizer = TfidfVectorizer(tokenizer=url_tokenize)
    x = url_vectorizer.fit_transform(grey_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 标签数值化=================================
    y_train = [0 if l == 'good' else 1 for l in y_train]
    y_test = [0 if l == 'good' else 1 for l in y_test]
    # 标签数值化=================================
    
    # 训练模型
    l_regress = LogisticRegression(solver='liblinear')
    l_regress.fit(x_train, y_train)
    test_score = l_regress.score(x_test, y_test)
    print(f"模型训练完成，测试拟合分数: {test_score:.4f}")

    # 保存模型
    dump_model_object(model_path, l_regress)
    dump_model_object(vectorizer_path, url_vectorizer)

    return l_regress, url_vectorizer


def get_url_confidence(url, model, vectorizer):
    """封装预测函数，提供更简洁的接口获取可信度"""
    return predict_url(url, model, vectorizer)


def main_lg():
    """主函数，处理用户输入"""
    global model, vectorizer

    # 加载或训练模型
    model, vectorizer = train_or_load_model()

    if not model or not vectorizer:
        print("无法初始化模型，程序退出")
        return

    print("\n===== URL可信度检测工具 =====")
    print("提示: 输入URL进行检测，输入'quit'退出程序")

    while True:
        try:
            url = input("\n请输入要检测的URL: ").strip()

            if url.lower() == 'quit':
                print("感谢使用，再见！")
                break

            if not url:
                print("请输入有效的URL")
                continue

            # 获取可信度（调用专门的获取函数）
            confidence = get_url_confidence(url, model, vectorizer)
            print(confidence)

            # if confidence is not None:
            #     print(f"URL: {url}")
            #     print(f"可信度: {confidence:.4f}")
            #     # 这里可以直接使用返回的confidence数值进行后续处理
            #     # 例如: if confidence < 0.89: print("低可信度，判定为恶意")
            if confidence[0][0] >= 0.88:
                result = 0
            else:
                result = 1
                
            return result

        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    main_lg()
