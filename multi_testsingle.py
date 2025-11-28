import feature_extraction
import use_sklearn
import pandas as pd
import sys
import tf_idf
from URLGledac.data_preprocess import create_vocab, CharVocab
from URLGledac.model import URLGledac
# from URLGledac.demo import encode_url_URLGledac, predict_url_URLGledac
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 加载六个机器学习模型
bayes, gbc, ada, rf, dt, lgs = use_sklearn.loadModel()

# 2. 加载TF-IDF模型和向量器
tf_idf_model, tf_idf_vectorizer = tf_idf.train_or_load_model()

# 3. 加载URLGledac模型和词表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = create_vocab()
char_vocab = CharVocab(vocab)
gledac_model_path = "./model/URLGledac/model_best.pth"  # 根据你的实际路径修改

gledac_model = URLGledac(len(vocab))
gledac_model.load_state_dict(torch.load(gledac_model_path, map_location=device))
gledac_model.to(device)
gledac_model.eval()
    
colnames = ['URL', 'label', 'URL_a', 'URL_b', 'URL_c', 'URL_d', 'URL_e', 'URL_f',
            'URL_g', 'URL_h', 'URL_i', 'URL_j', 'URL_k', 'URL_l', 'URL_m', 'URL_n',
            'URL_o', 'URL_p', 'URL_q', 'URL_r', 'URL_s', 'URL_t', 'URL_u', 'URL_v',
            'URL_w', 'URL_x', 'URL_y', 'URL_z', 'URL_depth', 'URL_len', 'exe_flag',
            'badword_n', 'popular_n', 'URL_point', 'http_flag', 'letter_ratio',
            'at_flag', 'dig_ratio', 'special_ch', 'special_ch_kind', 'TLD_id',
            'hash_token_n', 'hostname_a', 'hostname_b', 'hostname_c',
            'hostname_ch_n', 'hostname_d', 'hostname_dig_ratio', 'hostname_e',
            'hostname_entropy', 'hostname_f', 'hostname_g', 'hostname_h',
            'hostname_i', 'hostname_is_ip', 'hostname_j', 'hostname_k',
            'hostname_l', 'hostname_len', 'hostname_letter_ratio', 'hostname_m',
            'hostname_n', 'hostname_o', 'hostname_p', 'hostname_point_n',
            'hostname_q', 'hostname_r', 'hostname_s', 'hostname_std', 'hostname_t',
            'hostname_token_n', 'hostname_u', 'hostname_v', 'hostname_w',
            'hostname_x', 'hostname_y', 'hostname_z', 'pathname_ch_kind',
            'pathname_depth', 'pathname_len', 'pathname_longest_token',
            'pathname_std', 'pathname_token_n', 'search_and_n', 'search_len',
            'search_std', 'search_token_n']

# URLGledac模型的预测函数
def encode_url(url, char_vocab, seq_len=128):
    tokens = [char_vocab.ctoi(c) for c in url][:seq_len]
    tokens += [char_vocab.ctoi('[PAD]')] * (seq_len - len(tokens))
    return torch.tensor(tokens).unsqueeze(0).to(device)  # shape: (1, seq_len)

def predict_urlgledac(url, model):
    input_tensor = encode_url(url, char_vocab)
    with torch.no_grad():
        scores = model(input_tensor)
        probs = torch.exp(scores)
        pred = torch.max(scores, 1)[1].item()
    # return probs.cpu().numpy()[0], pred
    return probs.cpu().numpy(), pred

# 六个机器学习模型的预测函数
def test():
    use_sklearn.vote_to_predict_single_muti(colnames)
    
def test_string_features(url):
    # 特征提取
    features = feature_extraction.extract_url_features(url)

    # 对齐特征列
    # colnames = [ ... ]  # 这里填入你的特征名列表，建议直接复制single_test.py里的colnames
    df = pd.DataFrame([features])
    for col in colnames:
        if col not in df.columns:
            df[col] = 0
    df = df[colnames]
    label = 0
    # 调用已有的投票预测函数
    proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc = use_sklearn.vote_to_predict_single_proba(url, label, colnames)
    
    return proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc
    
def test_string_features_v2(url, bayes, gbc, ada, rf, dt, lgs):
    # 特征提取
    features = feature_extraction.extract_url_features(url)

    # 对齐特征列
    # colnames = [ ... ]  # 这里填入你的特征名列表，建议直接复制single_test.py里的colnames
    df = pd.DataFrame([features])
    for col in colnames:
        if col not in df.columns:
            df[col] = 0
    df = df[colnames]
    label = 0
    # 调用已有的投票预测函数
    
    proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc = use_sklearn.vote_to_predict_single_proba_v2(url, label, colnames, bayes, ada, rf, dt, lgs, gbc)
    
    return proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc    
       
# 平均投票函数
def mean_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac):
    # 将所有概率值存储在列表中
    probabilities = [
        proba_bayes,
        proba_ada,
        proba_rf,
        proba_dt,
        proba_lgs,
        proba_gbc,
        proba_tf_idf,
        proba_URLGledac
    ]
    
    # 计算平均概率
    mean_proba = sum(probabilities) / len(probabilities)
    
    if mean_proba >= 0.5:
        return 1, mean_proba  # 恶意URL
    else:
        return 0, mean_proba  # 良性URL

# 加权投票函数
def weighted_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac, weights):
    # 权重字典
    # weights = {"bayes": 0.7660, "ada": 0.8946, "rf": 0.9453, "dt": 0.9014, "lgs": 0.8406, "gbc": 0.9407, "tf_idf": 0.9325, "URLGledac": 0.9453}
    # weights = {"bayes": 26, "ada": 39, "rf": 39, "dt": 40, "lgs": 34, "gbc": 44, "tf_idf": 43, "URLGledac": 44}
    # weights = {"bayes": 0.10688769814690781, "ada": 0.12483255190890824, "rf": 0.13190723375753519, "dt": 0.1257814244250949, "lgs": 0.11729738780977898, "gbc": 0.13126534940835008, "tf_idf": 0.13012112078588972, "URLGledac": 0.13190723375753519}
    # weights = {"bayes": 1, "ada": 1, "rf": 1, "dt": 1, "lgs": 1, "gbc": 1, "tf_idf": 1, "URLGledac": 1}

    # good 值加权
    proba_good = [
        proba_bayes[0][0] * weights["bayes"],
        proba_ada[0][0] * weights["ada"],
        proba_rf[0][0] * weights["rf"],
        proba_dt[0][0] * weights["dt"],
        proba_lgs[0][0] * weights["lgs"],
        proba_gbc[0][0] * weights["gbc"],
        proba_tf_idf[0][0] * weights["tf_idf"],
        proba_URLGledac[0][0] * weights["URLGledac"]
    ]
    
    # bad 值加权
    prba_bad = [
        proba_bayes[0][1] * weights["bayes"],
        proba_ada[0][1] * weights["ada"],
        proba_rf[0][1] * weights["rf"],
        proba_dt[0][1] * weights["dt"],
        proba_lgs[0][1] * weights["lgs"],
        proba_gbc[0][1] * weights["gbc"],
        proba_tf_idf[0][1] * weights["tf_idf"],
        proba_URLGledac[0][1] * weights["URLGledac"]
    ]
    
    # 决策
    if sum(proba_good) > sum(prba_bad):
        return 0, sum(proba_good) / len(proba_good)
    else:
        return 1, sum(prba_bad) / len(prba_bad)

def weighted_vote_v3(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac, weights):
    # 权重字典
    # weights = {"bayes": 0.7660, "ada": 0.8946, "rf": 0.9453, "dt": 0.9014, "lgs": 0.8406, "gbc": 0.9407, "tf_idf": 0.9325, "URLGledac": 0.9453}
    # weights = {"bayes": 0.7660, "ada": 0.8946, "rf": 0.9453, "dt": 0.9014, "lgs": 0.8406, "gbc": 0.9407, "tf_idf": 0.9325, "URLGledac": 0.9453}

    # good 值加权
    proba_good = [
        proba_bayes[0][0] * weights["bayes"],
        proba_ada[0][0] * weights["ada"],
        proba_rf[0][0] * weights["rf"],
        proba_dt[0][0] * weights["dt"],
        proba_lgs[0][0] * weights["lgs"],
        proba_gbc[0][0] * weights["gbc"],
        proba_tf_idf[0][0] * weights["tf_idf"],
        proba_URLGledac[0][0] * weights["URLGledac"]
    ]
    
    # bad 值加权
    proba_bad = [
        proba_bayes[0][1] * weights["bayes"],
        proba_ada[0][1] * weights["ada"],
        proba_rf[0][1] * weights["rf"],
        proba_dt[0][1] * weights["dt"],
        proba_lgs[0][1] * weights["lgs"],
        proba_gbc[0][1] * weights["gbc"],
        proba_tf_idf[0][1] * weights["tf_idf"],
        proba_URLGledac[0][1] * weights["URLGledac"]
    ]
    
    # 决策
    if sum(proba_good) > sum(proba_bad):
        return 0, sum(proba_good) / len(proba_good)
    else:
        return 1, sum(proba_bad) / len(proba_bad)

def hard_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac):
    probabilities = [
        proba_bayes,
        proba_ada,
        proba_rf,
        proba_dt,
        proba_lgs,
        proba_gbc,
        proba_tf_idf,
        proba_URLGledac
    ]
    
    good_count = 0; bad_count = 0
    for proba in probabilities:
        if proba[0][0] > proba[0][1]:
            good_count += 1
        else:
            bad_count += 1
            
    if good_count > bad_count:
        return 0, good_count
    else:
        return 1, bad_count

def single_test_core(url):

    # 使用基于字符串特征的方法
    proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc = test_string_features(url)
    
    # 使用基于逻辑回归的处理方法
    model, vectorizer = tf_idf.train_or_load_model()
    # proba_tf_idf = tf_idf.get_url_confidence(url, model, vectorizer)
    proba_tf_idf = tf_idf.predict_url(url, model, vectorizer)
    
    # 使用基于URLGledac模型的方法

    proba_URLGledac, pred_URLGledac = predict_urlgledac(url, gledac_model)
    # print("URLGledac概率:", proba_URLGledac)
    # print("URLGledac预测类别:", "bad" if pred_URLGledac == 1 else "good")
    # print("URLGledac模型预测结果：", proba_URLGledac)
    
    print(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac)
    
    # # 2. 平均投票
    # vote, mean_proba = mean_vote(proba_bayes[0][1], proba_ada[0][1], proba_rf[0][1], proba_dt[0][1],
    #                              proba_lgs[0][1], proba_gbc[0][1], proba_tf_idf[0][1], proba_URLGledac[0][1])
    
    # # 加权投票
    vote, mean_proba = weighted_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac)
    
    return vote, mean_proba

def single_test_core_v2(url, bayes, gbc, ada, rf, dt, lgs, weights):

    # 使用基于字符串特征的方法
    proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc = test_string_features_v2(url, bayes, gbc, ada, rf, dt, lgs)
    
    # 使用基于逻辑回归的处理方法
    proba_tf_idf = tf_idf.predict_url(url, tf_idf_model, tf_idf_vectorizer)
    
    # 使用基于URLGledac模型的方法
    proba_URLGledac, pred_URLGledac = predict_urlgledac(url, gledac_model)
    
    # print(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac)
    
    # # 1. 平均投票
    # vote, mean_proba = mean_vote(proba_bayes[0][1], proba_ada[0][1], proba_rf[0][1], proba_dt[0][1],
    #                              proba_lgs[0][1], proba_gbc[0][1], proba_tf_idf[0][1], proba_URLGledac[0][1])
    
    # 2. 加权投票
    vote, mean_proba = weighted_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac, weights)
    
    # # 3. 硬投票
    # vote, mean_proba = hard_vote(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac)
    
    return vote, mean_proba

def single_test_core_v3(url, bayes, gbc, ada, rf, dt, lgs, weight):

    # 使用基于字符串特征的方法
    proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc = test_string_features_v2(url, bayes, gbc, ada, rf, dt, lgs)
    
    # 使用基于逻辑回归的处理方法
    model, vectorizer = tf_idf.train_or_load_model()
    # proba_tf_idf = tf_idf.get_url_confidence(url, model, vectorizer)
    proba_tf_idf = tf_idf.predict_url(url, model, vectorizer)
    
    # 使用基于URLGledac模型的方法

    proba_URLGledac, pred_URLGledac = predict_urlgledac(url, gledac_model)
    # print("URLGledac概率:", proba_URLGledac)
    # print("URLGledac预测类别:", "bad" if pred_URLGledac == 1 else "good")
    # print("URLGledac模型预测结果：", proba_URLGledac)
    
    # print(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac)
    
    # # 2. 平均投票
    # vote, mean_proba = mean_vote(proba_bayes[0][1], proba_ada[0][1], proba_rf[0][1], proba_dt[0][1],
    #                              proba_lgs[0][1], proba_gbc[0][1], proba_tf_idf[0][1], proba_URLGledac[0][1])
    
    # 加权投票
    vote, mean_proba = weighted_vote_v3(proba_bayes, proba_ada, proba_rf, proba_dt, proba_lgs, proba_gbc, proba_tf_idf, proba_URLGledac, weights = weight)
    
    return vote, mean_proba

def single_test01():
    # 1. 输入URL
    url = input("请输入要检测的URL：")

    vote, mean_proba = single_test_core(url)
    
    if vote == 1:
        print(f"检测结果：恶意URL，平均恶意概率为 {mean_proba:.4f}")
    elif vote == 0:
        print(f"检测结果：良性URL，平均恶意概率为 {mean_proba:.4f}")
    else:
        print("ERROR!", "vote:", vote, "mean_proba:", mean_proba)


def batch_vote_test(test_csv_path, vote_func=mean_vote):
    # 加载模型一次（全局已加载，无需重复加载）
    # test_string_features、tf_idf.train_or_load_model、predict_urlgledac 都用全局模型

    # 读取测试集
    df = pd.read_csv(test_csv_path)
    urls = df['URL'].tolist()
    labels = df['label'].tolist()
    y_true = [0 if l == 'good' else 1 for l in labels]

    y_pred = []
    
    # 权重
    weights = {"bayes": 0, "ada": 0, "rf": 0, "dt": 0, "lgs": 0, "gbc": 0, "tf_idf": 1, "URLGledac": 0}
    # weights = {"bayes": 0.10688769814690781, "ada": 0.12483255190890824, "rf": 0.13190723375753519, "dt": 0.1257814244250949, "lgs": 0.11729738780977898, "gbc": 0.13126534940835008, "tf_idf": 0.13012112078588972, "URLGledac": 0.13190723375753519}
    print("weights:", weights)
    
    for url in tqdm(urls, desc="Voting Test"):
        # 直接调用single_test_core，返回vote和mean_proba
        vote, _ = single_test_core_v2(url, bayes, gbc, ada, rf, dt, lgs, weights)
        y_pred.append(vote)

    # 评估
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    return accuracy, precision, recall, f1

def batch_vote_test_v3(test_csv_path, weight):
    # 加载模型一次（全局已加载，无需重复加载）
    # test_string_features、tf_idf.train_or_load_model、predict_urlgledac 都用全局模型

    # 读取测试集
    df = pd.read_csv(test_csv_path)
    urls = df['URL'].tolist()
    labels = df['label'].tolist()
    y_true = [0 if l == 'good' else 1 for l in labels]

    y_pred = []
    # 加载6个机器学习模型
    bayes, gbc, ada, rf, dt, lgs = use_sklearn.loadModel()
    
    for url in tqdm(urls, desc="Voting Test"):
        # 直接调用single_test_core，返回vote和mean_proba
        vote, _ = single_test_core_v3(url, bayes, gbc, ada, rf, dt, lgs, weight)
        y_pred.append(vote)

    # 评估
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    return accuracy, precision, recall, f1

# # 用法示例（建议在 __main__ 里调用）
# if __name__ == "__main__":
#     batch_vote_test("./data/merged_test_small.csv")

if __name__ == '__main__':
    # single_test01()
    batch_vote_test("./data/merged_test.csv")
    
