import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tf_idf

def evaluate_tfidf_model(model, vectorizer, test_csv_path):
    print("1")
    # 读取测试集
    df = pd.read_csv(test_csv_path)
    urls = df['URL'].tolist()
    labels = df['label'].tolist()
    # 标签转换为数字
    y_true = [0 if l == 'good' else 1 for l in labels]
    print("1.1")
    # 特征向量化
    print("测试集URL样本数：", len(urls))
    print("前5个URL：", urls[:5])
    print("URL类型：", [type(u) for u in urls[:5]])
    print("是否有空值：", any(pd.isnull(urls)))
    # X_test = vectorizer.transform(urls)
    try:
        X_test = vectorizer.transform(urls)
        print("1.2")
    except Exception as e:
        print("vectorizer.transform 出错：", str(e))
        import traceback
        traceback.print_exc()
    print("1.2")
    # 预测
    y_pred = model.predict(X_test)
    # =====================================
    y_pred = model.predict(X_test)
    y_pred = [0 if l == 'good' else 1 for l in y_pred]
    # ====================================
    
    print("2")
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

if __name__ == "__main__":
    # 加载模型和向量器
    model, vectorizer = tf_idf.train_or_load_model()
    print("load done!")
    # 测试集路径
    test_csv_path = "data/merged_test.csv"
    print("test load done! testing...")
    # 评估
    evaluate_tfidf_model(model, vectorizer, test_csv_path)
    print("evaluation done!")