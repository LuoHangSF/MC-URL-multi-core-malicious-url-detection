import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from model import URLGledac
from data_preprocess import load_data, create_vocab, preprocess_data, split_data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model_path, test_set, device, batch_size=128):
    vocab = create_vocab()
    url_gledac = URLGledac(len(vocab))
    url_gledac.load_state_dict(torch.load(model_path, map_location=device))
    url_gledac.to(device)
    url_gledac.eval()
    print(device)
    
    pred = []
    y = []
    num_batches = len(test_set) // batch_size
    for it in range(num_batches):
        urls = [url[0] for url in test_set[it * batch_size:(it + 1) * batch_size]]
        labels = [url[1] for url in test_set[it * batch_size:(it + 1) * batch_size]]

        with torch.no_grad():
            urls = torch.tensor(urls).to(device)
            scores = url_gledac(urls)
            preds = torch.max(scores.cpu(), 1)[1].detach()
            pred.append(preds)
            y.append(torch.tensor(labels))

    pred = torch.cat(pred, 0).squeeze().numpy()
    y = torch.cat(y, 0).squeeze().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    accuracy = accuracy_score(y, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b', label=f'AUC = {auc:.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
if __name__ == "__main__":

    data = load_data("../data/merged_test.csv")
    vocab = create_vocab()
    examples = preprocess_data(data, vocab)
    X_folds = split_data(examples, folds=5)
    test_set = X_folds[0][1]  # 取第一个fold的test_set
    
    model_path = '../model/URLGledac/model_best.pth'  # Path to the saved model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    evaluate_model(model_path, test_set, device)