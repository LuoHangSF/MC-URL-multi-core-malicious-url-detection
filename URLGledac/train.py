import torch
import random
from sklearn.metrics import accuracy_score
from data_preprocess import load_data, create_vocab, preprocess_data, split_data
from model import URLGledac
import os

def train_model(csv_path, device, batch_size=128, epochs=10, folds=5):
    # 数据加载和预处理
    data = load_data(csv_path)
    vocab = create_vocab()
    examples = preprocess_data(data, vocab)
    X_folds = split_data(examples, folds)
    
    # 模型初始化
    url_gledac = URLGledac(len(vocab))
    url_gledac.to(device)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(url_gledac.parameters(), lr=5e-5, betas=(0.8, 0.999))
    
    # 创建模型保存目录
    model_dir = '../model/URLGledac/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"已创建模型文件夹: {model_dir}")
        
    # 训练
    for train_set, test_set in X_folds[:1]:  # 只用第一个fold
        best_val_acc = 0.0
        for epoch in range(epochs):
            train_accuracy = 0.0
            train_loss = 0.0
            random.shuffle(train_set)

            url_gledac.train()
            num_batches = len(train_set) // batch_size
            for it in range(num_batches):
                urls = [url[0] for url in train_set[it * batch_size:(it + 1) * batch_size]]
                labels = [url[1] for url in train_set[it * batch_size:(it + 1) * batch_size]]

                urls = torch.tensor(urls).to(device)
                labels = torch.tensor(labels).to(device)

                scores = url_gledac(urls)
                loss_val = loss_fn(scores, labels)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                preds = torch.max(scores.cpu(), 1)[1].detach().numpy()
                train_accuracy += accuracy_score(labels.cpu().numpy(), preds)
                train_loss += loss_val.item()

            print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy / num_batches}, Train Loss: {train_loss / num_batches}")

            # 验证
            url_gledac.eval()
            validation_accuracy = 0.0
            num_batches = len(test_set) // batch_size
            for it in range(num_batches):
                urls = [url[0] for url in test_set[it * batch_size:(it + 1) * batch_size]]
                labels = [url[1] for url in test_set[it * batch_size:(it + 1) * batch_size]]

                with torch.no_grad():
                    urls = torch.tensor(urls).to(device)
                    scores = url_gledac(urls)
                    preds = torch.max(scores.cpu(), 1)[1].detach().numpy()
                    validation_accuracy += accuracy_score(labels, preds)

            validation_accuracy /= num_batches
            print(f"Validation Accuracy: {validation_accuracy}")
            torch.save(url_gledac.state_dict(), f"../model/URLGledac/checkpoint_epoch_{epoch + 1}.pth")
            
            if validation_accuracy > best_val_acc:
                best_val_acc = validation_accuracy
                print(f"New best validation accuracy: {best_val_acc}")
                torch.save(url_gledac.state_dict(), f"../model/URLGledac/model_best.pth")
                
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "../data/merged_train.csv"
    train_model(csv_path, device)