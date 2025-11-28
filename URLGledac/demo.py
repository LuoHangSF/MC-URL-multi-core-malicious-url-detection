import torch
from model import URLGledac
from data_preprocess import create_vocab, CharVocab

# 1. 加载模型和词表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = create_vocab()
char_vocab = CharVocab(vocab)
model_path = "../model/URLGledac/model_best.pth"  # 替换为你的模型文件名

model = URLGledac(len(vocab))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. 单个URL编码
def encode_url_URLGledac(url, char_vocab, seq_len=128):
    tokens = [char_vocab.ctoi(c) for c in url][:seq_len]
    tokens += [char_vocab.ctoi('[PAD]')] * (seq_len - len(tokens))
    return torch.tensor(tokens).unsqueeze(0)  # shape: (1, seq_len)

# 3. 检测函数
def predict_url_URLGledac(url):
    input_tensor = encode_url_URLGledac(url, char_vocab).to(device)
    with torch.no_grad():
        scores = model(input_tensor)
        probs = torch.exp(scores)
        pred = torch.max(scores, 1)[1].item()
        print(probs.cpu().numpy()[0])
    return 'malicious' if pred == 1 else 'benign'

# 4. 测试
if __name__ == "__main__":
    test_url = input("Enter a URL to check: ")
    result = predict_url_URLGledac(test_url)
    print(f"URL: {test_url}\nPrediction: {result}")