# MC-URL:基于多分类器加权投票的恶意URL检测模型

## 1. 准备工作

### 1.1环境相关

1. 环境安装

   ```shell
   cd MC-URL
   conda env create -f MC-URL2_environment.yml MC-URL
   ```

   **请注意:**安装完成环境后请根据自己的电脑的显卡和安装Pytorch，建议版本为2.3.0

2. 环境激活

   ```shell
   conda activate MC-URL
   ```

### 1.2 数据集

1. 数据集来自`https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/tree/master/data/data.csv`

2. 数据集存放位置为`./data`

3. 生成训练集、验证集和测试集：

   ```shell
   python gen_train_vali_te_data.py
   ```

4. 提取6个机器学习方法的训练特征：(时间较长请耐心等待)

   ```shell
   python gen_tvt_features.py
   ```

### 1.3 预训练模型

1. 将`model`文件夹放到`MC-URL`目录下

## 2. 运行方法

### 2.1 体验demo

1.  对单个url进行测试

   ```shell
   python single_test.py
   ```

### 2.2 测试预训练模型性能

1. 测试8个模型**加权软投票**的性能

   ```shell
   python multi_test.py --test_model=all_weighted
   ```


2. 测试8个模型**平均软投票**的性能

   ```shell
   python multi_test.py --test_model=all_mean
   ```

3. 测试贝叶斯分类器的性能

   ```shell
   python multi_test.py --test_model=bayes
   ```

4. 测试AdaBoost分类器的性能

   ```shell
   python multi_test.py --test_model=ada
   ```

5. 测试随机森林分类器的性能

   ```shell
   python multi_test.py --test_model=rf
   ```

6. 测试决策树分类器的性能

   ```shell
   python multi_test.py --test_model=dt
   ```

7. 测试逻辑回归分类器的性能

   ```shell
   python multi_test.py --test_model=lgs
   ```

8. 测试梯度提升树分类器的性能

   ```shell
   python multi_test.py --test_model=gbc
   ```

9. 测试基于TF-IDF的逻辑回归分类器的性能

   ```shell
   python multi_test.py --test_model=tf_idf
   ```

10. 测试URLGledac的分类器的性能

    ```shell
    python multi_test.py --test_model=URLGledac
    ```



### 2.3 模型训练

1.  训练六个机器学习模型组

   ```shell
   python source.py --mode=train
   ```

2. 训练基于TF-IDF的逻辑回归模型

   ```shell
   python tf_idf.py
   ```

3. 训练基于深度学习的URLGledac模型

   ```shell
   cd ./URLGledac
   python train.py
   ```

   
