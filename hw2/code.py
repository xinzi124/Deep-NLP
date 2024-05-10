import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import os
from tqdm import tqdm

#从每个标签中均匀抽取指定数量的段落，形成一个新的数据集，并返回这个数据集以及对应的标签。
def extract_dataset(corpus, labels, num_paragraphs):
    # 计算每个标签需要抽取的段落数量
    num_paragraphs_per_label = num_paragraphs // len(set(labels))
    dataset = []
    dataset_labels = []
    for label in set(labels):
        # 从具有特定标签的段落中均匀抽取指定数量的段落
        label_paragraphs = [paragraph for paragraph, paragraph_label in zip(corpus, labels) if paragraph_label == label]
        sampled_paragraphs = np.random.choice(label_paragraphs, num_paragraphs_per_label, replace=False)
        dataset.extend(sampled_paragraphs)
        dataset_labels.extend([label] * num_paragraphs_per_label)
    return dataset, dataset_labels #抽取的数据集 dataset 和相应的标签 dataset_labels。

# 构建语料库，corpus 中每个元素是一个段落，而 labels 中每个元素是对应的段落标签
folder_path = "jyxstxtqj_downcc.com"

 # 读取文件夹下所有txt文件的内容并合并成一个语料库，同时为每个段落标记相应的标签。
corpus = []
labels = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        with open(os.path.join(folder_path, file_name), "r",encoding='gb18030') as file:
            text = file.read()
            paragraphs = text.split("\n")
            corpus.extend(paragraphs)
            labels.extend(["book_" + file_name.split(".")[0]] * len(paragraphs))
    # 定义不同的 K
K_values = [20, 100, 500, 1000, 3000]

    # 定义不同的主题数量 T
T_values = [5, 10,25, 50, 100]

    # 定义交叉验证的次数
num_cross_val = 10

    # 定义分类器
classifiers = {
        "SVM": SVC()
    }

    # 定义结果存储列表
results = []
from sklearn.preprocessing import LabelEncoder
dataset, dataset_labels = extract_dataset(corpus, labels, num_paragraphs=1000)#从语料库 corpus 和标签列表 labels 中抽取数据集。这里指定了要抽取的段落数量为 1000。抽取的数据集存储在 dataset 中，相应的标签存储在 dataset_labels 中。

# 将字符串类别标签编码为整数类别标签
label_encoder = LabelEncoder()
dataset_labels = label_encoder.fit_transform(dataset_labels)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels, test_size=0.1, random_state=42)
for K in tqdm(K_values):
    for T in tqdm(T_values):
        lda_pipeline = Pipeline([
                ('vectorizer', CountVectorizer(max_features=K, analyzer='char')),#CountVectorizer 的参数 max_features 设置为 K，表示只考虑词频前 K 个最高频的词语。CountVectorizer 的参数 analyzer 设置为 'word'，表示将文本分析为单词级别；如果你想按照字符切分文本，可以将 analyzer 参数设置为 'char'。这样，文本将会被分割成单个字符，而不是单词
                ('lda', LatentDirichletAllocation(n_components=T, random_state=42))
            ]) #为每个参数组合创建一个包含 CountVectorizer 和 LatentDirichletAllocation 的管道，其中 CountVectorizer 用于将文本转换为词袋表示，LatentDirichletAllocation 用于执行 LDA 主题建模。

            # 将文本转换为主题分布
        X_train_lda = lda_pipeline.fit_transform(X_train)
        X_test_lda = lda_pipeline.transform(X_test)

            # 使用不同的分类器进行训练和评估
        for classifier_name, classifier in classifiers.items():
                # 保存结果

            classifier.fit(X_train_lda, y_train)
            accuracy = np.mean(cross_val_score(classifier, X_train_lda, y_train, cv=num_cross_val))
            test_accuracy = accuracy_score(y_test, classifier.predict(X_test_lda))

                # 保存结果

            results.append({
                    'K': K,
                    'T': T,
                    'Classifier': classifier_name,
                    'Analyzer': 'char',
                    'Training Accuracy': accuracy,
                    'Test Accuracy': test_accuracy
                })

    # 将结果转换为DataFrame
results = pd.DataFrame(results)

    # 保存结果到xlsx文件
results.to_excel("result.xlsx", index=False)