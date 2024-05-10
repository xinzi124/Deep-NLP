import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import random
from sklearn.svm import SVC  # 以SVM作为分类器示例
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import random
import string
from gensim import corpora, models
from collections import defaultdict

#包含所有停用词的列表
def load_stopwords(file_path):
    stop_words = []
    with open('cn_stopwords.txt', "r", encoding="gb18030", errors="ignore") as f:
        stop_words.extend([word.strip('\n') for word in f.readlines()])
    return stop_words

#文本预处理，去除文本中的停用词。
def preprocess_corpus( text,cn_stopwords):
    for tmp_char in cn_stopwords:
        text = text.replace(tmp_char, "")             
    return text 

#提取段落和对应的标签，并确保提取的段落数量不超过指定的数量。
def extract_paragraphs_and_labels(corpus_dict, num_paragraphs, k_value):
    result = []
    total_paragraphs_count = sum(len(corpus_dict[novel]) for novel in corpus_dict)

    if total_paragraphs_count < num_paragraphs:
        print(f"Warning: Only {total_paragraphs_count} paragraphs available in the corpus. Requested {num_paragraphs} will be returned.")
        num_paragraphs = total_paragraphs_count

    # 统计每个小说的段落数量，用于均匀抽取
    paragraph_counts = {novel: len(paragraphs) for novel, paragraphs in corpus_dict.items()} #paragraph_counts 字典中的每个键值对表示了一个小说的名称和该小说的段落数量
    # 均匀抽取指定数量的段落
    for _ in range(num_paragraphs):
        # 随机选择一个小说
        novel = random.choices(list(corpus_dict.keys()), weights=[count / total_paragraphs_count for count in paragraph_counts.values()], k=1)[0]
        # 从该小说中随机抽取一个段落
        paragraphs = corpus_dict[novel]
        paragraphs=re.split(r'\n\u3000\u3000', paragraphs)
        paragraph = random.choice(paragraphs)
        # 根据 K 值范围，为该段落选择一个随机的 token 数量
        # 对段落进行截断（如果需要），确保其包含指定数量的 token
        tokens = list(jieba.cut(paragraph)) #
        result.append(( tokens, novel, k_value))
    return result

def LDA(processed_data,num_topics = 10 ):


    X = [item[0] for item in processed_data]  # 段落文本列表
    y = [item[1] for item in processed_data]  # 段落所属小说标签列表

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 训练LDA模型
    dictionary = corpora.Dictionary(X_train)

    #将文本数据转换为 LDA 模型可以处理的词袋
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in X_train]
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=num_topics)
    #训练集上的文档主题分布
    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    
    X_train_lda = np.zeros((len(X_train), num_topics))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            X_train_lda[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]


    classifier = SVC(kernel='linear', C=1, random_state=42) #创建了一个支持向量分类器 SVC
    classifier.fit(X_train_lda, y_train)#使用 fit 方法将训练集的 LDA 表示 X_train_lda 和对应的标签 y_train 进行拟合。

    #对测试集的每个文档进行 LDA 主题建模，得到每个文档的主题分布，然后构建一个与训练集类似的二维 NumPy 数组 X_test_lda，用于存储测试集中每个文档的主题分布。最后，使用拟合好的分类器对测试集的 LDA 表示进行预测。
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in X_test]
    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    X_test_lda = np.zeros((len(X_test), num_topics))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            X_test_lda[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]

    #对测试集进行预测并评估分类器的性能
    y_pred = classifier.predict(X_test_lda)
    print("Classification Report:")#分类报告提供了每个类别的精确度、召回率和 F1 分数等指标。
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))#准确率衡量了模型正确分类的样本比例。
    print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))#F1 分数是精确率和召回率的加权平均，它同时考虑了分类器的精确度和召回率，特别适用于不平衡类别的情况。Macro-F1 是对每个类别的 F1 分数取算术平均。

if __name__ == '__main__':
    stopwords_file_path = 'cn_stopwords.txt'
    cn_stopwords = load_stopwords(stopwords_file_path)          
    corpus_dict = {}
    book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"#
    for book_title in book_titles_list.split(','):
        book_title = book_title.strip()  # 去除可能存在的多余空白字符
        file_path='jyxstxtqj_downcc.com\{}.txt'.format(book_title)
        merged_content = ''
        with open(file_path, 'r', encoding='gb18030') as f:
            merged_content += f.read()
        # 保存合并后的内容到新的文本文件
        merged_content=preprocess_corpus( merged_content,cn_stopwords)
        output_file_path = 'jyxstxtqj_downcc.com\{}.txt'.format(book_title)
        with open(output_file_path, 'w', encoding='gb18030') as f:
            f.write(merged_content)
        corpus_dict[book_title]=merged_content
    num_paragraphs = 1000
    num_topics = 10
    k_values = [20, 100, 500, 1000, 3000]
    k_value =k_values [4]
    processed_data = extract_paragraphs_and_labels(corpus_dict, num_paragraphs, k_value)
    LDA(processed_data,    num_topics = 10 )
        