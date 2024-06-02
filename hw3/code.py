# -*- coding: utf-8 -*-
import os
import jieba
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 文本预处理函数
def preprocess_text(text):
    punctuation_pattern = r'[。，、；：？！（）《》【】“”‘’…—\-,.:;?!\[\](){}\'"<>]'
    text0 = re.sub(punctuation_pattern, '', text)
    text1 = re.sub(r'[\n\r\t]', '', text0)
    text2 = re.sub(r'[^\u4e00-\u9fa5]', '', text1)
    return text2

# 读取停用词函数
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return stopwords

# 分词和去停用词函数
def cut_text(text, stopwords):
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and word.strip()]

# 计算段落向量
def get_paragraph_vector(paragraph, model):
    words = [word for word in jieba.cut(paragraph) if word not in stopwords]
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 计算段落间相似度
def calculate_similarity(paragraph1, paragraph2, model):
    vector1 = get_paragraph_vector(paragraph1, model)
    vector2 = get_paragraph_vector(paragraph2, model)
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0
    else:
        return cosine_similarity([vector1], [vector2])[0][0]

# 加载文本数据
corpus = []
directory = os.path.join(os.getcwd(), 'jyxstxtqj_downcc.com')
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='ansi') as file:
            text = file.read()
            cleaned_text = preprocess_text(text)
            corpus.append(cleaned_text)

# 加载停用词列表
stopwords_path = os.path.join(os.getcwd(), 'stopwords.txt')
stopwords = load_stopwords(stopwords_path)
processed_corpus = [cut_text(text, stopwords) for text in corpus]

# 训练Word2Vec模型
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=5, workers=16, epochs=50)

# 保存模型
model.save("word2vec.model")

# 加载模型
model = Word2Vec.load("word2vec.model")

# 查询词之间的相似度
word1 = "杨过"
word2 = "小龙女"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")

# 保存相似度结果到文件
with open('similarity_results.txt', 'a', encoding='utf-8') as file:
    file.write(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}\n")

# KMeans聚类
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
labels = kmeans.labels_

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
word_vectors_2d = tsne.fit_transform(word_vectors)

# 绘制散点图
plt.figure(figsize=(10, 8))
palette = sns.color_palette("viridis", as_cmap=True)
sns.scatterplot(x=word_vectors_2d[:, 0], y=word_vectors_2d[:, 1], hue=labels, palette=palette, legend="full", s=10)
plt.title('Word2Vec Word Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Cluster')

# 保存图像
plt.savefig('word_clusters.png')

plt.show()

# 打印每个簇中的词语
clusters = {}
for i in range(num_clusters):
    clusters[i] = []

for word, label in zip(words, labels):
    clusters[label].append(word)

with open('cluster_words.txt', 'w', encoding='utf-8') as file:
    for cluster_id, cluster_words in clusters.items():
        file.write(f"Cluster {cluster_id}: {', '.join(cluster_words)}\n")
        print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")

# 示例段落
paragraph1 = "当晚林震南安排了众镖师守夜，哪知自己仗剑巡查之时，见十多名镖师竟是团团坐在厅上，没一人在外把守。众镖师见到总镖头，都讪讪的站起身来，却仍无一人移动脚步。林震南心想敌人实在太强，局中已死了这样多人，自己始终一筹莫展，也怪不得众人胆怯，当下安慰了几句，命人送酒菜来，陪着众镖师在厅上喝酒。众人心头烦恼，谁也不多说话，只喝那闷酒，过不多时，便已醉倒了数人。次日午后，忽听得马蹄声响，有几骑马从镖局中奔了出去。林震南一查，原来是五名镖师耐不住这局面，不告而去。他摇头叹道：“大难来时各自飞。姓林的无力照顾众位兄弟，大家要去便去罢。”余下众镖师有的七张八嘴，指斥那五人太没义气；有几人却默不作声，只是叹气，暗自盘算：“我怎么不走？”傍晚时分，五匹马又驮了五具尸首回来。这五名镖师意欲逃离险地，反而先送了性命。"
paragraph2 ="林震南道：“他确是将福威镖局视若无物。”林平之道：“说不定他是怕了爹爹的七十二辟邪剑法，否则为甚么始终不敢明剑明枪的交手，只是趁人不备，暗中害人？”林震南摇头道：“平儿，爹爹的辟邪剑法用以对付黑道中的盗贼，那是绰绰有余，但此人的摧心掌功夫，实是远远胜过了你爹爹。我……我向不服人，可是见了霍镖头的那颗心，却是……却是……唉！”林平之见父亲神情颓丧，和平时大异，不敢再说甚么。王夫人道：“既然对头厉害，大丈夫能屈能伸，咱们便暂且避他一避。”林震南点头道：“我也这么想。”王夫人道：“咱们连夜动身去洛阳，好在已知道敌人来历，君子报仇，十年未晚。”林震南道：“不错！岳父交友遍天下，定能给咱们拿个主意。收拾些细软，这便动身。”林平之道：“咱们一走，丢下镖局中这许多人没人理会，那可如何是好？”林震南道：“敌人跟他们无冤无仇，咱们一走，镖局中的众人反而太平无事了。”林平之心道：“爹爹这话有理，敌人害死镖局中这许多人，其实只是为了我一人。我脱身一走，敌人决不会再和这些镖师、趟子手为难。”当下回到自己房中收拾。心想说不定敌人一把火便将镖局烧个精光，看着一件件衣饰玩物，只觉这样舍不得，那件丢不下，竟打了老大两个包裹，兀自觉得留下东西太多，左手又取过案上一只玉马，右手卷了张豹皮，那是从他亲手打死的花豹身上剥下来的，背负包裹，来到父母房中。王夫人见了不禁好笑，说道：“咱们是逃难，可不是搬家，带这许多劳甚子干么？”林震南叹了一口气，摇了摇头，心想：“我们虽是武学世家，但儿子自小养尊处优，除了学过一些武功之外，跟寻常富贵人家的纨裤子弟也没甚么分别，今日猝逢大难，仓皇应变，却也难怪得他。”不由得爱怜之心，油然而生，说道：“你外公家里甚么东西都有，不必携带太多物件。咱们只须多带些黄金银两，值钱的珠宝也带一些。此去到江西、湖南、湖北都有分局，还怕上讨饭么？包裹越轻越好，身上轻一两，动手时便灵便一分。”林平之无奈，只得将包裹放下。王夫人道：“咱们骑马从大门光明正大的冲出去，还是从后门悄悄溜出去？”林震南坐在太师椅上，闭起双目，将旱烟管抽得呼呼直响，过了半天，才睁开眼来，说道：“平儿，你去通知局中上下人等，大家收拾收拾，天明时一齐离去。叫帐房给大家分发银两。待瘟疫过后，大家再回来。”林平之应道：“是！”心下好生奇怪，怎地父亲忽然又改变了主意。王夫人道：“你说要大家一哄而散？这镖局子谁来照看？”林震南道：“不用看了，这座闹鬼的凶宅，谁敢进来送死？再说，咱三人一走，余下各人难道不走？”当下林平之出房传讯，局中登时四下里都乱了起来。林震南待儿子出房，才道：“娘子，咱父子换上趟子手的衣服，你就扮作个仆妇，天明时一百多人一哄而散，敌人武功再高，也不过一两个人，他又去追谁好？”王夫人拍掌赞道：“此计极高。”便去取了两套趟子手的污秽衣衫，待林平之回来，给他父子俩换上，自己也换了套青布衣裳，头上包了块蓝花布帕，除了肤色太过白皙，宛然便是个粗作仆妇。林平之只觉身上的衣衫臭不可当，心中老大不愿意，却也无可奈何。黎明时分，林震南吩咐打开大门，向众人说道：“今年我时运不利，局中疫鬼为患，大伙儿只好避一避。众位兄弟倘若仍愿干保镖这一行的，请到杭州府、南昌府去投咱们的浙江分局、江西分局，那边刘镖头、易镖头自不会怠慢了各位。咱们走罢！”当下一百余人在院子中纷纷上马，涌出大门。林震南将大门上了锁，一声呼叱，十余骑马冲过血线，人多胆壮，大家已不如何害怕，都觉早一刻离开镖局，便多一分安全。蹄声杂沓，齐向北门奔去，众人大都无甚打算，见旁人向北，便也纵马跟去。林震南在街角边打个手势，叫夫人和儿子留了下来，低声道：“让他们向北，咱们却向南行。”王夫人道：“去洛阳啊，怎地往南？”林震南道：“敌人料想咱们必去洛阳，定在北门外拦截，咱们却偏偏向南，兜个大圈子再转而向北，叫狗贼拦一个空。”林平之道：“爹！”林震南道：“怎么？”林平之不语，过了片刻，又道：“爹。”王夫人道：“你想说甚么，说出来罢。”林平之道：“孩儿还是想出北门，这狗贼害死了咱们这许多人，不跟他拚个你死我活，这口恶气如何咽得下去？”王夫人道：“这番大仇，自然是要报的，但凭你这点儿本领，抵挡得了人家的摧心掌么？”林平之气忿忿的道：“最多也不过像霍镖头那样，给他一掌碎了心脏，也就是啦。”"

# 计算段落相似度
similarity_score = calculate_similarity(paragraph1, paragraph2, model)
print(f"段落1与段落2的语义相似度：{similarity_score}")

# 保存段落相似度结果到文件
with open('similarity_results.txt', 'a', encoding='utf-8') as file:
    file.write(f"段落1与段落2的语义相似度：{similarity_score}\n")
