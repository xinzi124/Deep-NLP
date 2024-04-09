import jieba
import jieba.analyse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 指定了图表中要使用的字体，其中 [u'SimHei'] 表示使用宋体（SimHei）字体
mpl.rcParams['axes.unicode_minus'] = False #这一行代码用于解决负号显示问题。在一些系统或设置中，默认情况下，Matplotlib 中的负号可能会显示为方块或其他字符，这样会影响图表的可视化效果。将该设置设为 False 可以确保负号显示正常


class ChineseDataSet: #定义了一个名为 ChineseDataSet 的类
    def __init__(self, name): #在类的初始化方法 __init__() 中，接受一个参数 name，并对类的属性进行初始化设置。
        self.data = None #data 属性被设置为 None，表示数据集本身暂时没有被加载或设置。
        self.name = name #name 属性被设置为传入的 name 参数值
        # 单个字
        self.word = []  # 单个字列表
        self.word_len = 0  # 单个字总字数
        # 词
        self.split_word = []  # 单个词列表
        self.split_word_len = 0  # 单个词总数
        with open(".\cn_stopwords.txt", "r", encoding='utf-8') as f:
            self.stop_word = f.read().split('\n') #通过 read() 方法读取文件内容，并通过 split('\n') 方法将其按行分割，生成一个包含停用词的列表，并将其赋值给 stop_word 属性。
            f.close()

    def read_file(self, filename=""):
        # 如果未指定名称，则默认为类名
        if filename == "":
            filename = self.name
        target = "ChineseDataSet/" + filename + ".txt"
        with open(target, "r", encoding='gbk', errors='ignore') as f:
            self.data = f.read()
            self.data = self.data.replace(
                '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '') #对读取的数据进行一些处理，使用 replace() 方法将特定字符串替换为空字符串。
            f.close()


        # 分词 利用 jieba 库对文本进行分词，并过滤掉停用词和空白字符，最终将分词结果存储在 split_word 列表中，并更新 split_word_len 属性记录分词的总数。
        for words in jieba.cut(self.data): #jieba.cut(self.data) 使用了 jieba 库的 cut() 函数对 self.data 中的文本进行分词。jieba.cut() 函数返回的是一个生成器对象，可以迭代得到分词结果。for words in jieba.cut(self.data): 是一个循环语句，遍历了分词结果中的每一个词语，将其赋值给变量 words。
            if (words not in self.stop_word) and (not words.isspace()): #条件判断语句，判断当前词语 words 不在停用词列表中，并且不是空白字符（使用 isspace() 方法进行判断）。
                self.split_word.append(words) #如果满足上述条件，则将当前词语 words 添加到 split_word 列表中
                self.split_word_len += 1 #更新 split_word_len 属性，以记录分词的总数


        # 统计字数  遍历 字符串 self.data 的每一个字符，将不是停用词且不是空白字符的字符添加到 self.word 列表中，并更新 self.word_len 属性记录字符的总数。
        for word in self.data:
            if (word not in self.stop_word) and (not word.isspace()):
                # if not word.isspace():
                self.word.append(word)
                self.word_len += 1

    def write_file(self):
        # 将文件内容写入总文件
        target = "ChineseDataSet/data.txt"
        with open(target, "a") as f:  #使用 with 语句打开目标文件，打开模式为 "a"，表示以追加模式打开文件，如果文件不存在则创建新文件。as f 将打开的文件对象赋值给变量 f。
            f.write(self.data)
            f.close()

    def get_unigram_tf(self, word):
        # 计算单个词的词频（Term Frequency，TF）
        unigram_tf = {}
        for w in word: #遍历参数 word 中的每个词语，将每个词语赋值给变量 w。
            unigram_tf[w] = unigram_tf.get(w, 0) + 1  # unigram_tf[w]：表示在字典 unigram_tf 中查找键为 w 的值。如果 w 在字典中存在，则返回对应的值；如果 w 不存在，则会抛出一个 KeyError 错误。unigram_tf.get(w, 0)：这是字典的 get() 方法，用于获取键 w 对应的值。如果 w 存在于字典中，则返回其对应的值；如果 w 不存在，则返回默认值 0。+ 1：将上一步获取的值加1。最后将计算得到的值，赋值给字典 unigram_tf 中的键 w，表示更新了键 w 对应的值。这段代码的作用是对字典 unigram_tf 中的键 w 对应的值进行更新，如果 w 存在则在原有值上加1，如果 w 不存在则将其值设置为1。这样实现了对词语的计数，即统计词语出现的频次
        return unigram_tf

    def get_bigram_tf(self, word):
        # 计算二元词（Bigram）的词频（Term Frequency，TF）
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1   #对于当前词语 word[i] 和下一个词语 word[i + 1] 组成的二元词，使用 (word[i], word[i + 1]) 作为键，通过 bigram_tf.get() 方法获取该二元词在字典中的值，如果该二元词不在字典中，则返回默认值 0。然后将其加1，表示词频加1。最后将结果存储回字典中。
        return bigram_tf

    def get_trigram_tf(self, word):
        # 计算三元词（Trigram）的词频（Term Frequency，TF）
        trigram_tf = {}
        for i in range(len(word) - 2):
            trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
                (word[i], word[i + 1], word[i + 2]), 0) + 1
        return trigram_tf

    def calc_entropy_unigram(self, word, is_ci=0):
        # 计算一元模型（Unigram Model）的信息熵
        word_tf = self.get_unigram_tf(word) #调用了类中的 get_unigram_tf() 方法，计算给定词语列表 word 的词频字典，并将结果存储在 word_tf 中。
        word_len = sum([item[1] for item in word_tf.items()]) #word_tf.items() 是一个字典视图对象，它提供了字典中所有键值对的视图。在这个视图中，每个元素都是一个键值对，item[0] 表示键（即词语），item[1] 表示值（即该词语的词频）。[item[1] for item in word_tf.items()] 是一个列表推导式，它遍历了 word_tf.items() 中的每一个键值对，将每个键值对的值（词频）提取出来，形成一个新的列表，即词频列表。sum([item[1] for item in word_tf.items()]) 使用了 Python 内置的 sum() 函数，将词频列表中的所有词频值相加，从而计算了词频字典中所有词频的总和，即文本的总词数。
        entropy = sum(
            [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
             word_tf.items()])   #对于词频字典中的每个词语，计算其信息熵，然后将所有词语的信息熵相加，得到了整个文本的信息熵。
        if is_ci:
            print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_bigram(self, word, is_ci=0):
        # 计算二元模型的信息熵
        # 计算二元模型总词频
        word_tf = self.get_bigram_tf(word)
        last_word_tf = self.get_unigram_tf(word) #调用了类中的 get_unigram_tf() 方法，计算给定词语列表 word 的一元词频字典，并将结果存储在 last_word_tf 中，用于计算条件概率。
        bigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for bigram in word_tf.items():
            p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
            p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y) bigram[0][0] 表示二元词的第一个词，bigram[1] 表示二元词的词频。
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_trigram(self, word, is_ci):
        # 计算三元模型的信息熵
        # 计算三元模型总词频
        word_tf = self.get_trigram_tf(word)
        last_word_tf = self.get_bigram_tf(word)
        trigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for trigram in word_tf.items():
            p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
            p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y) 其中 trigram[0][0] 和 trigram[0][1] 表示三元词的前两个词，trigram[1] 表示三元词的词频。
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy


#定义了一个名为 my_plot() 的函数，用于绘制柱状图，展示不同模型（一元模型、二元模型、三元模型）在不同数据库上的信息熵情况。
def my_plot(X, Y1, Y2, Y3, num): #X、Y1、Y2、Y3、num 分别是函数的参数，分别表示数据库标签、一元模型信息熵、二元模型信息熵、三元模型信息熵以及图表的类型（1表示以字为单位的信息熵，2表示以词为单位的信息熵）。
    # 标签位置
    x = range(0, len(X))
    # 柱状图宽度
    width = 0.2
    # 各柱状图位置，确保每个模型的柱状图在 x 轴上分布均匀，并且不会相互重叠
    x1_width = [i - width * 2 for i in x] #将每个数据库标签的位置向左移动了两个柱状图的宽度
    x2_width = [i - width for i in x] #将每个数据库标签的位置向左移动了一个柱状图的宽度
    x3_width = [i for i in x]
    # 设置图片大小、绘制柱状图
    plt.figure(figsize=(19.2, 10.8))
    #使用 plt.bar() 函数绘制柱状图
    plt.bar(x1_width, Y1, fc="r", width=width, label="一元模型")
    plt.bar(x2_width, Y2, fc="b", width=width, label="二元模型")
    plt.bar(x3_width, Y3, fc="g", width=width, label="三元模型")
    # 设置x轴
    plt.xticks(x, X, rotation=40, fontsize=10) #使用 plt.xticks() 设置 x 轴的刻度和标签，使得数据库标签能够显示，并设置刻度的旋转角度和字体大小
    plt.xlabel('数据库', fontsize=10)
    # 设置y轴
    plt.ylabel('信息熵', fontsize=10)
    plt.ylim(0, max(Y1) + 2)  #设置 y 轴的范围
    # 标题
    if (num == 1):
        plt.title("以字为单位的信息熵", fontsize=10)
    elif num == 2:
        plt.title("以词为单位的信息熵", fontsize=10)
    # 标注柱状图上方文字
    autolabel(x1_width, Y1)#自定义函数 autolabel()，该函数用于在柱状图上方标注具体数值。
    autolabel(x2_width, Y2)
    autolabel(x3_width, Y3)

    plt.legend() #添加图例
    plt.savefig('chinese' + str(num) + '.png') #保存图表为图片文件
    plt.show() #显示图表

#定义了一个名为 autolabel() 的函数，用于在柱状图的每个柱子上方标注具体的数值。
def autolabel(x, y):
    for a, b in zip(x, y): #zip() 函数接受多个可迭代对象作为参数，然后从每个可迭代对象中依次取出相同位置的元素，将它们打包成一个元组。
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=10) #a 表示柱子的横坐标位置；b + 0.01 表示柱子高度加上一个微小的偏移量，以使文本标注位于柱子的上方；'%.3f' % b 表示要显示的数值，'%.3f' 是格式化字符串，表示保留小数点后三位；ha='center' 表示水平对齐方式为居中；va='bottom' 表示垂直对齐方式为底部，即文本位于柱子的上方；fontsize=10 表示文本的字体大小为10号字体。


if __name__ == "__main__":
    data_set_list = []
    # 每次运行程序将总内容文件清空
    with open("ChineseDataSet/data.txt", "w") as f: #"w" 参数表示以写入模式打开文件，如果文件已经存在，则先清空文件内容，然后重新写入。
        f.close()
    with open("log.txt", "w") as f:
        f.close()
    # 读取小说名字 动态创建 ChineseDataSet 对象，并将它们存储到一个列表中
    with open("ChineseDataSet/inf.txt", "r") as f:
        txt_list = f.read().split(',') #读取小说名字，并按逗号分隔成列表 txt_list。
        i = 0
        for name in txt_list:
            locals()[f'set{i}'] = ChineseDataSet(name) #locals()[f'set{i}'] 创建了一个新的局部变量，f'set{i}' 使用了 Python 中的 f-string 格式化字符串，将变量 i 的值插入到字符串中，从而动态生成变量名。
            data_set_list.append(locals()[f'set{i}'])
            i += 1
        f.close()
    # 分别针对每本小说进行操作
    word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy = [], [], [], [], [], [] #用于存储每本小说字和词的一元、二元和三元模型的信息熵。
    for set in data_set_list:
        set.read_file()
        set.write_file()
        # 字为单位
        word_unigram_entropy.append(set.calc_entropy_unigram(set.word, 0)) #set 是循环变量，表示 data_set_list 中的每个 ChineseDataSet 对象；set.word 是一个列表，其中包含了整个文本内容中的每个字；0 是一个布尔值参数，用于指示计算中采用的是基于词汇还是基于字的模型。在这里，当参数为 0 时，表示计算基于字的模型的信息熵。
        word_bigram_entropy.append(set.calc_entropy_bigram(set.word, 0))
        word_trigram_entropy.append(set.calc_entropy_trigram(set.word, 0))
        # 词为单位
        words_unigram_entropy.append(set.calc_entropy_unigram(set.split_word, 1)) #set.split_word 是一个列表，其中包含了文本内容中的每个词。当参数为 1 时，表示计算基于词汇的模型的信息熵。
        words_bigram_entropy.append(set.calc_entropy_bigram(set.split_word, 1))
        words_trigram_entropy.append(set.calc_entropy_trigram(set.split_word, 1))
        with open("log.txt", "a") as f:
            f.write("{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set.name,  #格式化 set.name 的输出，确保名字左对齐，并且最小宽度为 10 个字符。
                                                                                                               set.word_len, #格式化 set.word_len 的输出，确保输出的整数右对齐，并且最小宽度为 10 个字符
                                                                                                               set.split_word_len,
                                                                                                               word_unigram_entropy[
                                                                                                                   -1],
                                                                                                               word_bigram_entropy[
                                                                                                                   -1],
                                                                                                               word_trigram_entropy[
                                                                                                                   -1],
                                                                                                               words_unigram_entropy[
                                                                                                                   -1],
                                                                                                               words_bigram_entropy[  #格式化一个浮点数，保留小数点后四位
                                                                                                                   -1],
                                                                                                               words_trigram_entropy[
                                                                                                                   -1]))
            f.close()
    # 对所有小说进行操作
    set_total = ChineseDataSet("total")
    set_total.read_file("data")
    word_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.word, 0))
    word_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.word, 0))
    word_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.word, 0))

    words_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.split_word, 1))
    words_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.split_word, 1))
    words_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.split_word, 1))

    with open("log.txt", "a") as f:
        f.write(
            "{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set_total.name,
                                                                                                       set_total.word_len,
                                                                                                       set_total.split_word_len,
                                                                                                       word_unigram_entropy[
                                                                                                           -1],
                                                                                                       word_bigram_entropy[
                                                                                                           -1],
                                                                                                       word_trigram_entropy[
                                                                                                           -1],
                                                                                                       words_unigram_entropy[
                                                                                                           -1],
                                                                                                       words_bigram_entropy[
                                                                                                           -1],
                                                                                                       words_trigram_entropy[
                                                                                                           -1]))
        f.close()
    # 绘图
    x_label = [i.name for i in data_set_list]
    x_label.append(set_total.name)

    my_plot(x_label, word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, 1)
    my_plot(x_label, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy, 2)
