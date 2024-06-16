#!/usr/bin/python  # 指定脚本的解释器为Python
# -*- coding:utf-8 -*-  # 指定文件编码为utf-8

import numpy as np  # 导入NumPy库，用于数值计算
import random  # 导入random模块，用于生成随机数
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 从PyTorch导入神经网络模块
from torch.utils.data import Dataset, DataLoader  # 从PyTorch导入数据集和数据加载器
import os


class CorpusDataset(Dataset):  # 定义一个数据集类，用于处理源数据和目标数据，并进行数据对齐和索引转换
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx):  # 初始化函数，用于初始化数据集
        self.source_data = source_data  # 保存源数据
        self.target_data = target_data  # 保存目标数据
        self.source_word_2_idx = source_word_2_idx  # 保存源数据词到索引的映射
        self.target_word_2_idx = target_word_2_idx  # 保存目标数据词到索引的映射

    def __getitem__(self, index):  # 获取数据集中指定索引的数据
        src = self.source_data[index]  # 获取源数据中的一个句子
        tgt = self.target_data[index]  # 获取目标数据中的一个句子
        src_index = [self.source_word_2_idx[i] for i in src]  # 将源句子中的词转换为索引
        tgt_index = [self.target_word_2_idx[i] for i in tgt]  # 将目标句子中的词转换为索引
        return src_index, tgt_index  # 返回源句子和目标句子的索引表示

    def batch_data_alignment(self, batch_datas):  # 批量数据对齐函数
        global device  # 声明使用全局变量device
        src_index, tgt_index = [], []  # 初始化源索引和目标索引的列表
        src_len, tgt_len = [], []  # 初始化源长度和目标长度的列表
        for src, tgt in batch_datas:  # 遍历批量数据
            src_index.append(src)  # 添加源索引到列表中
            tgt_index.append(tgt)  # 添加目标索引到列表中
            src_len.append(len(src))  # 添加源句子的长度到列表中
            tgt_len.append(len(tgt))  # 添加目标句子的长度到列表中
        max_src_len = max(src_len)  # 获取源句子的最大长度
        max_tgt_len = max(tgt_len)  # 获取目标句子的最大长度
        src_index = [[self.source_word_2_idx["<BOS>"]] + tmp_src_index + [self.source_word_2_idx["<EOS>"]] +
                     [self.source_word_2_idx["<PAD>"]] * (max_src_len - len(tmp_src_index)) for tmp_src_index in src_index]  # 对源句子进行填充对齐
        tgt_index = [[self.target_word_2_idx["<BOS>"]] + tmp_src_index + [self.target_word_2_idx["<EOS>"]] +
                     [self.target_word_2_idx["<PAD>"]] * (max_tgt_len - len(tmp_src_index)) for tmp_src_index in tgt_index]  # 对目标句子进行填充对齐
        src_index = torch.tensor(src_index, device=device)  # 将源索引转换为PyTorch张量
        tgt_index = torch.tensor(tgt_index, device=device)  # 将目标索引转换为PyTorch张量
        return src_index, tgt_index  # 返回对齐后的源索引和目标索引

    def __len__(self):  # 获取数据集的长度
        assert len(self.source_data) == len(self.target_data)  # 确保源数据和目标数据的长度相同
        return len(self.target_data)  # 返回目标数据的长度

class Encoder(nn.Module):  # 定义一个编码器类，用于seq2seq模型处理输入序列并生成隐藏状态
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.embedding = nn.Embedding(source_corpus_len, dim_encoder_embbeding)  # 定义嵌入层，将词转换为向量
        self.lstm = nn.LSTM(dim_encoder_embbeding, dim_encoder_hidden, batch_first=True)  # 定义LSTM层，用于处理序列数据

    def forward(self, src_index):  # 前向传播函数
        en_embedding = self.embedding(src_index)  # 对源索引进行嵌入操作
        _, encoder_hidden = self.lstm(en_embedding)  # 将嵌入后的向量输入LSTM，获取隐藏状态
        return encoder_hidden  # 返回编码器的隐藏状态

class Decoder(nn.Module):  # 定义一个解码器类，与编码器配合使用，从编码器输出的隐藏状态开始生成目标序列
    def __init__(self, dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.embedding = nn.Embedding(target_corpus_len, dim_decoder_embedding)  # 定义嵌入层，将词转换为向量
        self.lstm = nn.LSTM(dim_decoder_embedding, dim_decoder_hidden, batch_first=True)  # 定义LSTM层，用于处理序列数据

    def forward(self, decoder_input, hidden):  # 前向传播函数
        embedding = self.embedding(decoder_input)  # 对解码器输入进行嵌入操作
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)  # 将嵌入后的向量输入LSTM，获取输出和隐藏状态
        return decoder_output, decoder_hidden  # 返回解码器的输出和隐藏状态

class Seq2Seq(nn.Module):  # 定义一个seq2seq模型类，用于将源序列映射到目标序列
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len,
                 dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.encoder = Encoder(dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len)  # 初始化编码器
        self.decoder = Decoder(dim_decoder_embedding, dim_decoder_hidden, target_corpus_len)  # 初始化解码器
        self.classifier = nn.Linear(dim_decoder_hidden, target_corpus_len)  # 定义线性层，用于将解码器输出转换为词的概率分布
        self.ce_loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    def forward(self, src_index, tgt_index):  # 前向传播函数
        decoder_input = tgt_index[:, :-1]  # 获取目标索引的输入部分
        label = tgt_index[:, 1:]  # 获取目标索引的标签部分
        encoder_hidden = self.encoder(src_index)  # 将源索引输入编码器，获取隐藏状态
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)  # 将解码器输入和编码器隐藏状态输入解码器，获取输出
        pre = self.classifier(decoder_output)  # 将解码器输出输入线性层，获取预测结果
        loss = self.ce_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))  # 计算损失
        return loss  # 返回损失

def generate_sentence(sentence):  # 定义一个函数，从一个给定的源句子生成目标语言的翻译或输出句子
    global source_word_2_idx, model, device, target_word_2_idx, target_idx_2_word  # 声明使用全局变量
    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)  # 将源句子转换为索引，并转换为PyTorch张量
    result = []  # 初始化结果列表
    encoder_hidden = model.encoder(src_index)  # 将源索引输入编码器，获取隐藏状态
    decoder_input = torch.tensor([[target_word_2_idx["<BOS>"]]], device=device)  # 初始化解码器输入为开始符
    decoder_hidden = encoder_hidden  # 将编码器隐藏状态作为解码器隐藏状态
    while True:  # 开始循环生成句子
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # 将解码器输入和隐藏状态输入解码器，获取输出和隐藏状态
        pre = model.classifier(decoder_output)  # 将解码器输出输入线性层，获取预测结果
        w_index = int(torch.argmax(pre, dim=-1))  # 获取预测结果中概率最大的索引
        word = target_idx_2_word[w_index]  # 将索引转换为词
        if word == "<EOS>" or len(result) > 40:  # 如果遇到结束符或生成的句子长度超过40，则停止生成
            break
        result.append(word)  # 将生成的词添加到结果列表中
        decoder_input = torch.tensor([[w_index]], device=device)  # 将生成的词作为下一个解码器输入
    return "".join(result)  # 返回生成的句子

if __name__ == '__main__':  # 如果当前模块是主模块，则执行以下代码
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 如果有GPU可用，则使用GPU，否则使用CPU
    batch_size = 2  # 设置批量大小为2
    num_corpus = 300  # 设置语料数量为300
    num_test_corpus = 10  # 设置测试语料数量为10
    txt_file_path = "jyxstxtqj_downcc.com/天龙八部.txt"  # 设置文本文件路径
    num_epochs = 50  # 设置训练的轮数为50
    lr = 0.001  # 设置学习率为0.001
    dim_encoder_embbeding = 150  # 设置编码器嵌入层的维度为150
    dim_encoder_hidden = 100  # 设置编码器隐藏层的维度为100
    dim_decoder_embedding = 150  # 设置解码器嵌入层的维度为150
    dim_decoder_hidden = 100  # 设置解码器隐藏层的维度为100
    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"  # 定义需要替换的字符
    source_target_corpus_ori = []  # 初始化源目标语料列表
    with open(txt_file_path, "r", encoding="gbk", errors="ignore") as tmp_file:  # 打开文本文件，使用gbk编码，忽略错误
        tmp_file_context = tmp_file.read()  # 读取文件内容
        for tmp_char in char_to_be_replaced:  # 遍历需要替换的字符
            tmp_file_context = tmp_file_context.replace(tmp_char, "")  # 将字符替换为空
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")  # 替换广告信息
        tmp_file_sentences = tmp_file_context.split("。")  # 将文本按句号分割成句子
        for tmp_idx, tmp_sentence in enumerate(tmp_file_sentences):  # 遍历句子
            if ("她" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):  # 如果句子符合条件
                source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))  # 添加句子对到语料列表中
    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)  # 随机抽取语料的索引
    source_corpus, target_corpus = [], []  # 初始化源语料和目标语料列表
    output_file_path = "./seq2seq_generated_sentences.txt"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for idx in sample_indexes:  # 遍历抽取的索引
        source_corpus.append(source_target_corpus_ori[idx][0])  # 添加源句子到源语料列表中
        target_corpus.append(source_target_corpus_ori[idx][1])  # 添加目标句子到目标语料列表中
    test_corpus = []  # 初始化测试语料列表
    for idx in range(len(source_target_corpus_ori)):  # 遍历所有语料的索引
        if idx not in sample_indexes:  # 如果索引不在抽取的索引中
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))  # 添加句子对到测试语料列表中
    test_corpus = random.sample(test_corpus, num_test_corpus)  # 随机抽取测试语料
    test_source_corpus, test_target_corpus = [], []  # 初始化测试源语料和测试目标语料列表
    for tmp_src, tmp_tgt in test_corpus:  # 遍历测试语料
        test_source_corpus.append(tmp_src)  # 添加源句子到测试源语料列表中
        test_target_corpus.append(tmp_tgt)  # 添加目标句子到测试目标语料列表中
    # one-hot编码字典
    idx_cnt = 0  # 初始化索引计数器
    word_2_idx_dict = dict()  # 初始化词到索引的字典
    idx_2_word_list = list()  # 初始化索引到词的列表
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:  # 遍历所有语料
        for tmp_sentence in tmp_corpus:  # 遍历每个句子
            for tmp_word in tmp_sentence:  # 遍历每个词
                if tmp_word not in word_2_idx_dict.keys():  # 如果词不在字典中
                    word_2_idx_dict[tmp_word] = idx_cnt  # 将词添加到字典中，并赋予索引
                    idx_2_word_list.append(tmp_word)  # 将词添加到列表中
                    idx_cnt += 1  # 索引计数器加1
    one_hot_dict_len = len(word_2_idx_dict)  # 获取字典的长度
    word_2_idx_dict.update({"<PAD>": one_hot_dict_len, "<BOS>": one_hot_dict_len + 1, "<EOS>": one_hot_dict_len + 2})  # 添加特殊符号到字典中
    idx_2_word_list += ["<PAD>", "<BOS>", "<EOS>"]  # 添加特殊符号到列表中
    one_hot_dict_len += 3  # 字典长度加3
    source_word_2_idx, target_word_2_idx = word_2_idx_dict, word_2_idx_dict  # 将字典赋值给源词和目标词字典
    source_idx_2_word, target_idx_2_word = idx_2_word_list, idx_2_word_list  # 将列表赋值给源索引和目标索引列表
    source_corpus_len, target_corpus_len = one_hot_dict_len, one_hot_dict_len  # 获取源语料和目标语料的长度
    # dataloader
    dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx)  # 初始化数据集
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_alignment)  # 初始化数据加载器
    # 模型初始化
    model = Seq2Seq(dim_encoder_embbeding,
                    dim_encoder_hidden,
                    source_corpus_len,
                    dim_decoder_embedding,
                    dim_decoder_hidden,
                    target_corpus_len)  # 初始化seq2seq模型
    model = model.to(device)  # 将模型移动到指定设备
    # 模型训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 初始化Adam优化器
    losses = []  # 初始化损失列表
    for epoch in range(num_epochs):  # 遍历每个训练轮次
        for step, (src_index, tgt_index) in enumerate(dataloader):  # 遍历数据加载器中的每个批次
            loss = model(src_index, tgt_index)  # 计算模型的损失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
        losses.append(loss)  # 将损失添加到损失列表中
        print("epoch: {}, training loss: {:.5f}".format(epoch + 1, loss))  # 打印当前轮次的损失

    # 画图
    plt.figure()  # 创建一个新的图形窗口
    plt.plot(np.array([i + 1 for i in range(num_epochs)]), [l.detach().cpu().numpy() for l in losses], "b-")# 绘制训练损失曲线，x轴为训练的轮次，y轴为损失值，"b-"表示蓝色实线
    plt.legend()  # 添加图例
    plt.xlabel("Epoch")  # 设置x轴标签为"Epoch"
    plt.ylabel("Training Loss")  # 设置y轴标签为"Training Loss"
    plt.title("Training Loss of Seq2Seq")  # 设置图形标题为"Training Loss of Seq2Seq"
    plt.savefig("./training_loss_Seq2Seq.png")  # 将图形保存为名为"training_loss_Seq2Seq.png"的图片

    # 生成句子
    model.eval()
    with torch.no_grad():
        with open(output_file_path, "a", encoding="utf-8") as f:
            for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
                tmp_generated_sentence = generate_sentence(tmp_src_sentence)
                f.write("----------------Result {}----------------\n".format(idx + 1))
                f.write("Source sentence: {}\n".format(tmp_src_sentence))
                f.write("True target sentence: {}\n".format(tmp_gt_sentence))
                f.write("Generated target sentence: {}\n".format(tmp_generated_sentence))
                f.write("\n")


