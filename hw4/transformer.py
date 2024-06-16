import numpy as np  # 导入numpy库，用于数值计算
import random  # 导入random库，用于随机操作
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于数据可视化
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch中的神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器模块
import math  # 导入math库，用于数学运算

class CorpusDataset(Dataset):  # 定义一个数据集类，用于处理源语言和目标语言数据，并且包括了批量数据对齐的功能
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx, device):
        self.source_data = source_data  # 保存源数据
        self.target_data = target_data  # 保存目标数据
        self.source_word_2_idx = source_word_2_idx  # 保存源词汇到索引的映射
        self.target_word_2_idx = target_word_2_idx  # 保存目标词汇到索引的映射
        self.device = device  # 保存设备信息

    def __getitem__(self, index):
        src = self.source_data[index]  # 获取指定索引的源数据
        tgt = self.target_data[index]  # 获取指定索引的目标数据
        src_index = [self.source_word_2_idx[i] for i in src]  # 将源数据转换为索引
        tgt_index = [self.target_word_2_idx[i] for i in tgt]  # 将目标数据转换为索引
        return src_index, tgt_index  # 返回转换后的索引

    def batch_data_alignment(self, batch_datas):
        global device  # 声明全局变量device
        src_index, tgt_index = [], []  # 初始化源数据和目标数据索引列表
        src_len, tgt_len = [], []  # 初始化源数据和目标数据长度列表
        for src, tgt in batch_datas:
            src_index.append(src)  # 添加源数据索引
            tgt_index.append(tgt)  # 添加目标数据索引
            src_len.append(len(src))  # 添加源数据长度
            tgt_len.append(len(tgt))  # 添加目标数据长度
        src_max_len = max(src_len)  # 获取源数据的最大长度
        tgt_max_len = max(tgt_len)  # 获取目标数据的最大长度
        src_index = [[self.source_word_2_idx["<BOS>"]] +
                    src +
                    [self.source_word_2_idx["<PAD>"]] * (src_max_len - len(src))
                    for src in src_index]  # 对源数据进行填充
        clipped_tgt_index = [[self.target_word_2_idx["<BOS>"]] +
                        tgt[:tgt_max_len-2] +
                        [self.target_word_2_idx["<EOS>"]]
                        for tgt in tgt_index if len(tgt) + 2 > tgt_max_len]  # 对超长目标数据进行截断
        normal_tgt_index = [[self.target_word_2_idx["<BOS>"]] +
                        tgt +
                        [self.target_word_2_idx["<EOS>"]] +
                        [self.target_word_2_idx["<PAD>"]] * (tgt_max_len - len(tgt) - 2)
                        for tgt in tgt_index if len(tgt) + 2 <= tgt_max_len]  # 对短目标数据进行填充
        tgt_index = clipped_tgt_index + normal_tgt_index  # 合并截断和填充后的目标数据
        src_index = torch.tensor(src_index, dtype=torch.long, device=device)  # 转换为张量并移动到设备上
        tgt_index = torch.tensor(tgt_index, dtype=torch.long, device=device)  # 转换为张量并移动到设备上
        return src_index, tgt_index  # 返回对齐后的数据

    def __len__(self):
        assert len(self.source_data) == len(self.target_data)  # 确保源数据和目标数据长度相等
        return len(self.target_data)  # 返回数据集的长度

class PositionalEncoding(nn.Module):  # 定义位置编码和Transformer模型
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 定义Dropout层
        position = torch.arange(max_len).unsqueeze(1)  # 创建位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 计算位置编码的分母
        pe = torch.zeros(max_len, 1, d_model)  # 初始化位置编码矩阵
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 计算位置编码的sin部分
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 计算位置编码的cos部分
        self.register_buffer('pe', pe)  # 注册位置编码缓冲区

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # 将位置编码加到输入上
        return self.dropout(x)  # 应用Dropout层

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size  # 保存嵌入维度
        self.dropout_layer = nn.Dropout(p=dropout)  # 定义Dropout层
        self.encoder = nn.Embedding(vocab_size, embed_size)  # 定义嵌入层
        self.pos_encoder = PositionalEncoding(embed_size, dropout)  # 定义位置编码层
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)  # 定义Transformer模型
        self.decoder = nn.Linear(embed_size, vocab_size)  # 定义解码层

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 创建上三角矩阵掩码
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 将掩码中的0填充为负无穷，将1填充为0
        return mask  # 返回生成的掩码

    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)  # 生成源数据掩码并移动到设备上
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # 生成目标数据掩码并移动到设备上
        src = self.encoder(src) * math.sqrt(self.embed_size)  # 对源数据进行嵌入并乘以嵌入维度的平方根
        src = self.dropout_layer(src)  # 应用Dropout层
        src = self.pos_encoder(src)  # 添加位置编码
        tgt = self.encoder(tgt) * math.sqrt(self.embed_size)  # 对目标数据进行嵌入并乘以嵌入维度的平方根
        tgt = self.pos_encoder(tgt)  # 添加位置编码
        output = self.transformer(src, tgt, src_mask, tgt_mask)  # 通过Transformer模型进行前向传播
        output = self.decoder(output)  # 通过解码层生成输出
        return output  # 返回模型输出

d_model = 256  # 模型的维度设置为256

def generate_sentence_transformer(sentence, model, max_len=40):  # 定义生成句子函数，输入句子、模型和最大长度（默认40）
    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)  # 将输入句子转换为索引并转为张量
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        memory = model.encoder(src_index) * math.sqrt(d_model)  # 通过编码器并进行缩放
        memory = model.pos_encoder(memory)  # 通过位置编码器处理
        outs = [target_word_2_idx["<BOS>"]]  # 初始化输出序列，包含起始标记
        tgt_tensor = torch.tensor([outs], device=device)  # 将初始输出序列转换为张量
        for i in range(max_len - 1):  # 迭代生成每个单词，最多 max_len 次
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)  # 生成目标序列掩码
            tgt_tensor_with_pe = model.pos_encoder(model.encoder(tgt_tensor) * math.sqrt(d_model))  # 目标序列位置编码
            out = model.transformer(memory, tgt_tensor_with_pe, tgt_mask=tgt_mask)  # 使用 transformer 生成输出
            next_word_probs = model.decoder(out[:, -1, :])  # 解码输出获取下一个单词的概率
            next_word_idx = next_word_probs.argmax(dim=-1).item()  # 获取概率最高的单词索引
            temperature = 0.6  # 设置温度参数，用于控制多样性
            next_word_probs = next_word_probs / temperature  # 调整概率
            next_word_probs = next_word_probs.softmax(dim=-1)  # 转换为概率分布
            next_word_idx = torch.multinomial(next_word_probs, 1).item()  # 根据概率分布采样下一个单词
            if next_word_idx == target_word_2_idx["<EOS>"]:  # 如果生成了结束标记则停止
                break
            outs.append(next_word_idx)  # 将生成的单词添加到输出序列中
            new_token = torch.tensor([[next_word_idx]], device=device)  # 转换新单词为张量
            tgt_tensor = torch.cat((tgt_tensor, new_token), dim=1)  # 更新目标序列张量
    return "".join([target_idx_2_word[i] for i in outs if i != target_word_2_idx["<BOS>"]])  # 返回生成的句子，忽略起始标记

def check_vocab_construction(word_2_idx_dict, idx_2_word_list):  # 检查词汇表构建过程
    print("Word to Index mapping:")  # 打印词到索引的映射
    for word, idx in word_2_idx_dict.items():  # 遍历词汇表
        print(f"Word: {word}, Index: {idx}")  # 打印词和对应的索引
    print("\nIndex to Word mapping:")  # 打印索引到词的映射
    for idx, word in enumerate(idx_2_word_list):  # 遍历索引表
        print(f"Index: {idx}, Word: {word}")  # 打印索引和对应的词

def check_dataset_indexing(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx):  # 检查数据集的索引转换
    for i in range(5):  # 仅检查前5个样本
        src_sentence = source_corpus[i]  # 获取源句子
        tgt_sentence = target_corpus[i]  # 获取目标句子
        src_index = [source_word_2_idx[word] for word in src_sentence]  # 将源句子转换为索引
        tgt_index = [target_word_2_idx[word] for word in tgt_sentence]  # 将目标句子转换为索引
        print(f"Source Sentence: {src_sentence}")  # 打印源句子
        print(f"Source Indexes: {src_index}")  # 打印源句子索引
        print(f"Target Sentence: {tgt_sentence}")  # 打印目标句子
        print(f"Target Indexes: {tgt_index}\n")  # 打印目标句子索引

def check_dataloader(dataloader):  # 检查数据加载器
    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):  # 遍历数据加载器中的批次
        if batch_idx == 0:  # 仅检查第一个批次
            print(f"Batch {batch_idx + 1}:")  # 打印批次编号
            print(f"Source Batch Shape: {src_batch.shape}")  # 打印源批次的形状
            print(f"Target Batch Shape: {tgt_batch.shape}")  # 打印目标批次的形状
            print(f"Source Batch: {src_batch}")  # 打印源批次数据
            print(f"Target Batch: {tgt_batch}\n")  # 打印目标批次数据
        else:
            break  # 仅检查一个批次后跳出循环

def check_special_tokens_application(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx):  # 检查特殊标记的应用
    for i in range(5):  # 仅检查前5个样本
        src_sentence = source_corpus[i]  # 获取源句子
        tgt_sentence = target_corpus[i]  # 获取目标句子
        src_index = [source_word_2_idx["<BOS>"]] + [source_word_2_idx[word] for word in src_sentence] + [source_word_2_idx["<PAD>"]]  # 添加起始和填充标记
        tgt_index = [target_word_2_idx["<BOS>"]] + [target_word_2_idx[word] for word in tgt_sentence] + [target_word_2_idx["<EOS>"]]  # 添加起始和结束标记
        print(f"Source Sentence: {src_sentence}")  # 打印源句子
        print(f"Source Indexes with Special Tokens: {src_index}")  # 打印带有特殊标记的源句子索引
        print(f"Target Sentence: {tgt_sentence}")  # 打印目标句子
        print(f"Target Indexes with Special Tokens: {tgt_index}\n")  # 打印带有特殊标记的目标句子索引


if __name__ == '__main__':  # 如果此模块是主程序入口点，则执行以下代码
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 选择设备，如果有GPU则用GPU，否则用CPU
    batch_size = 2  # 批处理大小为2
    num_corpus = 300  # 语料库的数量为300
    num_test_corpus = 10  # 测试语料库的数量为10
    txt_file_path = "jyxstxtqj_downcc.com/天龙八部.txt"  # 设置文本文件路径
    num_epochs = 50  # 训练的轮数为50
    lr = 0.001  # 学习率为0.001
    dim_encoder_embedding = 256  # 编码器嵌入层的维度为256
    dim_encoder_hidden = 512  # 编码器隐藏层的维度为512
    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"  # 要被替换的字符集合
    source_target_corpus_ori = []  # 原始的源-目标语料对列表
    with open(txt_file_path, "r", encoding="gbk", errors="ignore") as tmp_file:  # 打开文本文件，忽略编码错误
        tmp_file_context = tmp_file.read()  # 读取文件内容
        for tmp_char in char_to_be_replaced:  # 遍历要被替换的字符
            tmp_file_context = tmp_file_context.replace(tmp_char, "")  # 替换字符为空
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")  # 去除广告
        tmp_file_sentences = tmp_file_context.split("。")  # 按句号分割成句子列表
        for tmp_idx, tmp_sentence in enumerate(tmp_file_sentences):  # 遍历句子列表
            if ("她" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):  # 筛选包含“她”的句子且长度在10到40之间的句子对
                source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))  # 将符合条件的句子对加入列表
    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)  # 从原始语料对列表中随机抽取300个索引
    source_corpus, target_corpus = [], []  # 初始化源语料和目标语料列表
    for idx in sample_indexes:  # 遍历抽取的索引
        source_corpus.append(source_target_corpus_ori[idx][0])  # 添加源句子
        target_corpus.append(source_target_corpus_ori[idx][1])  # 添加目标句子
    test_corpus = []  # 初始化测试语料列表
    for idx in range(len(source_target_corpus_ori)):  # 遍历所有原始语料对
        if idx not in sample_indexes:  # 排除已抽取的索引
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))  # 添加未抽取的语料对
    test_corpus = random.sample(test_corpus, num_test_corpus)  # 从未抽取的语料对中随机抽取10个作为测试语料
    test_source_corpus, test_target_corpus = [], []  # 初始化测试源语料和测试目标语料列表
    for tmp_src, tmp_tgt in test_corpus:  # 遍历测试语料对
        test_source_corpus.append(tmp_src)  # 添加测试源句子
        test_target_corpus.append(tmp_tgt)  # 添加测试目标句子
    # one-hot编码字典
    idx_cnt = 0  # 初始化索引计数器
    word_2_idx_dict = dict()  # 初始化单词到索引的字典
    idx_2_word_list = list()  # 初始化索引到单词的列表
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:  # 遍历所有语料
        for tmp_sentence in tmp_corpus:  # 遍历每个句子
            for tmp_word in tmp_sentence:  # 遍历每个字
                if tmp_word not in word_2_idx_dict.keys():  # 如果字不在字典中
                    word_2_idx_dict[tmp_word] = idx_cnt  # 将字添加到字典中，并赋予索引
                    idx_2_word_list.append(tmp_word)  # 将字添加到索引列表中
                    idx_cnt += 1  # 索引计数器加1
    one_hot_dict_len = len(word_2_idx_dict)  # 获取字典长度
    word_2_idx_dict.update({"<PAD>": one_hot_dict_len, "<BOS>": one_hot_dict_len + 1, "<EOS>": one_hot_dict_len + 2})  # 更新字典，添加特殊符号
    idx_2_word_list += ["<PAD>", "<BOS>", "<EOS>"]  # 更新索引列表，添加特殊符号
    one_hot_dict_len += 3  # 字典长度加3
    source_word_2_idx, target_word_2_idx = word_2_idx_dict, word_2_idx_dict  # 源和目标的字到索引的字典
    source_idx_2_word, target_idx_2_word = idx_2_word_list, idx_2_word_list  # 源和目标的索引到字的列表
    source_corpus_len, target_corpus_len = one_hot_dict_len, one_hot_dict_len  # 源和目标的语料长度
    # dataloader
    dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx, device)  # 创建数据集
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_alignment)  # 创建数据加载器
    # 模型初始化
    transformer_model = TransformerModel(
        vocab_size=one_hot_dict_len,  # 词汇量大小
        embed_size=dim_encoder_embedding,  # 嵌入层维度
        num_heads=8,  # 多头注意力机制的头数
        num_layers=2,  # 编码器和解码器的层数
        hidden_dim=dim_encoder_hidden,  # 隐藏层维度
        dropout=0.1  # dropout概率
    ).to(device)  # 将模型移动到指定设备

    #模型训练
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)  # 初始化Adam优化器，传入模型参数和学习率
    losses = []  # 初始化用于存储每个epoch的损失的列表
    for epoch in range(num_epochs):  # 遍历每个训练周期
        for step, (src_index, tgt_index) in enumerate(dataloader):  # 遍历数据加载器中的每个批次
            src_index = src_index.clone().detach().to(device)  # 克隆并分离源索引张量，将其移动到指定设备
            tgt_index = tgt_index.clone().detach().to(device)  # 克隆并分离目标索引张量，将其移动到指定设备
            optimizer.zero_grad()  # 清除优化器的梯度
            output = transformer_model(src_index, tgt_index)  # 将源索引和目标索引传入transformer模型，得到输出
            output = output.permute(1, 0, 2)  # 调整输出张量的维度顺序
            loss = nn.CrossEntropyLoss(ignore_index=word_2_idx_dict["<PAD>"], reduction='mean')(
            output.reshape(-1, one_hot_dict_len), tgt_index.reshape(-1))  # 计算交叉熵损失，忽略PAD索引并取均值
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
        losses.append(loss.item())  # 将当前epoch的损失添加到列表中
        print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))  # 打印当前epoch的损失
    plt.figure()  # 创建一个新的图形
    plt.plot(np.arange(1, num_epochs + 1), losses, "b-")  # 绘制训练损失随epoch变化的折线图
    plt.xlabel("Epoch")  # 设置x轴标签
    plt.ylabel("Training Loss")  # 设置y轴标签
    plt.title("Training Loss of Transformer")  # 设置图表标题
    plt.savefig("./training_loss_transformer.png")  # 将图表保存为图片
    plt.show()  # 显示图表
    #生成句子
    for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
        tmp_generated_sentence = generate_sentence_transformer(tmp_src_sentence, transformer_model)
        print("----------------Result {}----------------".format(idx + 1))
        print("Source sentence: {}".format(tmp_src_sentence))
        print("True target sentence: {}".format(tmp_gt_sentence))
        print("Generated target sentence: {}".format(tmp_generated_sentence))





# 生成句子
with open("transformer_generated_sentences.txt", "w", encoding="utf-8") as file:
    for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
        tmp_generated_sentence = generate_sentence_transformer(tmp_src_sentence, transformer_model)
        print("----------------Result {}----------------".format(idx + 1))
        print("Source sentence: {}".format(tmp_src_sentence))
        print("True target sentence: {}".format(tmp_gt_sentence))
        print("Generated target sentence: {}".format(tmp_generated_sentence))

        # 将结果写入文件
        file.write("----------------Result {}----------------\n".format(idx + 1))
        file.write("Source sentence: {}\n".format(tmp_src_sentence))
        file.write("True target sentence: {}\n".format(tmp_gt_sentence))
        file.write("Generated target sentence: {}\n\n".format(tmp_generated_sentence))
