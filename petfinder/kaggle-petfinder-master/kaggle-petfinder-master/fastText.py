# 导入 fastText 库
import fastText as ft
import pandas as pd


# 使用 fastText 提取文本特征
def fastText_vectorize(X, col, textfile, dimension):
    # 训练无监督的 fastText 模型
    clf = ft.train_unsupervised(input=textfile,
                                ws=5, minCount=10, epoch=10,
                                minn=3, maxn=6, wordNgrams=3,
                                dim=dimension, thread=1)

    fastText_text_features = []
    # 遍历列中的每个描述
    for description in col:
        # 将每个描述转换为向量并添加到特征列表中
        fastText_text_features.append(clf.get_sentence_vector(description))

    # 将特征列表转换为 DataFrame
    fastText_text_features = pd.DataFrame(fastText_text_features)
    # 为特征列添加前缀
    fastText_text_features = fastText_text_features.add_prefix('fastText_Description_'.format(i))

    # 返回生成的特征 DataFrame
    return fastText_text_features


# 使用 fastText 提取文本特征
def fastText_vectorize(X, col, textfile, dimension):
    # 训练无监督的 fastText 模型
    clf = ft.train_unsupervised(input=textfile,
                                ws=5, minCount=10, epoch=10,
                                minn=3, maxn=6, wordNgrams=3,
                                dim=dimension, thread=1)

    fastText_text_features = []
    # 遍历列中的每个描述
    for description in col:
        # 将每个描述转换为向量并添加到特征列表中
        fastText_text_features.append(clf.get_sentence_vector(description))

    # 将特征列表转换为 DataFrame
    fastText_text_features = pd.DataFrame(fastText_text_features)
    # 为特征列添加前缀
    fastText_text_features = fastText_text_features.add_prefix('fastText_Description_'.format(i))

    # 返回生成的特征 DataFrame
    return fastText_text_features
