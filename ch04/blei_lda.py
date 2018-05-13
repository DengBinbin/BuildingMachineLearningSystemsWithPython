# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# from __future__ import print_function
import os
import sys
try:
    from gensim import corpora, models, matutils
except:
    print("import gensim failed.")
    print()
    print("Please install it")
    raise
import matplotlib.pyplot as plt
import numpy as np

from wordcloud import create_cloud

NUM_TOPICS = 100
os.chdir(os.path.dirname(sys.argv[0]))
# ROOT_DIR = (os.path.dirname(__file__))

# Check that data exists
if not os.path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')
    print('Please cd into ./data & run ./download_ap.sh')

# Load the data
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# 构建主题模型
model = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=None)
# topics中的每一个元素为某篇文档的主题分布[(topic_index, topic_weight)]
topics = [model[c] for c in corpus]
print(topics[-1])
# 遍历所有的主题
for ti in range(model.num_topics):
    #words为某一个主题的词语分布,64为显示该主题的topn词语
    words = model.show_topic(ti, topn=64)
    #tf为某一主题的topn词概率和，如体育主题中有（篮球，0.3），（足球，0.22），（乒乓球，0.12）
    # 如果topn为词表大小，则tf为1
    tf = sum(f for _, f in words)
    #将结果写入文件
    with open('topics.txt', 'a') as output:
        output.write('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for w, f in words))
        output.write("\n\n\n")


# We first identify the most discussed topic, i.e., the one with the
# highest total weight
topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
max_topic = weight.argmax()

# 返回最高前64个词
words = model.show_topic(max_topic, 64)

# This function will actually check for the presence of pytagcloud and is otherwise a no-op
create_cloud('cloud_blei_lda.png', words)

num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('Figure_04_01.png')


#对比不同alpha参数生成的主题模型之间的差异，并通过画图对其进行可视化
# ALPHA = 1.0

# model1 = models.ldamodel.LdaModel(
#     corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=ALPHA)
# num_topics_used1 = [len(model1[doc]) for doc in corpus]
# fig,ax = plt.subplots()
# ax.hist([num_topics_used, num_topics_used1], np.arange(42))
# ax.set_ylabel('Nr of documents')
# ax.set_xlabel('Nr of topics')
# # The coordinates below were fit by trial and error to look good
# ax.text(9, 223, r'default alpha')
# ax.text(26, 156, 'alpha=1.0')
# fig.tight_layout()
# fig.savefig('Figure_04_02.png')


