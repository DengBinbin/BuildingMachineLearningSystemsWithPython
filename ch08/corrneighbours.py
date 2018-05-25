# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np
from load_ml100k import get_train_test
from scipy.spatial import distance
from sklearn import metrics

from norm import NormalizePositive

def predict(otrain):
    #拿到打过分的index
    binary = (otrain > 0)
    norm = NormalizePositive(axis=1)
    #归一化
    train = norm.fit_transform(otrain)
    #计算用户打分之间的相似度，这里是只关心打分与否，而不关心多少分？
    dists = distance.pdist(binary, 'correlation')
    dists = distance.squareform(dists)
    #对dists进行排序，axis=1代表对行排序，每一行代表从近到远对应列下标，如 2 0 1代表第二列是距离最近，第0列其次..
    neighbors = dists.argsort(axis=1)
    filled = train.copy()
    for u in range(filled.shape[0]):
        # n_u 是第u个用户的邻居
        n_u = neighbors[u, 1:]
        #m是电影
        for m in range(filled.shape[1]):
            # This code could be faster using numpy indexing trickery as the
            # cost of readibility (this is left as an exercise to the reader):
            #对于用户u的邻居n_u，如果他对电影m打分了，则进入到revs列表中
            revs = [train[neigh, m]
                    for neigh in n_u
                    if binary[neigh, m]]
            #将revs列表中的前一半的均值作为 用户u对电影m的预测
            if len(revs):
                n = len(revs)
                n //= 2
                n += 1
                revs = revs[:n]
                filled[u,m] = np.mean(revs)

    return norm.inverse_transform(filled)

def main(transpose_inputs=False):
    train, test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test  = test.T

    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score (binary {} neighbours): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main() #R2 score (binary user neighbours): 28.5%
    main(transpose_inputs=True)#R2 score (binary movie neighbours): 29.8%
