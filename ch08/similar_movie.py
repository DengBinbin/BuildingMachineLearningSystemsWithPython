# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np


def nn_movie(ureviews, reviews, uid, mid, k=1):
    '''Movie neighbor based classifier

    Parameters
    ----------
    ureviews : ndarray
    reviews : ndarray
    uid : int
        index of user
    mid : int
        index of movie
    k : int
        index of neighbor to return

    Returns
    -------
    pred : float
    '''
    #ureviews是reviews的转置，ureviers[mid]代表第mid部电影的用户向量
    X = ureviews
    y = ureviews[mid].copy()
    y -= y.mean()
    y /= (y.std() + 1e-5)
    #corr是电影矩阵与mid的电影向量之间的点积
    corrs = np.dot(X, y)
    #likes是电影相关系数按照从大到小排序
    likes = corrs.argsort()
    likes = likes[::-1]
    c = 0
    pred = 3.
    for ell in likes:
        if ell == mid:
            continue
        #ell是按照电影m的相似程度进行遍历，如果用户u对电影m最相似的电影ell评分了，那么就返回用户u对ell的评分作为其对m的预测
        if reviews[uid, ell] > 0:
            pred = reviews[uid, ell]
            if c == k:
                return pred
            c += 1
    return pred


def all_estimates(reviews, k=1):
    '''Estimate all review ratings
    '''
    reviews = reviews.astype(float)
    k -= 1
    nusers, nmovies = reviews.shape
    estimates = np.zeros_like(reviews)
    #对于每一个用户u
    for u in range(nusers):
        #删除掉用户u的剩下评论
        ureviews = np.delete(reviews, u, axis=0)
        #归一化
        ureviews -= ureviews.mean(0)
        ureviews /= (ureviews.std(0) + 1e-5)
        ureviews = ureviews.T.copy()
        #m为用户u打分的那些电影，找到与m最接近的电影评分作为m的评分预测
        for m in np.where(reviews[u] > 0)[0]:
            estimates[u, m] = nn_movie(ureviews, reviews, u, m, k)
    return estimates

if __name__ == '__main__':
    from load_ml100k import load
    reviews = load()
    estimates = all_estimates(reviews)
    error = (estimates - reviews)
    error **= 2
    error = error[reviews > 0]
    rmse = np.sqrt(error.mean())
    print("RMSE is {0}.".format(rmse))  #RMSE is 1.2020440923693274.
