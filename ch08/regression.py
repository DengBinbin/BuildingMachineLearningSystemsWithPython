# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.linear_model import ElasticNetCV
from norm import NormalizePositive
from sklearn import metrics


def predict(train):
    binary = (train > 0)
    reg = ElasticNetCV(fit_intercept=True, alphas=[
                       0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
    norm = NormalizePositive()
    train = norm.fit_transform(train)

    filled = train.copy()
    # 对于用户u
    for u in range(train.shape[0]):
        # curtrain是去掉了用户u的训练集
        curtrain = np.delete(train, u, axis=0)
        bu = binary[u]
        #对于那些打分总数超过5的用户才进行预测
        if np.sum(bu) > 5:
            #输入是其余用户对 用户u打过分的这些电影 的打分，标签是用户u实际的打分
            reg.fit(curtrain[:,bu].T, train[u, bu])
            # 对于用户u没打分的那部分电影进行预测
            filled[u, ~bu] = reg.predict(curtrain[:,~bu].T)
    return norm.inverse_transform(filled)


def main(transpose_inputs=False):
    from load_ml100k import get_train_test
    train,test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T
    filled = predict(train)
    r2 = metrics.r2_score(test[test > 0], filled[test > 0])

    print('R2 score ({} regression): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main() #R2 score (user regression): 30.0%
    main(transpose_inputs=True)#R2 score (movie regression): 30.9%
