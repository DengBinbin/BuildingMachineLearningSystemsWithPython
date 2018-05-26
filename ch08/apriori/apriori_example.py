# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from apriori import apriori, association_rules
from gzip import GzipFile

# 读入数据，数据格式为二维列表，[[1,3],[22,1]]，第一维代表购物篮，第二维代表商品
dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]
freqsets, support = apriori(dataset, 80, maxsize=16)
rules = list(association_rules(dataset, freqsets, support, minlift=30.0))
rules.sort(key=(lambda ar: ar.lift),reverse=True)
for ar in rules:
    print('{} -> {} (lift = {:.4})'
          .format(set(ar.antecendent),
                    set(ar.consequent),
                    ar.lift))
