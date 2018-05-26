# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from collections import defaultdict
from itertools import chain
from gzip import GzipFile
minsupport = 280

dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]

counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1

# Only elements that have at least minsupport should be considered.
#valid为所有的频繁单项
valid = set(el for el, c in counts.items() if (c >= minsupport))

# Filter the dataset to contain only valid elements
# (This step is not strictly necessary, but will make the rest of the code
# faster as the itemsets will be smaller):
#对于每一个购物篮，将其中不属于频繁项的那些商品去除
dataset = [[el for el in ds if (el in valid)] for ds in dataset]

# Convert to frozenset for fast processing
dataset = [frozenset(ds) for ds in dataset]

itemsets = [frozenset([v]) for v in valid]
freqsets = itemsets[:]
for i in range(2):
    print("At iteration {}, number of frequent baskets: {}".format(
        i, len(itemsets)))
    nextsets = []

    tested = set()
    #it为每一个频繁项集，v为每一个频繁单项
    for it in itemsets:
        for v in valid:
            #如果频繁单项不在频繁项集中，就将该单项和频繁项集合并
            if v not in it:
                # Create a new candidate set by adding v to it
                c = (it | frozenset([v]))

                # Check if we have tested it already:
                if c in tested:
                    continue
                tested.add(c)

                # Count support by looping over dataset
                # This step is slow.
                # Check `apriori.py` for a better implementation.
                #对于每一个购物篮d，如果其包含了新增加的频繁项集c，则+1，最后求和；
                # 如果和大于最小支持度就加入到长度更长的项集合
                support_c = sum(1 for d in dataset if d.issuperset(c))
                if support_c > minsupport:
                    nextsets.append(c)
    freqsets.extend(nextsets)
    itemsets = nextsets
    if not len(itemsets):
        break
print("Finished!")


def rules_from_itemset(itemset, dataset, minlift=1.):
    nr_transactions = float(len(dataset))
    for item in itemset:
        consequent = frozenset([item])
        antecedent = itemset-consequent
        #base：后项的计数
        base = 0.0
        # acount: antecedent count 前项
        acount = 0.0

        # ccount : consequent count 前项+后项
        ccount = 0.0
        #d是一个购物篮,item是一个单项
        for d in dataset:
          if item in d: base += 1
          if d.issuperset(itemset): ccount += 1
          if d.issuperset(antecedent): acount += 1
        print(base,acount,ccount)
        import sys
        sys.exit(1)
        base /= nr_transactions
        p_y_given_x = ccount/acount
        lift = p_y_given_x / base
        if lift > minlift:
            print('Rule {0} ->  {1} has lift {2}'
                  .format(antecedent, consequent,lift))

for itemset in freqsets:
    if len(itemset) > 1:
        rules_from_itemset(itemset, dataset, minlift=4.)
