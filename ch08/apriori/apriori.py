# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from collections import namedtuple


def apriori(dataset, minsupport, maxsize):
    '''
    freqsets, support = apriori(dataset, minsupport, maxsize)

    Parameters
    ----------
    dataset : sequence of sequences
        input dataset
    minsupport : int
        Minimal support for frequent items
    maxsize : int
        Maximal size of frequent items to return

    Returns
    -------
    freqsets : sequence of sequences
    support : dictionary
        This associates each itemset (represented as a frozenset) with a float
        (the support of that itemset)
    '''
    from collections import defaultdict
    # baskets和pointers都是字典， key是商品，value是包含该商品的购物篮，区别是一个key是int型，一个key是fronzenset，
    # why?baskets存储的是所有的频繁商品集，而pointer存储的是所有的单项频繁商品
    baskets = defaultdict(list)
    pointers = defaultdict(list)

    for i, ds in enumerate(dataset):
        for ell in ds:
            pointers[ell].append(i)
            baskets[frozenset([ell])].append(i)
    # Convert pointer items to frozensets to speed up operations later
    new_pointers = dict()
    for k in pointers:
        if len(pointers[k]) >= minsupport:
            new_pointers[k] = frozenset(pointers[k])
    pointers = new_pointers
    for k in baskets:
        baskets[k] = frozenset(baskets[k])


    # Valid are all elements whose support is >= minsupport
    #valid是所有支持度超过最小支持度的商品
    valid = set()
    for el, c in baskets.items():
        if len(c) >= minsupport:
            valid.update(el)

    # Itemsets at first iteration are simply all singleton with valid elements:
    itemsets = [frozenset([v]) for v in valid]
    freqsets = []
    for i in range(maxsize - 1):
        print("At iteration {}, number of frequent baskets: {}".format(
            i, len(itemsets)))
        #newsets是新的高频商品集
        newsets = []
        for it in itemsets:
            #it是大于最小支持度的商品集
            ccounts = baskets[it]
            for v, pv in pointers.items():
                #对于大于最小值尺度的商品v，如果其不在it中，则计算包含其的购物篮pv与包含商品it的购物篮ccount之间的交集 csup
                if v not in it:
                    csup = (ccounts & pv)
                    #如果交集大于最小支持度，将商品V加入到商品集it中
                    if len(csup) >= minsupport:
                        new = frozenset(it | frozenset([v]))
                        #如果新的商品集new不在购物篮集合中
                        if new not in baskets:
                            newsets.append(new)
                            baskets[new] = csup
        freqsets.extend(itemsets)
        itemsets = newsets
        #如果没有新的商品集合产生，则结束
        if not len(itemsets):
            break
    support = {}
    for k in baskets:
        support[k] = float(len(baskets[k]))
    return freqsets, support


# A namedtuple to collect all values that may be interesting
AssociationRule = namedtuple('AssociationRule', ['antecendent', 'consequent', 'base', 'py_x', 'lift'])

def association_rules(dataset, freqsets, support, minlift):
    '''
    for assoc_rule in association_rules(dataset, freqsets, support, minlift):
        ...

    This function takes the returns from ``apriori``.

    Parameters
    ----------
    dataset : sequence of sequences
        input dataset
    freqsets : sequence of sequences
    support : dictionary
    minlift : int
        minimal lift of yielded rules

    Returns
    -------
    assoc_rule : sequence of AssociationRule objects
    '''
    #nr_transactions是购物篮的个数
    nr_transactions = float(len(dataset))
    print(nr_transactions)
    #freqsets是项数大于1的频繁项集
    freqsets = [f for f in freqsets if len(f) > 1]
    for fset in freqsets:
        #fset是频繁项集，f为其中的某一个单项
        for f in fset:
            consequent = frozenset([f])
            #antecendent为频繁项集减去当前项f
            antecendent = fset - consequent
            #待求项lift是指的提升度，它是通过计算p(y|x)/p(y)得出
            #p(y|x)是交易中同时包含y和x的比例；p(y)是所有交易中包含y的比例
            py_x = support[fset] / support[antecendent]
            base = support[consequent] / nr_transactions
            lift = py_x / base
            #如果提升度大于最小提升度，将其记录到迭代器中
            if lift > minlift:
                yield AssociationRule(antecendent, consequent, base, py_x, lift)

