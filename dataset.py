import numpy as np
import pandas as pd
from main import CART

datasets = np.array([['尚可', '短', '否', '一般', 0],
    ['尚可', '短', '否', '好', 0],
    ['尚可', '长', '否', '好', 1],
    ['尚可', '长', '是', '一般', 1],
    ['尚可', '短', '否', '一般', 0],
    ['不好', '短', '否', '一般', 0],
    ['不好', '短', '否', '好', 0],
    ['不好', '长', '是', '好', 1],
    ['不好', '短', '是', '非常好', 1],
    ['不好', '短', '是', '非常好', 1],
    ['疯魔', '短', '是', '非常好', 1],
    ['疯魔', '短', '是', '好', 1],
    ['疯魔', '长', '否', '好', 1],
    ['疯魔', '长', '否', '非常好', 1],
    ['疯魔', '短', '否', '一般', 1],
    ['一般', '短', '否', '一般', 1]])
datasets[:, 4].astype(int)

features = [0, 1, 2, 3]
feature_name = ['精神状态', '工龄', '有自己房子', '绩效情况']
label = 4
cart = CART(datasets, features, label, "c", feature_name)
cart.prune()

# Debug
# for root in cart.roots:
#    cart.pre_order(cart.root)
#    print("*******")
# cart.pre_order(cart.root)
# DT.prune(DT.root)
# DT.pre_order(DT.root)
# print("*******")
# DT.level_order()

testNumber = int(input("请输入需要测试的组数："))
for i in range(1 , testNumber + 1):
    print("第 #" + str(i) + " 组测试样例：")
    cond = input("请输入精神状态：（尚可/不好/疯魔）")
    years = input("请输入工龄：（长/短）")
    house = input("请输入是否有房子：（是/否）")
    work = input("请输入绩效情况：（一般/好/非常好）：")
    X = np.array([cond, years, house, work])
    cart.fit(cart.roots[0], X)

