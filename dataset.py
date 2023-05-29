import numpy as np
import pandas as pd
from main import CART

datasets = np.array([['青年', '否', '否', '一般', 0],
                     ['青年', '否', '否', '好', 0],
                     ['青年', '是', '否', '好', 1],
                     ['青年', '是', '是', '一般', 1],
                     ['青年', '否', '否', '一般', 0],
                     ['中年', '否', '否', '一般', 0],
                     ['中年', '否', '否', '好', 0],
                     ['中年', '是', '是', '好', 1],
                     ['中年', '否', '是', '非常好', 1],
                     ['中年', '否', '是', '非常好', 1],
                     ['老年', '否', '是', '非常好', 1],
                     ['老年', '否', '是', '好', 1],
                     ['老年', '是', '否', '好', 1],
                     ['老年', '是', '否', '非常好', 1],
                     ['老年', '否', '否', '一般', 1],
                     ['青年', '否', '否', '一般', 1]])
datasets[:,4].astype(int)
print(datasets[:,4].dtype)
features=[0,1,2,3]
feature_name=['年龄','有工作','有自己房子','信用情况']
label=4
cart=CART(datasets,features,label,"c",feature_name)
print("*******")
cart.prune()

for root in cart.roots:
    cart.pre_order(root)
    print("*******")

cart.pre_order(cart.root)
# DT.prune(DT.root)
# DT.pre_order(DT.root)
# print("*******")
# DT.level_order()
X=np.array(  ['中年', '否', '是', '非常好'])
cart.fit(cart.roots[0],X)

