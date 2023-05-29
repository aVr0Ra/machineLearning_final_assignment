import sys
import numpy as np
import pandas as pd
from Node import Node

class CART:
    def __init__(self,Train,features,label,tree_type,feature_name):
        self.Train=Train #训练集
        self.features=features #特征
        self.label=label #标签
        if tree_type=="c":
            self.root=self.create_classification_tree(Train)
        else:
            self.root=self.create_regression_tree(Train)
        self.feature_name=feature_name
        self.roots=[] #存放子树序列
        #self.roots.append(self.root)#首先将树的初始形式放入roots中
        self.a=sys.maxsize
        self.minnode=None
    def is_one_class(self, data, label):  # 判断data中的数据是否属于同一类 并返回类数组
        X = data[:, label:label + 1]
        labels = []
        for i in range(X.shape[0]):
            if X[i][0] not in labels:
                labels.append(X[i][0])
        if len(labels) == 1:
            return True
        else:
            return False
    def cost(self,data):
        X=data[:,self.label:self.label+1]
        X=X.astype(int)
        avg = np.sum(X)/X.shape[0] #平均值
        sum = 0
        for i in range(X.shape[0]):
           sum+= (X[i][0]-avg)**2
        return sum
    def create_regression_tree(self,data): #生成一棵回归树
        node=None
        min_cost=sys.maxsize
        min_feature=0
        min_split=0
        min_left=None
        min_right=None
        if self.is_one_class(data, self.label):  # 如果数据都属于一类
            node = Node(data, None,None, self.label, 0 ,self.max_num_class(data, self.label))  # 叶子结点
        elif len(self.features) == 0:  # 没有可供划分的特征了
            node = Node(data, None, None, self.label, 0, self.max_num_class(data, self.label))  # 叶子结点
        else:
            for i in self.features: #遍历所有特征取值
                d=data[np.argsort(data[:, i])] #以这个特征的大小来进行排序
                for j in range(d.shape[0]-1):#以特征j的每一个取值来分割成两个区域
                   left = d[0:j+1, :]
                   right=d[j+1:, ]
                   left_cost=self.cost(left)
                   right_cost=self.cost(right)
                   if left_cost+right_cost < min_cost:
                       min_cost=left_cost+right_cost
                       min_feature=i
                       min_split=d[j][i]
                       min_left=left
                       min_right=right
            self.features.remove(min_feature)
            left_node= self.create_regression_tree(min_left)
            right_node = self.create_regression_tree(min_right)
            node= Node(data, left_node,right_node, min_feature, min_split ,0)
        return node
    def class_num(self, data, feature):  # 对于某个特征而言 他有多少种取值
        X = data[: ,feature:feature + 1]
        fea_values = {}
        for i in range(X.shape[0]):
            # print(X[i])
            if X[i][0] not in fea_values:
                fea_values[X[i][0]] = 1
            else:
                fea_values[X[i][0]] += 1
        return fea_values

    def cal_gini(self,data):#计算一个数据的基尼指数
        label=self.class_num(data,self.label)#求每个类有多少条数据
        gini=0
        for k,v in label.items():
            p=v/data.shape[0]
            gini+=p*(1-p)
        return gini

    def basefeature_cal_gini(self,data,feature): #求某一个特征的基尼指数
        fea_values=self.class_num(data, feature)
        min_gini=sys.maxsize
        min_value=None
        min_left=None
        min_right=None
        for k in fea_values.keys(): #对feature的每一个取值计算gini系数
            d1=data[(data[:, feature] == k),: ] #特征feature 取值是k的集合
            d2=data[(data[:, feature] != k),: ]#特征feature 取值不是k 的集合
            r1=d1.shape[0]/data.shape[0]
            r2=d2.shape[0]/data.shape[0]
            gini=r1*self.cal_gini(d1)+r2*self.cal_gini(d2)
            if gini<min_gini:
                min_gini=gini
                min_value=k
                min_left=d1
                min_right=d2
        return min_gini,min_value,min_left,min_right

    def create_classification_tree(self,data): #生成一棵分类树
        min_gini=sys.maxsize
        min_split=None
        min_feature=None
        min_left=None
        min_right=None
        if data.shape[0]==0:
            return None
        elif self.is_one_class(data, self.label):  # 如果数据都属于一类
            node = Node(data, None, None, self.label, 0, self.max_num_class(data, self.label))  # 叶子结点
        elif len(self.features) == 0:  # 没有可供划分的特征了
            node = Node(data, None, None, self.label, 0, self.max_num_class(data, self.label))  # 叶子结点
        else:
            for i in self.features:
                gini,value,left,right=self.basefeature_cal_gini(data,i)
                if gini<min_gini:
                    min_gini=gini
                    min_split=value
                    min_feature=i
                    min_left=left
                    min_right=right
            self.features.remove(i)
            left_node = self.create_regression_tree(min_left)
            right_node = self.create_regression_tree(min_right)
            node = Node(data, left_node, right_node, min_feature, min_split, 0)
        return node

    def max_num_class(self, data, label):  # 返回取值最多的那一类
        # X = data[:, label:label + 1]
        labels = self.class_num(data, label)
        max_num = 0
        max_class = 0
        for k, v in labels.items():
            if v > max_num:
                max_num = v
                max_class = k
        return max_class
    def error(self,data):
        fea_value=self.class_num(data,self.label)
        max_class=self.max_num_class(data,self.label)
        err=0
        for k,v in fea_value.items():
            if k!=max_class:
                err+= (v/data.shape[0])*(data.shape[0]/self.Train.shape[0])
        return err
    def copy_tree(self,node):
        copy_node=None
        if node==None:
            copy_node=None
        else:
            left=self.copy_tree(node.left)
            right=self.copy_tree(node.right)
            copy_node=Node(node.data,left,right,node.feature,node.split,node.value)
        return copy_node

    def need_prune(self, node):
        if node.left.feature != self.label or node.right.feature != self.label:  # 还有内部结点
            return True
        else:
            return False

    def prune(self):
        root=self.root
        now_tree = self.copy_tree(root)
        self.roots.append(now_tree)
        while self.need_prune(root):
            self.a=sys.maxsize
            self.minnode=None
            self.cal_prune_what(root)
            self.minnode.left=None
            self.minnode.right=None
            self.minnode.feature=self.label
            self.split=None
            self.value=self.max_num_class(self.minnode.data, self.label)
            now_tree=self.copy_tree(root)
            self.roots.append(now_tree)

    def cal_prune_what(self,node):
        leaf_error=0
        leaf_num=0
        if node.left==None and node.right==None:#如果该结点是叶子结点
            return self.error(node.data),1
        else:
            if node.left!=None:
                left_error,left_num=self.cal_prune_what(node.left)
                leaf_error += left_error
                leaf_num+=left_num
            if node.right!=None:
                right_error,right_num=self.cal_prune_what(node.right)
                leaf_error += right_error
                leaf_num+=right_num
            node_err=self.error(node.data)
            a=(node_err-leaf_error)/(leaf_num-1)
            if node!=self.root:
               if a<self.a:
                  self.a=a
                  self.minnode=node
            return leaf_error,leaf_num

    def pre_order(self, node):
        if node != None:
            if node.left == None and node.right == None:  # 如果是叶子结点
                print(str(node.value) + "\n")
            else:
                print(self.feature_name[node.feature])
                self.pre_order(node.left)
                self.pre_order(node.right)

    def level_order(self):
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            if node.left == None and node.right == None:
                print(node.value)
            else:
                print(self.feature_name[node.feature])
                queue.append(node.left)
                queue.append(node.right)
    def fit(self,node,X):
        while node.left!=None and node.right!=None:# 如果不是叶子结点
            feature=node.feature
            split=node.split
            if X[feature] == split:
                node=node.left
            else:
                node=node.right
        print("X的预测结果")
        print(node.value)
