
class Node:
    def __init__(self,data,left,right,feature,split,value):
     self.data=data#这个结点包含的数据
     self.left=left #左孩子
     self.right=right #右孩子
     self.feature=feature #这个结点是用哪个特征进行划分的
     self.split=split # 这个结点的划分点值是多少
     self.value=value #这个结点属于哪一类 只有叶子结点才有值