class Node:
    def __init__(self, label=None, feature_name=None, feature=None):
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label:': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)


class CART:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 计算Gini指数
    @staticmethod
    def calc_gini(dataset):
        data_length = len(dataset)
        label_count = {}
        for i in range(data_length):
            label = dataset[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        gini = 1
        for key in label_count:
            prob = label_count[key] / data_length
            gini -= prob ** 2
        return gini

    # 计算每个特征的Gini指数
    def calc_conditional_gini(self, dataset, axis, value):
        data_length = len(dataset)
        feature_sets = {}
        for i in range(data_length):
            if dataset[i][axis] not in feature_sets:
                feature_sets[dataset[i][axis]] = []
            feature_sets[dataset[i][axis]].append(dataset[i])
        conditional_gini = 0
        for feature in feature_sets:
            prob = len(feature_sets[feature]) / data_length
            conditional_gini += prob * self.calc_gini(feature_sets[feature])
        return conditional_gini

    # 选择最好的特征
    def choose_best_feature(self, dataset):
        feature_count = len(dataset[0]) - 1
        best_gini = 999999
        best_feature = -1
        for i in range(feature_count):
            gini = self.calc_conditional_gini(dataset, i, dataset[0][i])
            if gini < best_gini:
                best_gini = gini
                best_feature = i
        return best_feature

    # 创建决策树
    def create_tree(self, dataset, feature_names):
        class_list = [data[-1] for data in dataset]
        if class_list.count(class_list[0]) == len(class_list):
            return Node(label=class_list[0])
        if len(dataset[0]) == 1:
            return Node(label=max(class_list, key=class_list.count))
        best_feature = self.choose_best_feature(dataset)
        best_feature_name = feature_names[best_feature]
        tree = Node(feature_name=best_feature_name, feature=best_feature)
        feature_names.remove(best_feature_name)
        feature_value_list = [example[best_feature] for example in dataset]
        unique_values = set(feature_value_list)
        for value in unique_values:
            sub_labels = feature_names[:]
            sub_dataset = self.split_dataset(dataset, best_feature, value)
            sub_tree = self.create_tree(sub_dataset, sub_labels)
            tree.tree[value] = sub_tree
        return tree

    # 分裂数据集
    @staticmethod
    def split_dataset(dataset, axis, value):
        ret_dataset = []
        for feature_vector in dataset:
            if feature_vector[axis] == value:
                reduced_feature_vector = list(feature_vector[:axis])
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                ret_dataset.append(reduced_feature_vector)
        return ret_dataset
