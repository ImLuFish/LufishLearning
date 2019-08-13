import pandas as pd
import numpy as np


class DecisionTree:
    def __init__(self):
        self.df = pd.read_excel("melon.xlsx")
        self.attrs = ['color', 'root', 'sound', 'line', 'belly', 'touch']

    def split_data(self, df, feature, value):
        df1 = df[df[feature] > value]
        df2 = df[df[feature] <= value]
        return df1, df2

    def reg_leaf(self, df):
        return df["res"].mean()

    def reg_err(self, df):
        if len(df)==1:
            return 0
        return df["res"].var() * len(df)

    def reg_hat_err(self, df):
        return sum((df["res"] - df["predict"]) ** 2)

    def choose_best_split(self, df, ops):
        split_feature = None
        split_value = None
        if len(set(df["res"])) == 1:
            return None, self.reg_leaf(df)

        RSS = self.reg_err(df)
        min_RSS = np.inf

        for attr in self.attrs:
            if df[attr].dtypes == object:
                now_RSS = 0
                for item in set(df[attr]):
                    now_RSS += self.reg_err(df[df[attr] == item])
                    if len(df[df[attr] == item]) < ops[1]:
                        now_RSS += np.inf
                if len(set(df[attr])) == 1:
                    now_RSS += np.inf
                if (now_RSS < min_RSS):
                    min_RSS = now_RSS
                    split_feature = attr
                    split_value = "category"
            else:
                for split_val in set(df[attr]):
                    df1, df2 = self.split_data(df, attr, split_val)
                    if (len(df1) < ops[1]) or (len(df2) < ops[1]):
                        continue
                    now_RSS = self.reg_err(df1) + self.reg_err(df2)
                    if now_RSS < min_RSS:
                        min_RSS = now_RSS
                        split_feature = attr
                        split_value = split_val
        if min_RSS == np.inf:
            return None, self.reg_leaf(df)
        if (RSS - min_RSS) < ops[0]:
            return None, self.reg_leaf(df)
        return split_feature, split_value

    def create_tree(self, df, ops):
        feature, value = self.choose_best_split(df, ops)
        if feature == None:
            return value
        tree = {}
        tree["split_feature"] = feature
        tree["split_value"] = value
        if value == "category":
            for item in set(df[feature]):
                tree[item] = self.create_tree(df[df[feature] == item], ops)
        else:
            df1, df2 = self.split_data(df, feature, value)
            tree["left"] = self.create_tree(df1, ops)
            tree["right"] = self.create_tree(df2, ops)
        return tree

    def run(self, df, ops=(1, 4)):
        self.tree = self.create_tree(df, ops)

    def predict(self, tree, df):
        if type(tree) == np.float64:
            return tree
        split_feature = tree["split_feature"]
        split_value = tree["split_value"]
        if tree["split_value"] == "category":
            try:
                return self.predict(tree[df[split_feature]], df)
            except Exception as e:
                return np.NaN
        else:
            if df[split_feature] > split_value:
                return self.predict(tree["left"], df)
            else:
                return self.predict(tree["right"], df)

    def run_predict(self, df, tree):
        for i in range(len(df)):
            df.loc[i, "predict"] = self.predict(tree, df.iloc[i])
        return df

    def sampling(self, df, frac=0.1, random_state=12345):
        df = df.reset_index()
        sample = df.sample(frac=frac, random_state=random_state)
        sample["flag"] = 1
        df2 = pd.merge(df, sample[["index", "flag"]], "left", on=["index"])
        return sample.drop(columns=["index", "flag"]).reset_index(drop=True), df2[df2["flag"].isna()].drop(
            columns=["index", "flag"]).reset_index(drop=True)

    def sampling_by_num(self, df, n=1, random_state=12345):
        df = df.reset_index()
        sample = df.sample(n=n, random_state=random_state)
        sample["flag"] = 1
        df2 = pd.merge(df, sample[["index", "flag"]], "left", on=["index"])
        return sample.drop(columns=["index", "flag"]).reset_index(drop=True), df2[df2["flag"].isna()].drop(
            columns=["index", "flag"]).reset_index(drop=True)

    def build_k_fold(self, df, k):
        n = int(len(df) * (1 / k))
        res_list = []
        for i in range(k - 1):
            df1, df = self.sampling_by_num(df, n=n)
            res_list.append(df1)
        res_list.append(df)
        return res_list

    def prune(self, df, tree):
        if type(tree) == np.float64:
            return tree
        no_split_RSS = self.reg_err(df)
        after_split_RSS = self.reg_hat_err(df)
        if no_split_RSS < after_split_RSS:
            print("merge happenned")
            return self.reg_leaf(df)
        else:
            new_tree = {}
            split_feature = tree["split_feature"]
            new_tree["split_feature"] = tree["split_feature"]
            split_value = tree["split_value"]
            if split_value == "category":
                new_tree["split_value"] = "category"
                for item in set(df[split_feature]):
                    new_tree[item] = self.prune(df[df[split_feature] == item], tree[item])
            else:
                new_tree["split_value"] = split_value
                df1, df2 = self.split_data(df, split_feature, split_value)
                new_tree["left"] = self.prune(df1, tree["left"])
                new_tree["right"] = self.prune(df2, tree["right"])
        return new_tree


DT = DecisionTree()
DT.run(DT.df, (0,1))
print(DT.tree)
result = DT.run_predict(DT.df, DT.tree)
print(result)