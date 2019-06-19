import pandas as pd
import matplotlib.pyplot as plt
maxdist = 999

class Hierarchy:
    def __init__(self):
        self.df = pd.read_csv("melon.csv")
        self.df = self.df.loc[:5, :]
        self.attrs = ["density", "sugar"]
        self.df.columns = ["cate", "density", "sugar"]
        self.df2 = self.df.copy()

        self.tot = []
        for i in range(len(self.df) - 1):
            for j in range(len(self.df) - i):
                j += i
                if i != j:
                    self.tot.append([i, j, self.calc(self.df.iloc[i], self.df.iloc[j])])

        self.maxid = len(self.tot)
        self.tot = pd.DataFrame(self.tot)
        self.tot.columns = ["1", "2", "dist"]
        self.tot["id"] = 0
        for i in range(len(self.tot)):
            self.tot.loc[i, ["id"]] = i + 1
        self.tot = self.tot[["1", "2", "dist", "id"]]

    def init(self):
        self.init_heap()

    def run(self):
        a, b = self.tot.loc[0, ["1", "2"]].astype(int)
        self.df2 = self.df2[(self.df2.cate != a) & (self.df2.cate != b)]
        self.df2 = self.df2.reset_index().drop(columns=["index"])
        cate = self.min2(self.df.loc[a, "cate"], self.df.loc[b, "cate"])
        self.df.loc[a, "cate"] = cate
        self.df.loc[b, "cate"] = cate

        seq = self.tot[(self.tot["1"] == a) | (self.tot["2"] == a) | (self.tot["1"] == b) | (self.tot["2"] == b)].index
        ids = self.tot.loc[seq, "id"]
        self.tot.loc[seq, "dist"] = maxdist

        for id in ids:
            self.down_heap(self.tot[self.tot.id == id].index[0] + 1)
            self.tot = self.tot[self.tot.id != id]

        now = pd.DataFrame(self.df[self.df.cate == cate].mean()).T
        x = cate
        for i in range(len(self.df2)):
            y = self.df2.loc[i, "cate"]
            if x < y:
                self.maxid += 1
                tmp = pd.DataFrame([x, y, self.calc(now.iloc[0], self.df2.iloc[i]), self.maxid]).T
                tmp.columns = self.tot.columns
                self.tot = self.tot.append(tmp, ignore_index=True)
            else:
                self.maxid += 1
                tmp = pd.DataFrame([y, x, self.calc(now.iloc[0], self.df2.iloc[i]), self.maxid]).T
                tmp.columns = self.tot.columns
                self.tot = self.tot.append(tmp, ignore_index=True)
            current = len(tot)
            self.update_heap(current)

        self.df2 = self.df2.append(now, ignore_index=True)

    def calc(self, x1, x2):
        res = 0
        for attr in self.attrs:
            res += (x1[attr] - x2[attr]) ** 2
        return res ** 0.5

    def init_heap(self):
        length = len(self.tot)
        for i in range(length):
            current_node = length - i
            self.update_heap(current_node)

    def swap(self, current_node, parent_node):
        for attr in self.tot.columns:
            tmp = self.tot.loc[parent_node - 1, attr]
            self.tot.loc[parent_node - 1, attr] = self.tot.loc[current_node - 1, attr]
            self.tot.loc[current_node - 1, attr] = tmp

    def update_heap(self, current_node):
        print("current node is ", current_node)
        if current_node == 1:
            return
        parent_node = current_node // 2
        if self.tot.loc[parent_node - 1, "dist"] > self.tot.loc[current_node - 1, "dist"]:
            print("swap with ", parent_node)
            self.swap(current_node, parent_node)
            self.update_heap(parent_node)

    def min(self, x, y):
        if x < y:
            return 1
        return 0

    def min2(self, x, y):
        if x < y:
            return x
        return y

    def down_heap(self, current_node):
        print("current_node is ", current_node)
        if current_node * 2 > len(self.tot):
            print("quit")
            return
        left = self.tot.loc[current_node * 2 - 1, "dist"]
        right = 999
        if current_node * 2 + 1 <= len(self.tot):
            right = self.tot.loc[current_node * 2 + 1 - 1, "dist"]

        print(left, " ", right)
        if min(left, right):
            self.swap(current_node, current_node * 2)
            print("down left side")
            self.down_heap(current_node * 2)
        else:
            self.swap(current_node, current_node * 2 + 1)
            print("down right side")
            self.down_heap(current_node * 2 + 1)

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.scatter(self.df["density"], self.df["sugar"], s=70, alpha=0.7)
        for i in range(len(self.df)):
            self.ax.text(self.df.loc[i, "density"], self.df.loc[i, "sugar"], self.df.loc[i, "cate"], horizontalalignment="center",verticalalignment="bottom")


if __name__ == "__main__":
    hier = Hierarchy()
    hier.init()
    print(hier.tot)
    hier.draw()
    hier.run()
    print(hier.tot)
    print(hier.df)
    hier.draw()

