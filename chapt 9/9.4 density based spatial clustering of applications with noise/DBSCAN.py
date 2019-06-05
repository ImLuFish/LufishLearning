import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Dbscan:
    def __init__(self):
        self.df = pd.read_csv("melon.csv")
        self.e = 0.11
        self.minpts = 5
        self.attrs = ["density", "sugar"]

        for i in range(len(self.df)):
            for j in range(len(self.df) - i):
                j += i
                self.df.loc[i, str(j)] = self.calc(self.df.iloc[i], self.df.iloc[j])

        for i in range(len(self.df)):
            for j in range(i):
                self.df.loc[i, str(j)] = self.df.loc[j, str(i)]

        cols = [str(i) for i in range(len(self.df))]
        self.df.loc[:, cols] = self.df.loc[:, cols].where(self.df.loc[:, cols] <= 0.11)

        self.df.loc[:, cols] = self.df.loc[:, cols].replace(0, np.nan)

        for i in range(len(self.df)):
            self.df.loc[i, "num"] = self.df.loc[i, cols].count()

        self.core = self.df[self.df.num >= self.minpts - 1]["id"].array

        self.df.loc[:, "core"] = False
        for i in self.core:
            self.df.loc[i, "core"] = True

        self.df1 = self.df.copy()
        self.df1["visit"] = True
        self.df["color"] = "lightblue"

    def calc(self, x1, x2):
        res = 0
        for attr in self.attrs:
            res += (x1[attr] - x2[attr]) ** 2
        return res ** 0.5

    def in_array(self, x):
        for element in x:
            if self.df1.loc[element, "visit"] and not self.df1.loc[element, "core"]:
                self.node.append(element)
                self.df1.loc[element, "visit"] = False
                print(element, " in")
        print("now array is: ", self.node, "\n")

    def core_in_array(self, x):
        for element in x:
            if self.df1.loc[element, "visit"] and self.df1.loc[element, "core"]:
                self.node.append(element)
                self.df1.loc[element, "visit"] = False
                print(element, " in")
        print("now array is: ", self.node, "\n")

    def out_array(self):
        res = self.node[0]
        print(res, " out")
        del self.node[0]
        self.tot.append(res)
        print("now array is: ", self.node, "\n")
        return res

    def run(self, start_node=7):
        self.node = []
        self.tot = []
        self.core_in_array([start_node])
        while len(self.node) > 0:
            current_node = self.out_array()
            self.core_in_array(self.df1[~self.df1[str(current_node)].isna()].id.array)

        self.cores = self.tot.copy()
        self.tot = []
        self.df1["visit"] = True

        self.core_in_array(self.cores)

        while len(self.node) > 0:
            current_node = self.out_array()
            self.in_array(self.df1[~self.df1[str(current_node)].isna()].id.array)


        self.df.loc[self.tot, "color"] = "orange"
        self.df.loc[self.cores, "color"] = "red"

    def after_run(self):
        print("OK, classified points removed")
        cols = [str(i) for i in self.tot]
        self.df1.drop(columns=cols)

    def draw(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.scatter(self.df["density"], self.df["sugar"], s=90, c=self.df["color"], alpha=0.6)
        for i in range(len(self.df)):
            self.ax.text(self.df.loc[i, "density"], self.df.loc[i, "sugar"] + 0.01, str(i + 1))
        self.ax.plot([self.df.iloc[10]["density"], self.df.iloc[11]["density"]], [self.df.iloc[10]["sugar"], self.df.iloc[11]["sugar"]],
                color='black', linestyle="--", alpha=0.5)
        font = {
            'weight': 'normal',
            'color': 'red',
            'size': 8
        }
        self.ax.text((self.df.iloc[11]["density"] + self.df.iloc[10]["density"]) / 2,
                (self.df.iloc[11]["sugar"] + self.df.iloc[10]["sugar"]) / 2, round(self.df.loc[10, "11"], 3),
                fontdict=font, rotation=18, horizontalalignment="center", verticalalignment="bottom")
        self.ax.set_xlabel("density")
        self.ax.set_ylabel("sugar")


if __name__ == "__main__":
    melon = Dbscan()
    melon.run(8)
    melon.draw()
    melon.after_run()
    melon.run(14)
    melon.draw()