import pandas as pd
import matplotlib.pyplot as plt


class Lvq:
    def __init__(self):
        self.df = pd.read_csv("melon.csv", index_col=0)
        self.color = {
            0: "blue",
            1: "red"
        }
        self.df.loc[:, "type"] = self.df["type"].apply(lambda x: self.color[x])

        self.k = 5
        self.ita = 0.1
        self.numlist = [18, 23, 28, 22, 15]
        self.attrs = ["density", "sugar"]
        self.p = self.df.iloc[self.numlist].reset_index().drop(columns=["id"])

    def draw(self, status=0):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.scatter(self.df["density"], self.df["sugar"], s=90, c=self.df["type"], alpha=0.5)
        for i in range(len(self.df)):
            self.ax.text(self.df.loc[i, "density"], self.df.loc[i, "sugar"] + 0.01, self.df.index[i])
        self.ax.set_xlabel("density")
        self.ax.set_ylabel("sugar")
        if status == 1:
            self.ax.scatter(self.p["density"], self.p["sugar"], marker="*", c="green", s=150, alpha=0.7)

    def calc(self, x, p):
        res = 0
        for attr in self.attrs:
            res += (x[attr] - p[attr]) ** 2
        return res ** 0.5

    def run(self, times):
        for time in range(times):
            i = self.df.sample(n=1).index[0]
            min_p = 2 ** 16 - 1
            for j in range(self.k):
                dist = self.calc(self.df.iloc[i], self.p.iloc[j])
                self.df.loc[i, str(j)] = dist
                if dist < min_p:
                    min_p = dist
                    self.df.loc[i, "min_p"] = j
                    self.df.loc[i, "min_p_type"] = self.p.iloc[j]["type"]

            j = int(self.df.loc[i, "min_p"])
            print("we choose: ", i, ", the closest is " + str(j))

            if self.df.loc[i, "type"] == self.df.loc[i, "min_p_type"]:
                for attr in self.attrs:
                    self.p.loc[j, attr] = self.p.loc[j, attr] + self.ita * (self.df.loc[i, attr] - self.p.loc[j, attr])
                print("    ---> move closer")
            else:
                for attr in self.attrs:
                    self.p.loc[j, attr] = self.p.loc[j, attr] - self.ita * (self.df.loc[i, attr] - self.p.loc[j, attr])
                print("    ---> move away")

        self.draw(status=1)


if __name__ == "__main__":
    melon = Lvq()
    melon.draw(status=1)
    melon.run(20)
