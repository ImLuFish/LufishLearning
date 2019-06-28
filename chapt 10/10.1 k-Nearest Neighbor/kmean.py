import pandas as pd


class KNN:
    def __init__(self):
        self.df = pd.read_csv("melon.csv")
        self.attrs = ["density", "sugar"]

    def calc_dist(self, x, y):
        res = 0
        for attr in self.attrs:
            res += (x[attr] - y[attr]) ** 2
        return res ** 0.5

    def kmean(self, x, k):
        for i in range(len(self.df)):
            self.df.loc[i, "dist"] = self.calc_dist(x, self.df.iloc[i])
            self.df = self.df.sort_values(["dist"]).reset_index().drop(columns=["index"])
            return self.df.iloc[: k].mean()["is_sweet"]


if __name__ == "__main__":
    df = pd.read_csv("melon.csv")
    calc = KNN()
    calc.kmean(df.iloc[0], 5)
