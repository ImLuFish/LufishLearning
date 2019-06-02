import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Mixture_of_Gaussian():
    def __init__(self):
        self.df = pd.read_csv("melon.csv")
        self.k = 3
        self.alpha = [1/3, 1/3, 1/3]
        self.u = self.df.iloc[[5, 21, 26]]
        self.u = self.u.reset_index().drop(columns=["index"])
        self.sigma = [0.1 * np.eye(2) for i in range(self.k)]
        self.attrs = ["density", "sugar"]

    def calc_p(self, x, n, u, sigma):
        sigma = np.matrix(sigma)
        u = np.matrix(u)
        x = np.matrix(x)
        first_part = (2 * np.pi) ** (n / 2) * (np.linalg.det(sigma) ** 0.5)
        second_part = np.e ** (-0.5 * ((x - u) * (sigma.I) * (x - u).T).A[0][0])
        return second_part / first_part


    def run(self):
        for j in range(len(self.df)):
            res = []
            for i in range(self.k):
                res.append(self.alpha[i] * self.calc_p(self.df.iloc[j][self.attrs], len(self.attrs), self.u.iloc[i][self.attrs], self.sigma[i]))
            div = np.array(res).sum()
            for i in range(self.k):
                self.df.loc[j, "gamma" + str(i)] = res[i] / div

        for i in range(self.k):
            div = self.df["gamma" + str(i)].sum()
            for j in range(len(self.df)):
                for attr in self.attrs:
                    self.df.loc[j, attr + "weight"] = self.df.loc[j, attr] * self.df.loc[j, "gamma" + str(i)]
            for attr in self.attrs:
                self.u.loc[i, attr] = self.df[attr + "weight"].sum() / div

        for i in range(self.k):
            div = self.df["gamma" + str(i)].sum()
            u1 = np.matrix(self.u.loc[i, self.attrs])
            matrix_list = []
            for j in range(len(self.df)):
                x = np.matrix(self.df.loc[j, self.attrs])
                matrix_list.append(self.df.loc[j, "gamma" + str(i)] * ((x - u1).T * (x - u1)))

            tot = matrix_list[0]
            for matrix in matrix_list[1:]:
                tot += matrix
            self.sigma[i] = tot / div
            self.alpha[i] = div / len(self.df)


if __name__ == "__main__":
    melon = Mixture_of_Gaussian()
    melon.run()
    print(melon.u)
    print(melon.sigma)
    print(melon.alpha)
