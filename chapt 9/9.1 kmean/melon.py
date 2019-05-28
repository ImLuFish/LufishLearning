"""
a = np.random.normal(2,0.5,30)
b = np.random.normal(10,3,30)
c = 5 + 10 * np.random.rand(30)
df = pd.DataFrame({"sugar":a,'density':b,"color":c})
df.to_csv("melon.csv",index=False)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family']=['Arial']

class Kmean():
    def __init__(self):
        self.df = pd.read_csv("melon.csv")
        self.attrs = self.df.columns
        self.df["colorflag"] = "blue"
        self.length = len(self.df)

        self.draw(self.df)
        
        self.color_list = ["red","gold","limegreen","grey","black","chocolate","pink"]

    def calc(self,x,u):
        res = 0
        for attr in self.attrs:
            res += (x[attr] - u[attr])**2
        return res ** 0.5

    def calc_min_idx(self,x):
        return x.T.idxmin().array[0]
    
    
    def setting(self,k,numlist):
        if k != len(numlist):
            print("nums not match")
            return
        
        self.k = k
        
        for i in range(k):
            self.df.loc[numlist[i],"colorflag"] = self.color_list[i]
        
        self.u = self.df.loc[numlist,self.attrs]
        
        self.draw(self.df)
        
    def draw(self,df):
        print("Hey, pleaz wait... 3D graphing takes time.")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.df.sugar,self.df.density,self.df.color,s=60,c=self.df["colorflag"],alpha=0.45)
        for i in range(self.length):
            ax.text(self.df.sugar[i],self.df.density[i],self.df.color[i],str(i))
        for i in range(self.length):
            ax.plot(xs=[self.df.sugar[i],self.df.sugar[i]],ys=[self.df.density[i],self.df.density[i]],zs=[0,self.df.color[i]],color="grey",linestyle="--",alpha=0.3)
        ax.set_xlabel("sugar")
        ax.set_ylabel("density")
        ax.set_zlabel("colour")
        return
    
    def run(self):
        for i in range(len(self.df)):
            for j in range(len(self.u)):
                self.df.loc[i,str(j)] = self.calc(self.df.iloc[i],self.u.iloc[j])

        grouped = self.df.groupby(level=0)

        attrs2 = [str(i) for i in range(self.k)]

        self.df["category"] = grouped[attrs2].apply(self.calc_min_idx).astype(int)

        for i in range(self.k):
            self.df.loc[self.df.category==i,"colorflag"] = self.color_list[i]

        self.draw(self.df)

        grouped = self.df.groupby("category")

        self.u = self.u.reset_index().drop(columns=["index"])

        copy = self.u.copy()

        for num,part in grouped:
            for attr in self.attrs:
                self.u.loc[num,attr] = part[attr].mean()

        if self.u[self.attrs].equals(copy[self.attrs]):
            print('OK, enough iteration.')

        return

if __name__ == "__main__":
    a = Kmean()

    a.setting(3,[0,5,18])

    a.run()



