import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from scipy import stats

def corrfunc(x, y, **kws):
    rp, _ = stats.pearsonr(x, y)
    rs, _ = stats.spearmanr(x, y)
    rk, _ = stats.kendalltau(x, y)
    ax = plt.gca()
    ax.annotate("r = %.2f %.2f %.2f"%(rp,rs,rk)+"\n by Yi Sung",
                xy=(.1, .7), xycoords=ax.transAxes)

#
# Load iris dataset
#
#iris = datasets.load_iris()
iris = sns.load_dataset("iris")
#
# Create dataframe using IRIS dataset
#
# df = pd.DataFrame(iris.data)
# df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

#df['class'] = iris.target

#dataLength = len(df)
#
# Create pairplot of all the variables with hue set to class
#
# g = sns.pairplot(df,kind='hist', diag_kind='hist',diag_kws={'edgecolor':'w'})#,diag_kws=dict(fill=False))
# g.map_upper(sns.scatterplot)
# g.map_lower(corrfunc)


g = sns.PairGrid(iris,diag_sharey=False)
g.map_diag(plt.hist,edgecolor='white')
g.map_upper(sns.scatterplot,data=iris, hue='species')
g.map_lower(corrfunc)

g.savefig("output.png")
plt.show()