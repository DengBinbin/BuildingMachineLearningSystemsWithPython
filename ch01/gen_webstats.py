# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script generates web traffic data for our hypothetical
# web startup "MLASS" in chapter 01

import os
import scipy as sp
from scipy.stats import gamma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
chinese_font = FontProperties(fname='/home/deng/workspace/venv/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/YaHei.ttf')
from utils import DATA_DIR, CHART_DIR



sp.random.seed(3)  # to reproduce the data later on

x = sp.arange(1, 31*24)
y = sp.array(200*(sp.sin(2*sp.pi*x/(7*24))), dtype=float)
y += gamma.rvs(15, loc=0, scale=100, size=len(x)) #discrete=True
y += 2 * sp.exp(x/100)
y = sp.ma.array(y, mask=[y<0])
print(sum(y), sum(y<0))

plt.scatter(x, y)

plt.title("Web traffic over the last month", fontproperties=chinese_font)
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(5)], 
           ['week %i' %(w+1) for w in range(5)])
plt.autoscale(tight=True)
plt.grid()
plt.savefig(os.path.join(CHART_DIR, "1400_01_01.png"))

sp.savetxt(os.path.join(DATA_DIR, "web_traffic.tsv"), 
           list(zip(x, y)), delimiter="\t", fmt="%s")


