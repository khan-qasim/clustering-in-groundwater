#Python code for the implementation of DBSCAN
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
df = pd.read_csv("Rn_data.csv")
df.head()
z = StandardScaler()
df[["Longitude", "Latitude"]] = z.fit_transform(df
[["Longitude", "Latitude"]])
#plt.scatter(df["Longitude"], df["Latitude"])
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.show()
X = df[["Longitude", "Latitude"]]
df.head()
db = DBSCAN(eps = 3, min_samples= 4)
db.fit(X)
y_predict = db.fit_predict(df[["Rn"]])
df["cluster"] = y_predict
df1 = df[df.cluster == -1]
df2 = df[df.cluster == 0]
#Figure size
plt.figure(figsize = (9, 5))
#plt.figure(plt.savefig("DBSCAN_new.png", format =
"png", dpi = 800))
plt.scatter(df1["Longitude"], df1["Latitude"],
color = "red")
plt.scatter(df2["Longitude"], df2["Latitude"],
color = "green")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.ticklabel_format(style = "plain")
plt.legend()
#plt.figure(plt.figure(figsize=(9, 5)))
#plt.show()
plt.savefig("DBSCAN.png", format="png", dpi=800)
plt.close()
eps_rng = range(0,5)
mins = 3
nn = NearestNeighbors(mins + 1)
nn.fit(df[["Longitude", "Latitude"]])
distances, neighbors = nn.kneighbors(df[["Longitude",
"Latitude"]])
distances = np.sort(distances[:, mins], axis = 0)
print(distances)
distances_df = pd.DataFrame({"distances":
distances,
"index": list(range(0, len(distances)))})
distances_d = pd.DataFrame({"distances": distances,
"index": list(range(0, len(distances)))})
plt.plot(distances_d.index, distances_d.distances)
plt.xlabel("Index")
plt.ylabel("Distances")
#plt.show()
plt.savefig("IndexvDist.png", format="png", dpi=800)
