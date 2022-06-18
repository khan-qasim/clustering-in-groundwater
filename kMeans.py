Python code for implementation of KMeans
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df = pd.read_csv(’Radon_data.csv’)
df.head()
plt.scatter(df[’Rn’], df[’pH’])
#plt.show()
km = KMeans(n_clusters= 4)
km
y_predicted = km.fit_predict(df[[’Rn’, ’TDS’]])
y_predicted
df[’cluster’] = y_predicted
df.head()
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
#df4 = df[df.cluster == 3]
plt.scatter(df1.Rn, df1[’TDS’], color = ’green’)
plt.scatter(df2.Rn, df2[’TDS’], color = ’red’)
plt.scatter(df3.Rn, df3[’TDS’], color = ’yellow’)
#plt.scatter(df4.Rn, df4[’TDS’], color = ’red’)
plt.xlabel(’Rn-222 (Bq/L)’)
plt.ylabel(’pH’)
plt.show()
#plt.savefig(’KMeans.png’, format=’png’, dpi=800)
#plt.close()
k_rng = range(1,10)
sse = []
for k in k_rng:
km = KMeans(n_clusters=k)
km.fit(df[[’Rn’, ’TDS’]])
sse.append(km.inertia_)
sse
print(sse)
print(km.cluster_centers_)
plt.xlabel(’K’)
plt.ylabel(’Sum of squared error’)
value_sse = plt.plot(k_rng, sse)
plt.show()
#plt.savefig(’KvsSSE.png’, format=’png’, dpi=800)