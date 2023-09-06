import pandas as pd 
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn


df = pd.read_csv("iris.data", header=None)
#Ejercicio 3.1
print(df.describe())
print(df.dtypes)
print(df.head(10))

print("llaves:", df.keys(), "\nNumero de renglones:", len(df), "\nNumero de columnas:", len(df.columns))

print("\nNumero de Nan:", df.isnull().sum().sum())

arr = [[1,0,0,0,0],
       [0,1,0,0,0],
       [0,0,1,0,0],
       [0,0,0,1,0],
       [0,0,0,0,1]]

npArr = np.array(arr)

sparseArr = sparse.csr_matrix(npArr)

print("\nMatriz dispersa: \n", sparseArr)

print("\n",df.describe())

print("\n",df.describe().loc[['mean', 'std']])

df = df.set_index(4)

print("\nNumero de muestras para Iris Setosa:",len(df.loc['Iris-setosa']))
print("\nNumero de muestras para Iris Versicolor",len(df.loc['Iris-versicolor']))
print("\nNumero de muestras para Iris Virginica",len(df.loc['Iris-virginica']))

df.columns=['sepal length', 'sepal width', 'petal length', 'petal width']

df.index.names=['class']

print("\nData frame con encabezados: \n",df)

print("\nNumero de muestras para Iris Setosa:",len(df.loc['Iris-setosa']))
print("\nNumero de muestras para Iris Versicolor",len(df.loc['Iris-versicolor']))
print("\nNumero de muestras para Iris Virginica",len(df.loc['Iris-virginica']))

print("\nPrimeros 10 renglones de primeras 2 columnas: \n", df.iloc[0:10, 0:2])

labels = 'Iris Setosa', 'Iris Versicolor', 'Iris Virginica'
iris_set = len(df.loc['Iris-setosa'])
iris_vers = len(df.loc['Iris-versicolor'])
iris_virg = len(df.loc['Iris-virginica'])
sizes =[iris_set, iris_vers, iris_virg]

df = df.dropna();
values = df.values.min(), df.values.max(), df.values.mean()

fig, ax = plt.subplots()
langs = ['min', 'max', 'mean']
ax.bar(langs, values)
plt.show()

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)

plt.show()


df2 = df.sort_values(by=['sepal length'])

df2.plot(x="sepal length", y="sepal width", kind="scatter")
plt.show()

plt.title("Histograma del largo del sepalo")
plt.hist(df["sepal length"])
plt.show()

plt.title("Histograma del ancho del sepalo")
plt.hist(df["sepal width"])
plt.show()

plt.title("Histograma del largo de los petalos")
plt.hist(df["petal length"])
plt.show()

plt.title("Histograma del ancho de los petalos")
plt.hist(df["petal width"])
plt.show()


df.reset_index(inplace=True)
seaborn.pairplot(df, hue='class')
plt.show()

seaborn.jointplot(df, x='sepal length', y='sepal width')
plt.show()

seaborn.jointplot(df, x='sepal length', y='sepal width', kind="hex")
plt.show()

