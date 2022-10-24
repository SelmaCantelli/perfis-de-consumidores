# perfis-de-consumidores - 3 Sprints 

Perfis de Consumidores a partir da Redução de Dimensionalidade por Componentes Principais
# coding: utf-8

import pandas as pd


###Carregando os dados
data = pd.read_excel('vendasloja.xlsx')
dim_item = pd.read_excel('basecompleta.xlsx')


###Preparando o arquivo de informações dos itens 
dim_item.groupby(['id_item','nm_item','cd_ean']).agg({'vl_quantity_sold':['sum','count',lambda x: sum((x>0))]})


### Análise PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pca = PCA()



scaled_values = scaler.fit_transform(data.iloc[:,1:].values.T)
pca.fit(scaled_values)



pca.transform(scaled_values).shape




pd.DataFrame(scaled_values).T.to_excel('pospca.xlsx')


# In[8]:


pca.explained_variance_.shape





pca.explained_variance_ratio_.sum()




pca.explained_variance_ratio_.shape





pca.explained_variance_ratio_.cumsum()




import matplotlib.pyplot as plt





plt.plot(pca.explained_variance_ratio_.cumsum())





### Histograma de frequências para representar meus dados
import matplotlib.pyplot as plt





dados = ([0.54333475, 0.61304935, 0.64940112, 0.6775558 , 0.70406772,
       0.72303032, 0.74055695, 0.75653088, 0.76815192, 0.77867318,
       0.78827749, 0.79698598, 0.80475474, 0.81169824, 0.81806326,
       0.8236275 , 0.82903225, 0.8340572 , 0.83892104, 0.84357219,
       0.84803428, 0.85239016, 0.85629484, 0.85999793, 0.8636376 ,
       0.86698213, 0.87019401, 0.87329331, 0.87618576, 0.87904927,
       0.88184495, 0.88445242, 0.88695179, 0.88942564, 0.89182868,
       0.89412937, 0.89631788, 0.89842552, 0.90045556, 0.90236366,
       0.90425958, 0.90608294, 0.90785718, 0.90961706, 0.91132739,
       0.91296733, 0.91454079, 0.9160984 , 0.91764029, 0.91912287,
       0.92056881, 0.9219837 , 0.92335905, 0.92472424, 0.92604061,
       0.92731103, 0.92857742, 0.92983822, 0.93103729, 0.93220465,
       0.93333255, 0.93444962, 0.93555395, 0.93663336, 0.93767483,
       0.93869753, 0.93970196, 0.94066813, 0.94160643, 0.94253488,
       0.94344084, 0.9443097 , 0.94516687, 0.94600543, 0.9468314 ,
       0.94763836, 0.94841313, 0.94917396, 0.94991976, 0.95066313,
       0.95139392, 0.95210787, 0.95281193, 0.95350123, 0.95417254,
       0.95483104, 0.95546317, 0.95607701, 0.9566819 , 0.95726851,
       0.95784688, 0.95841997, 0.95898196, 0.95953375, 0.96007373,
       0.96060693, 0.96112842, 0.96164201, 0.96214933, 0.96264788,
       0.96313475, 0.96361742, 0.96409231, 0.964556  , 0.96500914,
       0.96545302, 0.96588774, 0.96631546, 0.96673486, 0.96714746,
       0.96755004, 0.96795014, 0.96834596, 0.96873834, 0.96912432,
       0.96950503, 0.96988267, 0.97025017, 0.97060924, 0.97096597,
       0.97131691, 0.97166228, 0.97200398, 0.97234402, 0.97267767,
       0.97300562, 0.9733276 , 0.97364754, 0.97396498, 0.97427582,
       0.97458352, 0.97488915, 0.97518581, 0.97548113, 0.975773  ,
       0.97606105, 0.97634654, 0.97662802, 0.97690709, 0.97718355,
       0.97745798, 0.97772429, 0.97798438, 0.97824207, 0.97849794,
       0.97875073, 0.97900101, 0.97924823, 0.97949264, 0.97973425,
       0.97996975, 0.98020251, 0.980433  , 0.9806611 , 0.9808877 ,
       0.98111363, 0.98133363, 0.98155171, 0.98176905, 0.98198219,
       0.98219335, 0.98240317, 0.98261121, 0.98281763, 0.98301977,
       0.98322096, 0.9834192 , 0.9836135 , 0.98380729, 0.98399998,
       0.98419073, 0.98437814, 0.98456484, 0.98474771, 0.98492949,
       0.98510762, 0.98528497, 0.98546109, 0.98563593, 0.98580793,
       0.98597728, 0.98614448, 0.98631072, 0.98647637, 0.9866407 ,
       0.98680418, 0.98696422, 0.98712253, 0.98727937, 0.98743506,
       0.98758876, 0.98774111, 0.98789116, 0.98804017, 0.98818664,
       0.98833188, 0.98847626, 0.98861951, 0.98876062, 0.9888995 ,
       0.98903805, 0.98917516, 0.98931061, 0.98944496, 0.98957873,
       0.98971086, 0.98984047, 0.98996977, 0.99009766, 0.99022401,
       0.99034803, 0.99047087, 0.99059292, 0.99071382, 0.99083379,
       0.99095343, 0.99107167, 0.99118928, 0.99130638, 0.99142183,
       0.99153659, 0.9916504 , 0.9917637 , 0.99187479, 0.99198442,
       0.99209336, 0.99220182, 0.99230911, 0.99241494, 0.99252003,
       0.99262419, 0.99272757, 0.99283049, 0.99293245, 0.99303231,
       0.99313197, 0.9932313 , 0.99332849, 0.9934249 , 0.99352043,
       0.99361526, 0.99370955, 0.99380308, 0.99389581, 0.99398752,
       0.99407756, 0.99416713, 0.99425539, 0.99434258, 0.99442918,
       0.99451513, 0.99460049, 0.99468557, 0.99477013, 0.99485329,
       0.99493531, 0.99501679, 0.99509686, 0.99517628, 0.99525555,
       0.99533417, 0.9954127 , 0.99549006, 0.99556639, 0.9956422 ,
       0.99571652, 0.99579019, 0.99586324, 0.99593567, 0.99600743,
       0.99607818, 0.99614865, 0.99621847, 0.99628751, 0.99635618,
       0.99642411, 0.99649144, 0.99655781, 0.99662381, 0.99668962,
       0.99675493, 0.99681994, 0.99688395, 0.99694705, 0.9970096 ,
       0.99707127, 0.99713196, 0.99719229, 0.99725175, 0.99731025,
       0.99736804, 0.99742512, 0.99748178, 0.99753779, 0.9975937 ,
       0.99764884, 0.99770344, 0.99775791, 0.99781197, 0.99786539,
       0.99791786, 0.99796989, 0.99802151, 0.99807204, 0.9981224 ,
       0.99817193, 0.99822105, 0.99826935, 0.99831723, 0.99836476,
       0.99841162, 0.99845793, 0.99850338, 0.99854871, 0.99859297,
       0.99863696, 0.99868032, 0.99872336, 0.99876552, 0.99880736,
       0.99884863, 0.99888922, 0.99892958, 0.99896961, 0.99900891,
       0.99904772, 0.99908587, 0.99912319, 0.99916003, 0.99919641,
       0.9992316 , 0.99926648, 0.99930064, 0.99933418, 0.9993676 ,
       0.99940072, 0.99943316, 0.99946451, 0.99949559, 0.99952594,
       0.99955597, 0.99958531, 0.99961429, 0.99964309, 0.99967146,
       0.99969894, 0.99972479, 0.99975016, 0.99977472, 0.99979882,
       0.99982219, 0.99984499, 0.99986725, 0.99988785, 0.99990745,
       0.99992609, 0.99994418, 0.99996068, 0.99997305, 0.99998362,
       0.99999334, 1.        ])


plt.hist(dados, 5, rwidth=0.5)
plt.show()




### SPRINT 2  ****




data.head()





from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pca = PCA()




data.iloc[:,1:]





from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt




# Criando os dados
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
 
# Visualizando os dados
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()





distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Construindo e ajustando o modelo
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_




for key, val in mapping1.items():
    print(f'{key} : {val}')





plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distorção')
plt.title('Método do Cotovelo mostrando o k')





for key, val in mapping2.items():
    print(f'{key} : {val}')





plt.plot(K, inertias, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('Inertia')
plt.title('Método do cotovelo usando a inertia')
plt.show()





import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans





df = sns.load_dataset("iris")




kmeans = KMeans(n_clusters = 5)
kmeans.fit(X)

# Centroides das entradas
kmeans.cluster_centers_

# Clusters das entradas
kmeans.labels_





plt.plot(K, distortions, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('Distorção')
plt.title('Método de Cotovelo usando Distorção')
plt.show()




### Dimensionando os dados
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    

pca = PCA()
x_new = pca.fit_transform(X)

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1, coeff[i,1] * 1, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1, coeff[i,1] * 1, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

myplot(x_new[:,0:2],np.transpose(pca.components_[0:4,1 :]))
plt.show()





### K- Means - AGRUPAMENTO COM SCIPY - Clusterização
import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2





# observations
data =([0.54333475, 0.61304935, 0.64940112, 0.6775558 , 0.70406772,
       0.72303032, 0.74055695, 0.75653088, 0.76815192, 0.77867318,
       0.78827749, 0.79698598, 0.80475474, 0.81169824, 0.81806326,
       0.8236275 , 0.82903225, 0.8340572 , 0.83892104, 0.84357219,
       0.84803428, 0.85239016, 0.85629484, 0.85999793, 0.8636376 ,
       0.86698213, 0.87019401, 0.87329331, 0.87618576, 0.87904927,
       0.88184495, 0.88445242, 0.88695179, 0.88942564, 0.89182868,
       0.89412937, 0.89631788, 0.89842552, 0.90045556, 0.90236366,
       0.90425958, 0.90608294, 0.90785718, 0.90961706, 0.91132739,
       0.91296733, 0.91454079, 0.9160984 , 0.91764029, 0.91912287,
       0.92056881, 0.9219837 , 0.92335905, 0.92472424, 0.92604061,
       0.92731103, 0.92857742, 0.92983822, 0.93103729, 0.93220465,
       0.93333255, 0.93444962, 0.93555395, 0.93663336, 0.93767483,
       0.93869753, 0.93970196, 0.94066813, 0.94160643, 0.94253488,
       0.94344084, 0.9443097 , 0.94516687, 0.94600543, 0.9468314 ,
       0.94763836, 0.94841313, 0.94917396, 0.94991976, 0.95066313,
       0.95139392, 0.95210787, 0.95281193, 0.95350123, 0.95417254,
       0.95483104, 0.95546317, 0.95607701, 0.9566819 , 0.95726851,
       0.95784688, 0.95841997, 0.95898196, 0.95953375, 0.96007373,
       0.96060693, 0.96112842, 0.96164201, 0.96214933, 0.96264788,
       0.96313475, 0.96361742, 0.96409231, 0.964556  , 0.96500914,
       0.96545302, 0.96588774, 0.96631546, 0.96673486, 0.96714746,
       0.96755004, 0.96795014, 0.96834596, 0.96873834, 0.96912432,
       0.96950503, 0.96988267, 0.97025017, 0.97060924, 0.97096597,
       0.97131691, 0.97166228, 0.97200398, 0.97234402, 0.97267767,
       0.97300562, 0.9733276 , 0.97364754, 0.97396498, 0.97427582,
       0.97458352, 0.97488915, 0.97518581, 0.97548113, 0.975773  ,
       0.97606105, 0.97634654, 0.97662802, 0.97690709, 0.97718355,
       0.97745798, 0.97772429, 0.97798438, 0.97824207, 0.97849794,
       0.97875073, 0.97900101, 0.97924823, 0.97949264, 0.97973425,
       0.97996975, 0.98020251, 0.980433  , 0.9806611 , 0.9808877 ,
       0.98111363, 0.98133363, 0.98155171, 0.98176905, 0.98198219,
       0.98219335, 0.98240317, 0.98261121, 0.98281763, 0.98301977,
       0.98322096, 0.9834192 , 0.9836135 , 0.98380729, 0.98399998,
       0.98419073, 0.98437814, 0.98456484, 0.98474771, 0.98492949,
       0.98510762, 0.98528497, 0.98546109, 0.98563593, 0.98580793,
       0.98597728, 0.98614448, 0.98631072, 0.98647637, 0.9866407 ,
       0.98680418, 0.98696422, 0.98712253, 0.98727937, 0.98743506,
       0.98758876, 0.98774111, 0.98789116, 0.98804017, 0.98818664,
       0.98833188, 0.98847626, 0.98861951, 0.98876062, 0.9888995 ,
       0.98903805, 0.98917516, 0.98931061, 0.98944496, 0.98957873,
       0.98971086, 0.98984047, 0.98996977, 0.99009766, 0.99022401,
       0.99034803, 0.99047087, 0.99059292, 0.99071382, 0.99083379,
       0.99095343, 0.99107167, 0.99118928, 0.99130638, 0.99142183,
       0.99153659, 0.9916504 , 0.9917637 , 0.99187479, 0.99198442,
       0.99209336, 0.99220182, 0.99230911, 0.99241494, 0.99252003,
       0.99262419, 0.99272757, 0.99283049, 0.99293245, 0.99303231,
       0.99313197, 0.9932313 , 0.99332849, 0.9934249 , 0.99352043,
       0.99361526, 0.99370955, 0.99380308, 0.99389581, 0.99398752,
       0.99407756, 0.99416713, 0.99425539, 0.99434258, 0.99442918,
       0.99451513, 0.99460049, 0.99468557, 0.99477013, 0.99485329,
       0.99493531, 0.99501679, 0.99509686, 0.99517628, 0.99525555,
       0.99533417, 0.9954127 , 0.99549006, 0.99556639, 0.9956422 ,
       0.99571652, 0.99579019, 0.99586324, 0.99593567, 0.99600743,
       0.99607818, 0.99614865, 0.99621847, 0.99628751, 0.99635618,
       0.99642411, 0.99649144, 0.99655781, 0.99662381, 0.99668962,
       0.99675493, 0.99681994, 0.99688395, 0.99694705, 0.9970096 ,
       0.99707127, 0.99713196, 0.99719229, 0.99725175, 0.99731025,
       0.99736804, 0.99742512, 0.99748178, 0.99753779, 0.9975937 ,
       0.99764884, 0.99770344, 0.99775791, 0.99781197, 0.99786539,
       0.99791786, 0.99796989, 0.99802151, 0.99807204, 0.9981224 ,
       0.99817193, 0.99822105, 0.99826935, 0.99831723, 0.99836476,
       0.99841162, 0.99845793, 0.99850338, 0.99854871, 0.99859297,
       0.99863696, 0.99868032, 0.99872336, 0.99876552, 0.99880736,
       0.99884863, 0.99888922, 0.99892958, 0.99896961, 0.99900891,
       0.99904772, 0.99908587, 0.99912319, 0.99916003, 0.99919641,
       0.9992316 , 0.99926648, 0.99930064, 0.99933418, 0.9993676 ,
       0.99940072, 0.99943316, 0.99946451, 0.99949559, 0.99952594,
       0.99955597, 0.99958531, 0.99961429, 0.99964309, 0.99967146,
       0.99969894, 0.99972479, 0.99975016, 0.99977472, 0.99979882,
       0.99982219, 0.99984499, 0.99986725, 0.99988785, 0.99990745,
       0.99992609, 0.99994418, 0.99996068, 0.99997305, 0.99998362,
       0.99999334, 1])
  
# normalize
data = whiten(data)
  
print(data)





# Média das distâncias euclidianas
centroids, mean_value = kmeans(data, 3)
  
print("Códigos :\n", centroids, "\n")
print("Média das distâncias euclidianas :", 
      mean_value.round(5))





# mapeando os centróides
clusters, distances = vq(data, centroids)
  
print("Cluster index :", clusters, "\n")
print("Distancia para os centroides :", distances)





# Clusterização
# atribuir centroides e clusters
centroids, clusters = kmeans2(data, 3, 
                              minit='random')
  
print("Centroids :\n", centroids, "\n")
print("Clusters :", clusters)
plt.show()





# Plotando com os dados de distâncias dos clusters
x1 = np.array([1, 1, 1, 1, 1 ,1 ,1, 1 ,1, 1, 1, 1 ,1 ,1, 1 ,1 ,1 ,1 ,1, 1 ,1, 1, 1, 1 ,1 ,1 ,1 ,1, 1 ,1, 1 ,1, 1, 1, 1, 1, 1,
 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1 ,0 ,0, 0, 2, 2, 2, 0, 2, 0, 2 ,2 ,2, 2, 2,2 ,0, 2, 2, 2, 2, 0, 2, 2, 2,
 2, 0, 0 ,0, 2, 2, 2 ,2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2 ,2, 2 ,2 ,2, 2, 2 ,2 ,0 ,2 ,0 ,0, 0, 0, 2, 0 ,0 ,0 ,0,
 0,0,2, 0, 0, 0, 0, 0, 2 ,0 ,2, 0, 2, 0 ,0 ,2, 0, 0, 0, 0, 0, 0, 2 ,2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2 ,0,0 ,0])
x2 = np.array([1, 1, 1 ,1 ,1, 1, 1, 1, 1 ,1, 1 ,1 ,1 ,1, 1 ,1, 1 ,1, 1 ,1 ,1 ,1, 1, 1, 1, 1, 1, 1 ,1, 1, 1 ,1, 1, 1 ,1, 1, 1,
 1, 1, 1, 1 ,0, 1, 1, 1 ,1, 1, 1, 1 ,1 ,2 ,2 ,2 ,0 ,2, 0, 2 ,0, 2, 0, 0, 0, 0 ,0, 0, 2, 0, 0, 0, 0, 2, 0, 0 ,0,
 0 ,2 ,2, 2 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,2, 2 ,0 ,0, 0, 0 ,0 ,0, 0, 0 ,0 ,0 ,0, 0 ,0 ,2, 0, 2, 2, 2, 2, 0, 2 ,2, 2, 2,
 2 ,2 ,0, 2 ,2, 2, 2, 2 ,0 ,2, 0 ,2, 2, 2, 2, 2, 2 ,2 ,2, 2, 2, 2, 2, 0, 2 ,2 ,2, 2 ,2, 2 ,2, 0, 2, 2 ,2 ,0, 2,
 2, 2])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)





plt.plot()
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()





import mpl_toolkits.mplot3d 

from sklearn import datasets
from sklearn.decomposition import PCA

# importando os dados
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plotando
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Comprimento")
plt.ylabel("Largura")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())





fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

ax.set_title("Três primeiras direções do PCA")
ax.set_xlabel("Primeiro Autovetor")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Segundo Autovetor")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Terceiro Autovetor")
ax.w_zaxis.set_ticklabels([])

plt.show()





### Sprint 3 *** - Filtrando os itens pelos números de centróides, para identificar os personas
### Analisando com os centróides localizados na clusterização com Kmeans 
### Para utilizar na visualização dos itens comprados para identificar as gerações dos consumidores




data = pd.read_excel('vendasloja.xlsx')
dim_item = pd.read_excel('basecompleta.xlsx')





dim_item.groupby(['id_item','nm_item','cd_ean']).agg({'vl_quantity_sold':['sum','count',lambda x: sum((x>0))]})





tabela = pd.read_excel ('basecompleta.xlsx') 

display (tabela) 


# In[57]:


### Filtro utilizando o valor do centróide - Características da Geração Y ou Millennials (nascidos em 1981 a 1996)
tabela[tabela['vl_quantity_sold'] <0.07877817]


# In[52]:


### Filtro utilizando o valor do centróide - Características da Geração X (nascidos em1965 a 1980) 
tabela[tabela['vl_quantity_sold'] <0.08442037]




### Filtro utilizando o valor do centróide - Características da Babyboomers (nascidos entre 1946 a 1964)
tabela[tabela['vl_quantity_sold'] <4.55312925]





for key, val in mapping1.items():
    print(f'{key} : {val}')





#### Utilizando os 3 primeiros resultados 
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(
    x=[0.88, 1.76, 3.45],
    y=[0.88, 1.76, 3.45],
       mode='markers',
    marker=dict(
        color=[30, 50, 80],
        size=[25, 35, 65],
        showscale=True
        )
)])

fig.show()






