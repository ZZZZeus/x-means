#! /user/bin/python
# coding:UTF-8
"""
本代码参考：
https://github.com/shun-sy/x_means/blob/master/xmeans.py
"""
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import sys

class XMeans:
    def __init__(self, k_init = 2, **k_means_args):
        self.k_init = k_init
        self.k_means_args = k_means_args
        
    def fit(self,X):
        self.__clusters = []
        self.all_clusters = []
        #进行初步k=2的k-means分类
        clusters = self.Cluster.build(X,KMeans(self.k_init,**self.k_means_args).fit(X))
        self.all_clusters.extend(clusters)

        plt.scatter(clusters[0].data[:,0], clusters[0].data[:,1],s = 30)
        plt.scatter(clusters[0].center[0], clusters[0].center[1], c = "r", marker = "+", s = 100)
        plt.scatter(clusters[1].data[:,0], clusters[1].data[:,1],s = 30)
        plt.scatter(clusters[1].center[0], clusters[1].center[1], c = "r", marker = "+", s = 100)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.title("x-means example")
        plt.show()   
   		
        #进行递归，确定最合适的k
        self.__recursively_split(clusters)
        
        #数据分类号码的再分配
        self.labels_ = np.empty(X.shape[0],dtype = np.intp)
        for i,c in enumerate(self.__clusters):
            self.labels_[c.index] = i
        
        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])
        
        return self
        
    #递归分割
    def __recursively_split(self,clusters):
        for cluster in clusters:
            # 包含3个以下数据元素的聚类不需要再分割
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue
				
            self.all_clusters.remove(cluster)
			
            k_means = KMeans(2,**self.k_means_args).fit(cluster.data)
            c1,c2 = self.Cluster.build(cluster.data,k_means,cluster.index)
            self.all_clusters.append(c1)
            self.all_clusters.append(c2)
            try:
                if np.array(c1.cov).shape == ():
                    beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(c1.cov + c2.cov)                
                else:
                    beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            except:
                print(np.array(c1.cov))
                print(np.array(c1.cov).shape)
                print(np.array(c1.cov).shape[0])
            #正态分布下的概率
            alpha = 0.5 / stats.norm.cdf(beta)
            
            bic = (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) - cluster.df * np.log(cluster.size)
            
            for one in self.all_clusters:
                plt.scatter(one.data[:,0], one.data[:,1],s = 30)
                plt.scatter(one.center[0], one.center[1], c = "r", marker = "+", s = 100)
            if cluster.center[0]<1.5 and cluster.center[1]<1.5:
                bicx = 0.1
                bicy = 0.1
            elif cluster.center[0]<1.5 and cluster.center[1]>1.5:
                bicx = 0.5
                bicy = 2.5
            elif cluster.center[0]>1.5 and cluster.center[1]<1.5:
                bicx = 2
                bicy = 0.1
            else:
                bicx = 2
                bicy = 2.5
            plt.text(bicx, bicy, 'BIC1=%f' % cluster.bic(),  fontdict={'size': 16, 'color': 'r'})
            plt.text(bicx, bicy+0.2, 'BIC2=%f' % bic,  fontdict={'size': 16, 'color': 'r'})
            plt.annotate('',xy=cluster.center,xytext=(bicx,bicy),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.title("x-means example")
            plt.show()   
            
            if bic > cluster.bic():
                self.__recursively_split([c1,c2])
            else:
                self.__clusters.append(cluster)
                self.all_clusters.remove(c1)
                self.all_clusters.remove(c2)
                self.all_clusters.append(cluster)
            
            
    #聚类的信息
    class Cluster:
        @classmethod
        def build(cls,X,k_means,index = np.array([])):
            if any(index) == False:
                index = np.array(range(0,X.shape[0]))
            labels = range(0,k_means.get_params()["n_clusters"])    

            #return tuple(cls(X,index,k_means,label) for label in labels)
            return [cls(X,index,k_means,label) for label in labels]
        
        def __init__(self,X,index,k_means,label):
            self.data = X[k_means.labels_ == label]
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            #计算协方差矩阵
            if len(self.data) == 1:
                self.cov = 0
            else:
                self.cov = np.cov(self.data.T)
				
            
        
        #似然函数的数值计算
        def log_likelihood(self,alpha=1):
            result = 1
            try:
                if len(self.data) == 1:
                    result = -9999
                elif self.cov.shape == ():
                    if self.cov == 0:
                        result = -9999
                    else:
                        result = -1*np.log(np.pi * self.cov)/2.0 
                else:
                    result = sum(stats.multivariate_normal.logpdf(alpha * x,self.center,self.cov) for x in self.data)
            except Exception as e:
                print('=== エラー内容 ===')
                print('type:' + str(type(e)))
                print('args:' + str(e.args))
                print(self.cov.dtype)
                sys.exit()
            return result
            
        def bic(self):
            return self.log_likelihood() - self.df * np.log(self.size)/2
            
        
"""
np.random.normal(mu, var, num)
np.repeat
np.repeat([1,2], 2)] -> ([1,1,2,2])
http://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
np.tile
np.tile([1,2], 2) -> ([1,2,1,2])
http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
np.flatten()
http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #数据准备  产生正态分布数组80个元素
    x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
    y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()

    plt.scatter(x, y, s = 30)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title("x-means example")
    plt.show()
	
    #执行x-means算法进行聚类操作
    x_means = XMeans(random_state = 1).fit(np.c_[x,y]) 
    print(x_means.labels_)
    print(x_means.cluster_centers_)
    print(x_means.cluster_log_likelihoods_)
    print(x_means.cluster_sizes_)

    #作图结果
    #plt.rcParams["font.family"] = "Hiragino Kaku Gothic Pro"
    plt.scatter(x, y, c = x_means.labels_, s = 30)
    plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", s = 100)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title("x-means example")
    plt.show()
    
    
    
    
    
    
