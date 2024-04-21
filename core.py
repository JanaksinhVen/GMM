import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
class MY_GMM:
    def __init__(self):
        # self.means = []
        # self.cov = []
        # self.weights = []
        pass

    def fit(self, data, n_clusters,max_iter = 5,tol = 0.001):
        self.n_clusters = n_clusters
        X_data = data
        #initalize param
        n_s, n_f = data.shape
        np.random.seed(0)   # Ensure reproducibility
     
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X_data)
        means = kmeans.cluster_centers_
        kmeans_labels = kmeans.labels_
        # Calculate cluster covariances as full covariance matrices
        kmeans_cluster_covariances = []
        for i in range(self.n_clusters):
            cluster_data = X_data[kmeans_labels == i]
            cluster_covariance = np.cov(cluster_data, rowvar=False)
            kmeans_cluster_covariances.append(cluster_covariance)

        cov = np.array(kmeans_cluster_covariances)

        # means = np.random.rand(self.n_clusters, n_f)
        # cov = [np.eye(n_f)/self.n_clusters for _ in range(self.n_clusters)]
        weights = np.ones(self.n_clusters) / self.n_clusters


        prev_log_likelihood = 0.000
        for _ in range(max_iter):
            # E-step
            lambda_latent = self.E_step(X_data, means, cov, weights)
            # print(lambda_latent)
            # M-step
            new_means, new_cov, new_weights = self.M_step(X_data, lambda_latent)
            means, cov, weights = new_means, new_cov, new_weights
            # Check for convergence
            log_likelihood = self.calculate_log_likelihood(X_data, means, cov, weights)
            if np.abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
    
        return means, cov, weights



    def E_step(self, data, means, cov, weights):
        n_s, n_f = data.shape    #no. of samples, no. of features
        
        lambda_latent = np.zeros((n_s, self.n_clusters))    
        
        for i in range(self.n_clusters):
            component_mean = means[i]
            component_cov = cov[i]
            component_cov += 1e-5 * np.eye(n_f)
            # likelihood = multivariate_normal.pdf(data, component_mean, component_cov)
            # Calculate the cluster likelihood
            difference = data - component_mean
            exponent = -0.5 * np.sum(np.dot(difference, np.linalg.inv(component_cov)) * difference, axis=1)
            likelihood = np.exp(exponent) / np.sqrt((2 * np.pi) ** n_f * np.maximum(1e-5,np.linalg.det(component_cov)))
            # print(exponent.shape)
            # lambda calculation
            lambda_latent[:, i] = weights[i] * likelihood
        # vector of the lambda
        lambda_latent /= lambda_latent.sum(axis=1, keepdims=True)
        return lambda_latent


    
    def M_step(self, data, lambda_latent):
        n_s, n_f = data.shape    #no. of samples, no. of features
        
        # Update mixing coefficients
        weights = lambda_latent.mean(axis=0)
        
        means = np.zeros((self.n_clusters, n_f))
        covariances = [np.zeros((n_f, n_f)) for _ in range(self.n_clusters)]
        
        for i in range(self.n_clusters):
            # Update means
            means[i] = np.sum(lambda_latent[:, i][:, np.newaxis] * data, axis=0) / lambda_latent[:, i].sum()
            
            # Update covariances
            diff = data - means[i]
            covariances[i] = np.dot((lambda_latent[:, i][:, np.newaxis] * diff).T, diff) / lambda_latent[:, i].sum()
        
        return means, covariances, weights

    def calculate_log_likelihood(self, data, means, cov, weights):
        n_s, n_f = data.shape
    
        
        log_likelihoods = np.zeros(n_s)
        
       
        sample_likelihood = np.zeros(n_s)
        for j in range(self.n_clusters):
            component_mean = means[j]
            component_cov = cov[j]
            component_weight = weights[j]
            
            # Calculate the likelihood of the sample under the component
            # likelihood = multivariate_normal.pdf(data, component_mean, component_cov)
            component_cov += 1e-4 * np.eye(n_f)
            # Calculate the cluster likelihood
            difference = data - component_mean
            exponent = -0.5 * np.sum(np.dot(difference, np.linalg.inv(component_cov)) * difference, axis=1)
            likelihood = np.exp(exponent) / np.sqrt((2 * np.pi) ** n_f * np.maximum(1e-5,np.linalg.det(component_cov)))
        
            
            # Accumulate the likelihood weighted by the component's mixing coefficient
            sample_likelihood += component_weight * likelihood
        
            # Take the logarithm of the accumulated likelihood
            log_likelihoods = np.log(sample_likelihood)
        
        return np.sum(log_likelihoods)

    

    def predict(self, X_data, means, cov, weights_p):
        n_s, n_f = X_data.shape
        likelihood = np.zeros( (n_s, self.n_clusters) )
        for i in range(self.n_clusters):
            distribution = multivariate_normal(mean=means[i], cov=cov[i])
            likelihood[:,i] = distribution.pdf(X_data)
        
        numerator = likelihood * weights_p
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator

        return np.argmax(weights, axis=1)



class K_means:
    def __init__(self,n_clusters):
        self.means = []
        self.n_clusters = n_clusters

    def fit(self,X_data, max_iter=5,tol = 0.01):
        #initialize the centroid of clusters (Kxf)
        clusters = {}  
        # np.random.seed(11) 
        n_samples =  X_data.shape[0]
        for i in range(self.n_clusters):
            points = []
            center = 2*(2*np.random.random((X_data.shape[1],))-1)
            cluster = {'center' : center, 'points' : []}
            clusters[i] = cluster
        pre_s = 0
        for j in range(max_iter):
            #E-step(assign the data point to particular cluster based on Euclidian distance)
            for k in range(self.n_clusters):
                 clusters[k]['points'] = []
            for d_point in range(X_data.shape[0]):
                dis = []
                for k in range(self.n_clusters):
                    # d = self.distance(X_data[d_point], clusters[k]['center'] )
                    d = X_data[d_point] - np.array(clusters[k]['center'])
                    s = np.sum(d*d)

                    dis.append(s)
                curr_cluster = np.argmin(dis)
                clusters[curr_cluster]['points'].append(X_data[d_point])
            #convergence criteria (SSE)
            SSE = self.sum_of_squred_error(clusters)
            if SSE-pre_s<tol:
                break
            pre_s = SSE
            #M-step(find centroids of the clusters)
            for i in range(self.n_clusters):
                points = np.array(clusters[i]['points'])
                if points.shape[0] > 0:
                    new_center = points.mean(axis =0)
                    clusters[i]['center'] = new_center

        return clusters, SSE

    # def distance(self,p1,p2):
    #     return np.sqrt(np.sum((p1-p2)**2))

    def sum_of_squred_error(self,clusters):
        total = 0
        for k in range(self.n_clusters):
            points = np.array(clusters[k]['points'])
            if points.shape[0] > 0:
                d = points - np.array(clusters[k]['center'])
                s = np.sum(d*d)
                total += s
        return total

    def pred_cluster(self,X, clusters):
        pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(self.n_clusters):
                d = X[i] - np.array(clusters[j]['center'])
                s = np.sum(d*d)

                dist.append(s)
            pred.append(np.argmin(dist))
        return pred