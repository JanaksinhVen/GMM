# GMM
The question-answer format repository for GMM:  

The EM algorithm is used for obtaining maximum likelihood estimates of parameters when some of the data is missing. More generally, however, the EM
algorithm can also be applied when there is latent, i.e. unobserved, data which
was never intended to be observed in the first place. In that case, we simply
assume that the latent data is missing and proceed to apply the EM algorithm.
The EM algorithm has many applications throughout statistics. It is often used
for example, in machine learning and data mining applications, and in Bayesian
statistics where it is often used to obtain the mode of the posterior marginal
distributions of parameters. [[Columbia University](http://www.columbia.edu/~mh2078/MachineLearningORFE/EM_Algorithm.pdf)]  

Membership value ric of a sample xi
is the probability that the sample
belongs to cluster c, in a given GMM (Gaussian Mixture Model). Likelihood
values for a set of samples, measures the likelihood of the given data under a
fixed model. In other words, likelihoods are about how likely the data is given
the model, while membership values are about how likely the model is given the
data. [[reference](https://hastie.su.domains/Papers/ESLII.pdf)]
Use only NumPy, Pandas, Matplotlib, and Plotly libraries for the tasks.
## Tasks
This task requires you to implement the EM algorithm for GMM and perform
clustering operations on a given dataset(s). The list of subtasks is given below.
1. Find the parameters of GMM associated with the customer-dataset, using the EM method. Vary the number of components, and observe the
results. Implement GMM in a class which has the routines to fit data (e.g.
gmm.fit(data, number of clusters)), a routine to obtain the parameters, a
routine to calculate the likelihoods for a given set of samples and a routine
to obtain the membership values of data samples.

2. Perform clustering on the wine-dataset using Gaussian Mixture Model
(GMM) and K-Means algorithms. Find the optimal number of clusters
for GMM using BIC (Bayesian Information Criterion) and AIC (Akaike
Information Criterion). Reduce the dataset dimension to 2 using Principal
Component Analysis (PCA), plot scatter plots for each of the clustering
mentioned above, analyze your observations and report them. Also, compute the silhouette scores for each clustering and compare the results.
You are free to use sklearn for the dataset, PCA, and Silhouette Score
computation.
