# Import libraries: NumPy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # or: from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA #import PCA
from sklearn.decomposition import FastICA #import ICA
import seaborn as sns #Seaborn is a Python visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive statistical graphics.
from sklearn.cluster import KMeans # Import clustering modules
from sklearn.mixture import GMM # Import clustering modules

def pca_model(data, n):
    #PCA finds vectors that explains the variance of the data
    pca = PCA(n_components = n).fit(data)
    # print the components of the data
    print pd.DataFrame(pca.components_, columns = list(data.columns))
    #print the variance of the data
    print "------------------------------------------"
    print "Variance Ratio of Individual Components"
    print "------------------------------------------"
    #print columns names
    print list(data.columns)
    print pca.explained_variance_ratio_
    return pca
  

def pca_plot(pca):
    # plot the Variance Trend by PCA Dimensions graph 
    print "------------------------------------------" 
    plt.figure(figsize=(11,5))
    plt.title("Variance Trend by PCA Dimensions")
    plt.plot(np.arange(1,7),np.cumsum(pca.explained_variance_ratio_))
    plt.legend()
    plt.savefig("VarianceTrendbyPCADimensions.png")
    plt.xlabel("number of pca components")
    plt.ylabel("cumulative variance explained by each dimension")
    plt.show()

def ica_model(data, n):
    #using ICA
    ica = FastICA(n_components = n).fit(data)
    print "Output from ica_model: "
    print ica
    print "ICA components: "
    # print the components of the data
    print pd.DataFrame(ica.components_, columns = list(data.columns))
    return ica

def ica_plot(ica, data):
    # TODO: Fit an ICA model to the data
    # Note: Adjust the data to have center at the origin first!
    # import seaborn as sns #Seaborn is a Python visualization library based on matplotlib. 
    # Seaborn provides a high-level interface for drawing attractive statistical graphics. 

    #plotting heat map for better visualization of matrix
    #heatmap = plt.pcolor(data) #produces a table of the data but instead of values there are colors
    plt.figure(figsize = (11,5))
    plt.title("Heatmap for better visualization of matrix")
    sns.heatmap(pd.DataFrame(ica.components_, columns = list(data.columns)),annot = True)
    plt.legend()
    plt.savefig("HeatmapToVisualizeMatrix.png")
    plt.xlabel("independent components (ica values)")
    plt.ylabel("number of ica independent components")
    plt.show()

def cluster_code(clustersKMeans, clustersGMM):
    print "Here is the clustersKMeans (code): " + str(clustersKMeans)
    print "----------------"
    print "Here is the clustersGMM (code): " + str(clustersGMM)

def centroid_coordinates(pd, centroidsKMeans, centroidsGMM):
    print "centroids of clustersKMeans: "
    print pd.DataFrame(centroidsKMeans, columns = ["x-axis", "y-axis"])
    print "---------------"
    print "cluster means of clustersGMM: "
    print pd.DataFrame(centroidsGMM, columns = ["x-axis", "y-axis"])

def centroidsGMMplot(Z_GMM, xx, yy, x_min, x_max, y_min, y_max, reduced_data, centroidsGMM):
    #Putting the result into a color plot
    Z_GMM = Z_GMM.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z_GMM, interpolation='nearest', \
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),\
                   cmap=plt.cm.Paired,\
                   aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
    plt.scatter(centroidsGMM[:, 0], centroidsGMM[:, 1], \
                    marker='x', s=169, linewidths=3,\
                    color='w', zorder=10)
    plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data),Centroids marked w/white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("GMM_K_components-clusters.png")
    plt.show()

def centroidsKMeansplot(Z_KMeans, xx, yy, x_min, x_max, y_min, y_max, reduced_data, centroidsKMeans):
        #Putting the result into a color plot
    Z_KMeans = Z_KMeans.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z_KMeans, interpolation='nearest', \
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),\
                   cmap=plt.cm.Paired,\
                   aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
    plt.scatter(centroidsKMeans[:, 0], centroidsKMeans[:, 1],\
                    marker='x', s=169, linewidths=3,\
                    color='w', zorder=10)
    plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data),Centroids marked w/white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("centroidsKMeans_K_clusters.png")
    plt.show()

#In the case of the documentation page for GridSearchCV, it might be the case that the example is just a demonstration of syntax for use of the function, rather than a statement about 
def main():
    """Analyze the Wholesale customers data. Evaluate and utilize different methods for finding the best segmentation model.
    Use the best model to make prediction on unseen data."""

    # Read dataset
    data = pd.read_csv("wholesale-customers.csv")
    print "Dataset has {} rows, {} columns".format(*data.shape)
    print data.head()  # print the first 5 rows"


    #PCA evaluation. Parameters: data and number of components (n)
    pca = pca_model(data, 6)

    # plot the Variance Trend by PCA Dimensions graph
    pca_plot(pca)

    # TODO: Fit an ICA model to the data
    # Note: Adjust the data to have center at the origin first!
    centered_data = data.copy() - data.mean() #centering data at the origin
    
    #ICA evaluation. Parameters: centered data and number of components (n)
    ica = ica_model(centered_data, 6)

    #Heatmap
    ica_plot(ica, data)

    # TODO: First we reduce the data to two dimensions using PCA to capture variation
    reduced_data = PCA(n_components = 2).fit_transform(centered_data)
     #len(reduced_data) # it's equivalent to the number of samples in the .csv file, 440
    print pd.DataFrame(reduced_data[:10], columns = ["1st-dimension (x)", "2nd-dimension (y)"])


    # TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
    # The visualizer below assumes your clustering object is named 'clusters'
    #KMeans_n (n clusters), GMM_n (n components)
    num_clusters = 2
    num_components = 2

    clustersKMeans = KMeans(n_clusters = num_clusters).fit(reduced_data) #Adjust 3 or 4 or k clusters
    clustersGMM = GMM(n_components = num_components).fit(reduced_data) #Adjust 2 or 3 or 4  or k clusters

    #Print cluster means code
    cluster_code(clustersKMeans, clustersGMM)
    
    # Plot the decision boundary by building a mesh grid to populate a graph.
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    hx = (x_max-x_min)/1000
    hy = (y_max-y_min)/1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    # Obtain labels for each point in mesh. Use last trained model.
    Z_KMeans = clustersKMeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_GMM = clustersGMM.predict(np.c_[xx.ravel(), yy.ravel()])

    #print "Centroids (clustersKMeans) and cluster means (clustersGMM): "
    
    # TODO: Find the centroids for KMeans 
    centroidsKMeans = clustersKMeans.cluster_centers_
    centroidsGMM = clustersGMM.means_
    
    #Print centroidsKMeans and centroidsGMM coordinates
    centroid_coordinates(pd, centroidsKMeans, centroidsGMM)

    #############   centroidsKMeans (2, 3, 4, .., k clusters)  ############
    centroidsKMeansplot(Z_KMeans, xx, yy, x_min, x_max, y_min, y_max, reduced_data, centroidsKMeans)

    ############  centroidsGMM  (2, 3, 4, .., k clusters-components) ###############
    centroidsGMMplot(Z_GMM, xx, yy, x_min, x_max, y_min, y_max, reduced_data, centroidsGMM)
    
    # Done!
    print "Finished"



if __name__ == "__main__":
    # suppresses deprecation warning 
    #warnings.filterwarnings('ignore')
    main()

#OUTPUTS:
# Andreas-MacBook-Pro-2:customer_segments andreamelendezcuesta$ python edited_customer_segments.py
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/IPython/html.py:14: ShimWarning: The `IPython.html` package has been deprecated. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.
#   "`IPython.html.widgets` has moved to `ipywidgets`.", ShimWarning)
# Dataset has 440 rows, 6 columns
#    Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicatessen
# 0  12669  9656     7561     214              2674          1338
# 1   7057  9810     9568    1762              3293          1776
# 2   6353  8808     7684    2405              3516          7844
# 3  13265  1196     4221    6404               507          1788
# 4  22615  5410     7198    3915              1777          5185
#       Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicatessen
# 0 -0.976537 -0.121184 -0.061540 -0.152365          0.007054     -0.068105
# 1 -0.110614  0.515802  0.764606 -0.018723          0.365351      0.057079
# 2 -0.178557  0.509887 -0.275781  0.714200         -0.204410      0.283217
# 3 -0.041876 -0.645640  0.375460  0.646292          0.149380     -0.020396
# 4  0.015986  0.203236 -0.160292  0.220186          0.207930     -0.917077
# 5 -0.015763  0.033492  0.410939 -0.013289         -0.871284     -0.265417
# ------------------------------------------
# Variance Ratio of Individual Components
# ------------------------------------------
# ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
# [ 0.45961362  0.40517227  0.07003008  0.04402344  0.01502212  0.00613848]
# ------------------------------------------
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/axes/_axes.py:519: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
#   warnings.warn("No labelled objects found. "
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/linalg/basic.py:884: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
#   warnings.warn(mesg, RuntimeWarning)
# Output from ica_model: 
# FastICA(algorithm='parallel', fun='logcosh', fun_args=None, max_iter=200,
#     n_components=6, random_state=None, tol=0.0001, w_init=None,
#     whiten=True)
# ICA components: 
#           Fresh          Milk       Grocery        Frozen  Detergents_Paper  \
# 0 -8.652941e-07 -1.401485e-07  7.744414e-07  1.114606e-05     -5.567618e-07   
# 1  2.115546e-07 -1.896905e-06  6.354089e-06  4.204556e-07     -6.408347e-07   
# 2 -3.863546e-07 -2.195141e-07 -6.018605e-07 -5.220764e-07      5.111305e-07   
# 3  1.532954e-07  9.847029e-06 -5.804927e-06 -3.647531e-07      3.300554e-06   
# 4 -3.014910e-07  2.291457e-06  1.210044e-05 -1.460395e-06     -2.821402e-05   
# 5 -3.975732e-06  8.574337e-07  6.152927e-07  6.782096e-07     -2.041792e-06   

#    Delicatessen  
# 0     -0.000006  
# 1     -0.000001  
# 2      0.000018  
# 3     -0.000006  
# 4     -0.000006  
# 5      0.000001  
#    1st-dimension (x)  2nd-dimension (y)
# 0        -650.022122        1585.519090
# 1        4426.804979        4042.451509
# 2        4841.998707        2578.762176
# 3        -990.346437       -6279.805997
# 4      -10657.998731       -2159.725815
# 5        2765.961593        -959.870727
# 6         715.550892       -2013.002266
# 7        4474.583667        1429.496972
# 8        6712.095397       -2205.909156
# 9        4823.634354       13480.559205
# Here is the clustersKMeans (code): KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
#     n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
#     verbose=0)
# ----------------
# Here is the clustersGMM (code): GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#   n_components=2, n_init=1, n_iter=100, params='wmc', random_state=None,
#   thresh=None, tol=0.001, verbose=0)
#Centroids (clustersKMeans) and cluster means (clustersGMM): 
# centroids of clustersKMeans: 
#          x-axis       y-axis
# 0 -24088.332767  1218.179383
# 1   4175.311013  -211.151093
# ---------------
# cluster means of clustersGMM: 
#          x-axis       y-axis
# 0   3308.393018 -3017.017397
# 1 -10810.230089  9858.155324
# Finished
