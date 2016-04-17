#For converting the .ipynb file to .pdf: ipython nbconvert --to pdf customer_segments.ipynb  
#or                                      jupyter nbconvert --to pdf customer_segments.ipynb
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

#############################################################
################### Barchart of Centroids ###################
#Sources:
#* google Query: high dimensional centroids
#* https://logfc.files.wordpress.com/2013/06/pca3d_centroids.png
#* https://logfc.wordpress.com/tag/pca/
#* http://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
#* http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/ #more legible
#############################################################
#import matplotlib.pyplot as plt #already have it 
import matplotlib.cm as cm 
import operator as o ### install operator? #CHECK!
#import numpy as np #already have it
#############################################################
#############################################################

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

def find_clusters(clusters, reduced_data):
    clustersKMeans = KMeans(n_clusters = clusters).fit(reduced_data) #Adjust 3 or 4 or k clusters #original code
    clustersGMM = GMM(n_components = clusters).fit(reduced_data) #Adjust 2 or 3 or 4  or k clusters #original code
    print "Here is the clustersKMeans (code): " + str(clustersKMeans)
    print "----------------"
    print "Here is the clustersGMM (code): " + str(clustersGMM)
    #return the centroids for clustersKMeans and clustersGMM
    return (clustersKMeans, clustersGMM)

def centroid_coordinates(pd, centroidsKMeans, centroidsGMM):
    print "centroids of clustersKMeans: "
    print pd.DataFrame(centroidsKMeans, columns = ["x-axis", "y-axis"])
    print "---------------"
    print "cluster means of clustersGMM: "
    print pd.DataFrame(centroidsGMM, columns = ["x-axis", "y-axis"])

def centroidsGMMplot(clusters, reduced_data, Z_GMM, xx, yy, x_min, x_max, y_min, y_max, centroidsGMM):
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
    plt.savefig("GMM_{}_components-clusters.png".format(clusters)) #k components, where k is clusters
    plt.show()

def centroidsKMeansplot(clusters, reduced_data, Z_KMeans, xx, yy, x_min, x_max, y_min, y_max, centroidsKMeans):
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
    plt.savefig("centroidsKMeans_{}_clusters.png".format(clusters)) #k components, where k is clusters
    plt.show()


def all_centroids_plot(pd, clusters, reduced_data):
    ###plotting
    count = clusters #counts items
    while count >= 2:
        #clusters for KMeans and GMM
        print "------------------------------------------------"
        print "number of clusters: " + str(count) #Print number of clusters
        clustersKMeans, clustersGMM = find_clusters(count, reduced_data)
        #clustersKMeans = find_clusters(count, reduced_data)[0]
        #clustersGMM= find_clusters(count, reduced_data)[1]
        # TODO: Find the centroids for KMeans and GMM
        centroidsKMeans = clustersKMeans.cluster_centers_
        centroidsGMM = clustersGMM.means_
        # Plot the decision boundary by building a mesh grid to populate a graph.
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1#original code
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1#original code
        hx = (x_max-x_min)/1000
        hy = (y_max-y_min)/1000
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
        # Obtain labels for each point in mesh. Use last trained model.
        Z_KMeans = clustersKMeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_GMM = clustersGMM.predict(np.c_[xx.ravel(), yy.ravel()])
        #Print centroidsKMeans and centroidsGMM coordinates
        centroid_coordinates(pd, centroidsKMeans, centroidsGMM)
        #############   centroidsKMeans (2, 3, 4, .., k clusters)  ############
        centroidsKMeansplot(count, reduced_data, Z_KMeans, xx, yy, x_min, x_max, y_min, y_max, centroidsKMeans)
        ############  centroidsGMM  (2, 3, 4, .., k clusters-components) ###############
        centroidsGMMplot(count, reduced_data, Z_GMM, xx, yy, x_min, x_max, y_min, y_max, centroidsGMM)
        count -=1

#############################################################
################### Barchart of Centroids ###################
#############################################################

####Bringing the centroids back to its original dimensions####
def original_dimensions(pca, centered_data, clusters):
    # similar process to code 293
    #print "----------- Change1 from question #5 ------------"
    kmeans = KMeans(n_clusters=clusters).fit(centered_data) 
    centers_kmeans = pca.inverse_transform(kmeans.cluster_centers_)
    #print (centers_kmeans)
    #print "printing centers_kmeans[0]"
    #print centers_kmeans[0]
    #print "----------- Change2 from question #5 ------------"
    GMMclusters = GMM(n_components = clusters).fit(centered_data) 
    centers_GMM = pca.inverse_transform(GMMclusters.means_)
    #print(centers_GMM)
    #print "printing centers_GMM[0]"
    #print centers_GMM[0]
    #print "printing (centers_kmeans, centers_GMM)"
    #print (centers_kmeans, centers_GMM)
    return (centers_kmeans, centers_GMM)

#### List of Values to plot####
def dpoints_func(items, centers_kmeans, centers_GMM):
    num_items = len(items)
    centroids = len(centers_kmeans)# Equivalent to centroids_GMM = len(centers_GMM)
    all_bars = []
    count = 0 #counts items
    while count < num_items:
        counter = 1 #counter kmeans and GMM centroids 
        while counter <= centroids:
            b_GMM = []
            b_GMM.append(items[count])
            b_GMM.append('centroid_GMM#{}'.format(counter))
            b_GMM.append(centers_GMM[counter-1][count])
            all_bars.append(b_GMM) #appending to the array of all bars
            counter +=1
        counter = 1 #counter kmeans and GMM centroids 
        while counter <= centroids:
            b_kmeans = []
            b_kmeans.append(items[count])
            b_kmeans.append('centroid_kmeans#{}'.format(counter))
            b_kmeans.append(centers_kmeans[counter-1][count])
            all_bars.append(b_kmeans) #appending to the array of all bars
            counter +=1
        count +=1
    #print "printing all_bars"
    #print all_bars #testing the final output of the function
    return all_bars

#### Barplot Categories ####
def barplot_categories(clusters):
    #creating a list of categories 
    all_categories = []
    count_GMM =  1 #counts
    while count_GMM <= clusters:
        all_categories.append('centroid_GMM#{}'.format(count_GMM))
        count_GMM +=1
    count_KMeans = 1 #counts
    while count_KMeans <= clusters:
        all_categories.append('centroid_kmeans#{}'.format(count_KMeans))
        count_KMeans +=1
    return all_categories
    #print "all_categories:"
    #print all_categories
    # all_categories:
    # ['centroid_GMM#1', 'centroid_kmeans#1', 'centroid_GMM#2', 'centroid_kmeans#2']

#### General barplot ####
def barplot(ax, dpoints, items, clusters):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    #removed the code before dpoints and reassigned values to conditions and categories
    #original code: http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
    conditions = items
    categories = barplot_categories(clusters)

    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))
    
    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))
    
    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories)
    plt.setp(plt.xticks()[1], rotation=90)
    
    # Add the axis labels
    ax.set_ylabel("Consumed Items")
    ax.set_xlabel("Item categories")
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')

#### Barchart of all KMeans and GMM centroids ####
def all_barcharts(pca, centered_data, clusters, items):
    ###plotting
    count = clusters #counts items
    while count >= 2:
        centers_kmeans = original_dimensions(pca, centered_data, count)[0]
        centers_GMM = original_dimensions(pca, centered_data, count)[1]
        dpoints = np.array(dpoints_func(items, centers_kmeans, centers_GMM))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        barplot(ax, dpoints, items, count)
        plt.savefig('barchart_{}clusters.png'.format(count))
        plt.show()    
        count -=1

#############################################################
#############################################################

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

    #Added after the review Change in code #2
    #centered_data /= centered_data.std()
    unit_variance = centered_data / centered_data.std()

    #ICA evaluation. Parameters: centered data and number of components (n)
    #ica = ica_model(centered_data, 6) #original code
    ica = ica_model(unit_variance, 6) #modified after review

    #Heatmap
    ica_plot(ica, data)

    # TODO: First we reduce the data to two dimensions using PCA to capture variation
    reduced_data = PCA(n_components = 2).fit_transform(centered_data)#original code centered_data
    
     #len(reduced_data) # it's equivalent to the number of samples in the .csv file, 440
    print pd.DataFrame(reduced_data[:10], columns = ["1st-dimension (x)", "2nd-dimension (y)"]) #original code

    # TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
    # The visualizer below assumes your clustering object is named 'clusters'
    #KMeans_n (n clusters), GMM_n (n components)
    clusters = 4    

    #### currently I am only including the centers_kmeans. Later on I will include centers_GMM too
    print "------------------------------------------------"
    print "Plot of centroidsKMeans and centroidsGMM" #Bar chart demo with pairs of bars grouped for easy comparison
       
    # TODO: Find the centroids for KMeans and GMM clusters (included in all_centroids_plot function, check LINE 150)
    all_centroids_plot(pd, clusters, reduced_data)

    print "-----------------------------------------------------"
    print "Bar chart of the centroids in the original dimensions" #Bar chart demo with pairs of bars grouped for easy comparison
    
    #argument of all_barcharts function
    items = list(data.columns)
    #print "items:"
    #print items
    # items:
    # ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

    #one simple function to print barcharts of Kmeans clusters and GMM clusters with multiple k values
    all_barcharts(pca, centered_data, clusters, items)
    print "-----------------------------------------------------"

    # Done!
    print "Finished"


if __name__ == "__main__":
    # suppresses deprecation warning 
    #warnings.filterwarnings('ignore')
    main()

#Output (after modification of code. Prints the plots and barcharts of centroidsKMeans and centroidsGMM):
# -------------------------------------- RESULTS ----------------------------------------
# Andreas-MacBook-Pro-2:customer_segments andreamelendezcuesta$ python customer_segments_Edited.py
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
#       Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicatessen
# 0  0.003808 -0.016937 -0.114908  0.007092          0.134513      0.016171
# 1  0.004887  0.001620  0.005717  0.002535         -0.002436     -0.051024
# 2 -0.050283  0.006331  0.005868  0.003292         -0.009757      0.002953
# 3  0.010943  0.001035 -0.007359 -0.054111          0.002653      0.016787
# 4  0.002673 -0.013982  0.060533  0.002031         -0.003227     -0.004025
# 5  0.001940  0.072672 -0.055176 -0.001770          0.015753     -0.017088
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
# ------------------------------------------------
# Plot of centroidsKMeans and centroidsGMM
# ------------------------------------------------
# number of clusters: 4
# Here is the clustersKMeans (code): KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=4, n_init=10,
#     n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
#     verbose=0)
# ----------------
# Here is the clustersGMM (code): GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#   n_components=4, n_init=1, n_iter=100, params='wmc', random_state=None,
#   thresh=None, tol=0.001, verbose=0)
# centroids of clustersKMeans: 
#          x-axis        y-axis
# 0   3496.788187  -5024.808114
# 1   6166.173051  11736.813841
# 2 -14526.876149  50607.641373
# 3 -23984.557618  -4910.936734
# ---------------
# cluster means of clustersGMM: 
#          x-axis        y-axis
# 0   2339.152042  -6708.930657
# 1   7174.547193   5469.028765
# 2 -15372.371943  -3334.433799
# 3  -9486.974257  34645.204282
# ------------------------------------------------
# number of clusters: 3
# Here is the clustersKMeans (code): KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10,
#     n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
#     verbose=0)
# ----------------
# Here is the clustersGMM (code): GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#   n_components=3, n_init=1, n_iter=100, params='wmc', random_state=None,
#   thresh=None, tol=0.001, verbose=0)
# centroids of clustersKMeans: 
#          x-axis        y-axis
# 0   1341.311246  25261.391897
# 1   4165.121782  -3105.158115
# 2 -23978.865666  -4445.566118
# ---------------
# cluster means of clustersGMM: 
#          x-axis        y-axis
# 0   6987.950791   4249.829140
# 1    269.053187  -6506.886834
# 2 -17879.186238  10122.792466
# ------------------------------------------------
# number of clusters: 2
# Here is the clustersKMeans (code): KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10,
#     n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
#     verbose=0)
# ----------------
# Here is the clustersGMM (code): GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#   n_components=2, n_init=1, n_iter=100, params='wmc', random_state=None,
#   thresh=None, tol=0.001, verbose=0)
# centroids of clustersKMeans: 
#          x-axis       y-axis
# 0 -24088.332767  1218.179383
# 1   4175.311013  -211.151093
# ---------------
# cluster means of clustersGMM: 
#          x-axis       y-axis
# 0   3308.393018 -3017.017397
# 1 -10810.230089  9858.155324
# -----------------------------------------------------
# Bar chart of the centroids in the original dimensions
# -----------------------------------------------------
# Finished

# ------------------------------------- END ----------------------------------------




