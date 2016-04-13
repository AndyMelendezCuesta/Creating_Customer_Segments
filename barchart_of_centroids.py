#google high dimensional centroids
#https://logfc.files.wordpress.com/2013/06/pca3d_centroids.png
#https://logfc.wordpress.com/tag/pca/
#http://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
#http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/ #more legible
#### draft ready for testing #####
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

import numpy as np

#items = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergent-Paper', 'Delicatessen'] #put in main
#function for more than two clusters (in two clusters there are two centroids)
def dpoints_func(items, kmeans_center):
    num_items = len(items)
    centroids = len(kmeans_center)
    count = 0
    counter = 1
    all_bars = []
    while count < num_items:
        while counter <= centroids:
            b = []
            b.append(items[count])
            b.append('centroid #{}'.format(counter))
            b.append(kmeans_center[counter-1][count])
            print b #testing b
            all_bars.append(b)
            print all_bars #testing all_bars
            counter +=1
        count +=1
    print all_bars #testing the final output of the function
    return all_bars

dpoints = np.array(all_bars) #for more than two clusters (in two clusters there are two centroids)

#for only two clusters (two centroids)
dpoints = np.array([['Fresh', '1st centroid', kmeans_center[0][0]], #b[0]
           ['Fresh', '2nd centroid', kmeans_center[1][0]],  #b[1]
           ['Milk', '1st centroid', kmeans_center[0][1]],   #b[2]
           ['Milk', '2nd centroid', kmeans_center[1][1]],   #b[3]
           ['Grocery', '1st centroid', kmeans_center[0][2]],#b[4]
           ['Grocery', '2nd centroid', kmeans_center[1][2]],#b[5]
           ['Frozen', '1st centroid', kmeans_center[0][3]], #b[6]
           ['Frozen', '2nd centroid', kmeans_center[1][3]], #b[7]
           ['Detergent-Paper', '1st centroid', kmeans_center[0][4]], #b[8]
           ['Detergent-Paper', '2nd centroid', kmeans_center[1][4]], #b[9]
           ['Delicatessen', '1st centroid', kmeans_center[0][5]], #b[10]
           ['Delicatessen', '2nd centroid', kmeans_center[1][5]]])#b[11]

# dpoints = np.array([['rosetta', '1mfq', 9.97],
#            ['rosetta', '1gid', 27.31],
#            ['rosetta', '1y26', 5.77],
#            ['rnacomposer', '1mfq', 5.55],
#            ['rnacomposer', '1gid', 37.74],
#            ['rnacomposer', '1y26', 5.77],
#            ['random', '1mfq', 10.32],
#            ['random', '1gid', 31.46],
#            ['random', '1y26', 18.16]])

fig = plt.figure()
ax = fig.add_subplot(111)

def barplot(ax, dpoints, centroids):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    
    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]
    
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
    
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
    ax.set_ylabel("Items sold")
    ax.set_xlabel("Item categories")
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
        
barplot(ax, dpoints)
savefig('barchart_{}.png'.format(str(centroids)))
plt.show()
