# Way back when...

import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.datasets import make_blobs as mb
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler as MMS

plt.style.use('dark_background')

__version__ = '0.0.1'

class KMeansAnim():
    FIG_SIZE    = (8,8)
    ALPHA_DATA  = 0.6
    ALPHA_CENT  = 0.9
    SIZE_DATA   = 20
    SIZE_CENT   = 500

    DT          = 1

    def __init__(self, 
                 n_points           = 500, 
                 n_clusters         = 3, 
                 n_clusters_guess   = None, 
                 method             = 'naive',
                 seed               = None):
        
        # Initialize settings
        self.n_points   = n_points
        self.n_clusters = n_clusters
        
        if n_clusters_guess is None:
            # Default guess is the clairvoyant one, since my AI is the best AI ;-)
            self.n_clusters_guess = self.n_clusters
        else:
            self.n_clusters_guess = n_clusters_guess
        
        if seed is None:
            self.seed = 42
        else:
            self.seed = seed
        
        self.rng = np.random.RandomState(self.seed)

        # Create the data
        self.data, self.y = mb(n_samples    = self.n_points, 
                               centers      = self.n_clusters, 
                               n_features   = 2,  
                               random_state = self.rng)
        self.data = MMS().fit_transform(self.data) # Rescale data to be in the unit square

        # Create the centroids
        if method == 'naive':
            self.centroids = self.rng.random(size=(self.n_clusters, 2))
        elif method == 'from_data':
            self.centroids = self.data[self.rng.choice(self.data.shape[0], replace=False, size=self.n_clusters_guess)]
        elif method == 'k++':
            raise(NotImplementedError())
        self.cluster_colors = get_cmap('rainbow')(np.linspace(0, 1, self.n_clusters_guess))[:,:-1] # Everything except the alpha

        # Initialize the figure objects
        self.fig, self.ax = plt.subplots(figsize=self.FIG_SIZE)

        self.scat_data = plt.scatter([], [], ec='k', alpha = self.ALPHA_DATA, s = self.SIZE_DATA)
        self.scat_cent = plt.scatter([], [], ec='w', s = self.SIZE_CENT, marker='*', linewidths=1)

        # Initialize the color arrays
        self.color_data = np.ones([len(self.data),3])
        self.color_cent = self.cluster_colors

        # Make the centroids invisible for now
        self.scat_cent.set_alpha(0)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(-0.01,1.01)
        self.ax.set_ylim(-0.01,1.01)
        self.ax.set_title('Initializing...')
        return

    def ShowInitialData(self):
        # Nothing here for now
        sleep(self.DT)
        return self.scat_data, self.scat_cent,

    def UpdateClusters(self):
        distances = []
        for c in self.centroids:
            d = np.sum((self.data - c)**2,axis=1)
            distances.append(d)
        self.labels = np.argmin(np.array(distances).T,axis=1)

        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)
            self.color_data[idx, :] = self.cluster_colors[label, :]
        return
    
    def UpdateCentroids(self):
        for i in range(self.n_clusters_guess):
            idx = self.labels == i
            self.centroids[i,:] = np.mean(self.data[idx,:], axis=0)
        return

    def Update(self, frame_number):
        if frame_number == 0:
            self.scat_cent.set_facecolor(self.color_cent)
            self.scat_cent.set_alpha(self.ALPHA_CENT)
            self.scat_data.set_offsets(self.data)
            sleep(self.DT)
        elif frame_number == 1:
            self.UpdateClusters()
            sleep(self.DT)
        else:
            self.UpdateClusters()
            self.UpdateCentroids()
            sleep(self.DT)

        self.ax.set_title('Frame ' + str(frame_number))
        
        self.scat_data.set_color(self.color_data)
        self.scat_cent.set_offsets(self.centroids)

        return self.scat_data, self.scat_cent, 

    def Go(self):
        self.ani = FuncAnimation(self.fig, 
                                 self.Update,
                                 init_func = self.ShowInitialData,
                                 blit=False,  # Can't seem to get blitting to work with titles, even if using ax.text inside the plot area :-(
                                 interval=33,
                                 cache_frame_data=True)
        plt.show()
        return

k = KMeansAnim()
k.Go()