##########################################################################
#                                                                        #
#      May you do good and not evil                                      #
#      May you find forgiveness for yourself and forgive others          #
#      May you share freely, never taking more than you give.            #
#                                                                        #
##########################################################################


        ##############################
        #         K-means            #
        # The Fun and Fancy Version! #
        ##############################

# This code runs an animated example of K-means.
# It is specifically designed for demonstration purposes,
# and is not intended for any actual analysis!

import numpy as np                                          
from time import sleep
import matplotlib                                      
import matplotlib.pyplot as plt                             # we're making pictures
from matplotlib.cm import get_cmap                          # get_cmap returns a colormap object which can be used to easily generate colors
from sklearn.datasets import make_blobs as mb               # this function makes blobs of data
from matplotlib.animation import FuncAnimation              # animation driver, it's kind of weird
from sklearn.preprocessing import MinMaxScaler as MMS       # all of the action will happen in the unit square

from typing_extensions import Literal


import cProfile, pstats, io
from pstats import SortKey

plt.style.use('dark_background')                            # it looks better in the dark
matplotlib.use('TkAgg')

__version__ = '1.0.1'                                       # I might remember to update the version number occasionally

class KMeansAnim():
    _FIG_SIZE    = (8,8)
    
    # Marker settings
    #  Data
    _ALPHA_DATA  = 0.6
    _SIZE_DATA   = 20

    #  Centroids
    _ALPHA_CENT      = 0.9
    _SIZE_CENT       = 500

    # Initialization choices for centroids
    INIT_METHODS    = (
                      'k++',               # k-means++
                      'Uniform',           # select uniformly from data
                      'Naive'              # any point in the plot area (unit square)
                      )

    # Transition curve for animation, (1/f) where f = 1 + (1/x^a - 1)^k
    _SIGMOID_A   = 1.6       # "Timing"
    _SIGMOID_K   = 2.4       # "Sharpness"
    _SIGMOID_RES = 30        # Number of points
    _SIGMOID_MIN = 0.01      # Not zero
    _SIGMOID_MAX = 1.00      # Definitely one

    _DT          = 1.00      # For sleeps, in seconds. So the animation doesn't run too fast.
    
    def __init__(self, 
                 n_points: int           = 420, 
                 n_clusters: int         = 42, 
                 n_clusters_guess: int   = None, 
                 method: str             = INIT_METHODS[0],
                 seed: int               = None,
                 cmap: str               = 'rainbow'
                 ):
        
        # Initialize cluster settings
        self._n_points   = n_points
        self._n_clusters = n_clusters
        
        if n_clusters_guess is None:
            # Default guess is the clairvoyant one
            self._n_clusters_guess = self._n_clusters
        else:
            self._n_clusters_guess = n_clusters_guess
        
        # Initialize PRNG
        if seed is None:
            self._seed = np.random.randint(1000)
        else:
            self._seed = seed
        print('Running with seed: ', self._seed)
        self._rng = np.random.RandomState(self._seed)

        # Create the data
        self._create_data()

        # Create the centroids
        if method not in self.INIT_METHODS:
            raise ('Unknown initialization method:', method)
        else:
            self._method = method
        self._create_centroids()
        self._cluster_colors = get_cmap(cmap)(np.linspace(0, 1, self._n_clusters_guess))[:,:-1] # Everything except the alpha

        # Initialize a placeholder for new centroids
        self._new_centroids = self._centroids.copy()

        # Transition curve for animations
        self._sigmoid = 1/(1+(1/np.linspace(self._SIGMOID_MIN, 
                                             self._SIGMOID_MAX, 
                                             self._SIGMOID_RES)**self._SIGMOID_A-1)**self._SIGMOID_K)
        self._path_pos = self._SIGMOID_RES-1

        # Initialize the figure
        self._create_figure()

        # Run the animation
        print("backend:", plt.rcParams["backend"])
        self._go()
        return
    
    def _create_data(self) -> None:
        '''
        Generate random blobs of data within a unit square
        '''
        self._data, self._y = mb(n_samples    = self._n_points, 
                                   centers      = self._n_clusters, 
                                   n_features   = 2,  
                                   random_state = self._rng)
        self._data = MMS().fit_transform(self._data)
        return

    def _create_centroids(self) -> None:
        '''
        Generate the initial centroid positions
        '''
        if self._method == 'Naive':
            self._centroids = self._rng.random(size=(self._n_clusters_guess, 2))
        elif self._method == 'Uniform':
            self._centroids = self._data[self._rng.choice(self._data.shape[0], replace=False, size=self._n_clusters_guess)]
        elif self._method == 'k++':
            # Select first centroid uniformly from the data
            self._centroids = self._data[self._rng.choice(self._data.shape[0]), :].reshape(1,-1)
            while self._centroids.shape[0] < self._n_clusters_guess:
                # Initialize distance array
                d = np.zeros(shape=[self._data.shape[0], self._centroids.shape[0]])
                
                # Compute squared distance from each point to each centroid
                for i, c in enumerate(self._centroids):
                    d[:,i] = np.sum((self._data-c)**2,axis=1)
                
                # Minimum squared distance for each point
                p = np.min(d, axis=1)
                
                # Select the next centroid from the data with probability proportional
                # to the squared distance to the closest centroid
                idx = self._rng.choice(len(self._data), p=p/sum(p))
                new_centroid = self._data[idx,:].reshape(1,-1)
                self._centroids = np.concatenate([self._centroids, new_centroid], axis=0)
        else:
            # Should never happen [_]
            raise ValueError('Unknown initialization method:', self._method)
        return
    
    def _create_figure(self) -> None:
        '''
        Initialize figure objects
        '''
        self._fig, self._ax = plt.subplots(figsize=self._FIG_SIZE)

        # Initialize empty plots with display settings for data and centroids
        self._scat_data = plt.scatter([], [], ec='k', alpha = self._ALPHA_DATA, s = self._SIZE_DATA)
        self._scat_cent = plt.scatter([], [], ec='w', s = self._SIZE_CENT, marker='*', linewidths=1.5)

        # Initialize the color arrays
        self._color_data = np.ones([len(self._data),3])
        self._color_cent = self._cluster_colors

        # Make the centroids invisible at first
        self._scat_cent.set_alpha(0)

        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_xlim(-0.01,1.01)
        self._ax.set_ylim(-0.01,1.01)
        return
    
    def _update_clusters(self) -> None:
        '''
        Update cluster labels according to their closest centroid
        '''
        distances = []
        for c in self._centroids:
            d = np.sum((self._data - c)**2,axis=1)
            distances.append(d)
        self._labels = np.argmin(np.array(distances).T,axis=1)

        for label in np.unique(self._labels):
            idx = np.where(self._labels == label)
            self._color_data[idx, :] = self._cluster_colors[label, :]
        self._scat_data.set_color(self._color_data)
        return
    
    def _update_centroids(self) -> None:
        '''
        Compute new centroid positions (do not move the old ones yet)
        '''
        for i in range(self._n_clusters_guess):
            idx = self._labels == i
            if sum(idx) == 0:
                # Throw any unused centroids out of frame
                self._new_centroids[i,:] = [5,5]
            else:
                self._new_centroids[i,:] = np.mean(self._data[idx,:], axis=0)
        return
    
    def _create_centroid_path(self) -> None:
        '''
        Create a path from the old centroids to the new ones (for animation purposes)
        '''
        self._path_pos = 0
        self._centroid_path = self._centroids + np.array([si*(self._new_centroids - self._centroids) for si in self._sigmoid])
        return
    
    def _move_along_centroid_path(self) -> None:
        '''
        Take a step along the path from the old centroids to the new ones (for animation purposes)
        '''
        self._path_pos += 1
        self._centroids = self._centroid_path[self._path_pos, :]
        return
    
    def _check_if_done(self) -> bool:
        '''
        Check if we are done moving from old centroid positions to new ones
        '''
        return np.max(np.abs(self._centroids - self._new_centroids)) == 0

    def _update(self, 
               frame_number: int
               )->tuple[matplotlib.collections.PathCollection]:
        '''
        Updater function for the animation
        '''
        if frame_number == 0:
            self._scat_cent.set_facecolor(self._color_cent)
            self._scat_cent.set_alpha(self._ALPHA_CENT)
            self._scat_data.set_offsets(self._data)
            self._steps = 0
            self._hold = True
        else:
            if self._path_pos == self._SIGMOID_RES-1:
                self._steps += 1
                self._ax.set_title('Iteration: ' + str(self._steps))
                sleep(self._DT)
                self._update_clusters()
                self._update_centroids()
                self._create_centroid_path()
            else:
                self._move_along_centroid_path()
                self._hold=True

        self._scat_cent.set_offsets(self._centroids)

        if self._hold:
            self._hold = False
        elif self._check_if_done():
            self._ax.set_title('We are done after ' + str(self._steps-1) + ' iteration(s).')
            self._ani.event_source.stop() 
        
        return self._scat_data, self._scat_cent, 

    def _go(self) -> None:
        '''
        Run the animation
        '''
        self._ani = FuncAnimation(self._fig, 
                                   self._update,
                                   blit = False,  # Can't seem to get blitting to work when the figure has titles, even if using ax.text inside the plot area as many have recommended :-(
                                   interval = 10,
                                   cache_frame_data = False
                                   )
        plt.show()
        return

# In case we want to run it like a script.
if __name__ == '__main__':

    pr = cProfile.Profile()
    pr.enable()
    
    
    k = KMeansAnim()
    
    
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    