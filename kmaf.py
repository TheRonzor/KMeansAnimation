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

plt.style.use('dark_background')                            # it looks better in the dark

__version__ = '1.0.1'                                       # I might remember to update the version number occasionally

class KMeansAnim():
    __FIG_SIZE    = (8,8)
    
    # Marker settings
    #  Data
    __ALPHA_DATA  = 0.6
    __SIZE_DATA   = 20

    #  Centroids
    __ALPHA_CENT      = 0.9
    __SIZE_CENT       = 500

    # Initialization choices for centroids
    INIT_METHODS    = (
                      'k++',               # k-means++
                      'Uniform',           # select uniformly from data
                      'Naive'              # any point in the plot area (unit square)
                      )

    # Transition curve for animation, (1/f) where f = 1 + (1/x^a - 1)^k
    __SIGMOID_A   = 1.6       # "Timing"
    __SIGMOID_K   = 2.4       # "Sharpness"
    __SIGMOID_RES = 30        # Number of points
    __SIGMOID_MIN = 0.01      # Not zero
    __SIGMOID_MAX = 1.00      # Definitely one

    __DT          = 1.00      # For sleeps, in seconds. So the animation doesn't run too fast.
    
    def __init__(self, 
                 n_points: int           = 420, 
                 n_clusters: int         = 42, 
                 n_clusters_guess: int   = None, 
                 method: str             = INIT_METHODS[0],
                 seed: int               = None,
                 cmap: str               = 'rainbow'
                 ):
        
        # Initialize cluster settings
        self.__n_points   = n_points
        self.__n_clusters = n_clusters
        
        if n_clusters_guess is None:
            # Default guess is the clairvoyant one
            self.__n_clusters_guess = self.__n_clusters
        else:
            self.__n_clusters_guess = n_clusters_guess
        
        # Initialize PRNG
        if seed is None:
            self.__seed = np.random.randint(1000)
        else:
            self.__seed = seed
        print('Running with seed: ', self.__seed)
        self.__rng = np.random.RandomState(self.__seed)

        # Create the data
        self.__create_data()

        # Create the centroids
        if method not in self.INIT_METHODS:
            raise ('Unknown initialization method:', method)
        else:
            self.__method = method
        self.__create_centroids()
        self.__cluster_colors = get_cmap(cmap)(np.linspace(0, 1, self.__n_clusters_guess))[:,:-1] # Everything except the alpha

        # Initialize a placeholder for new centroids
        self.__new_centroids = self.__centroids.copy()

        # Transition curve for animations
        self.__sigmoid = 1/(1+(1/np.linspace(self.__SIGMOID_MIN, 
                                             self.__SIGMOID_MAX, 
                                             self.__SIGMOID_RES)**self.__SIGMOID_A-1)**self.__SIGMOID_K)
        self.__path_pos = self.__SIGMOID_RES-1

        # Initialize the figure
        self.__create_figure()

        # Run the animation
        self.__go()
        return
    
    def __create_data(self) -> None:
        '''
        Generate random blobs of data within a unit square
        '''
        self.__data, self.__y = mb(n_samples    = self.__n_points, 
                                   centers      = self.__n_clusters, 
                                   n_features   = 2,  
                                   random_state = self.__rng)
        self.__data = MMS().fit_transform(self.__data)
        return

    def __create_centroids(self) -> None:
        '''
        Generate the initial centroid positions
        '''
        if self.__method == 'Naive':
            self.__centroids = self.__rng.random(size=(self.__n_clusters_guess, 2))
        elif self.__method == 'Uniform':
            self.__centroids = self.__data[self.__rng.choice(self.__data.shape[0], replace=False, size=self.__n_clusters_guess)]
        elif self.__method == 'k++':
            # Select first centroid uniformly from the data
            self.__centroids = self.__data[self.__rng.choice(self.__data.shape[0]), :].reshape(1,-1)
            while self.__centroids.shape[0] < self.__n_clusters_guess:
                # Initialize distance array
                d = np.zeros(shape=[self.__data.shape[0], self.__centroids.shape[0]])
                
                # Compute squared distance from each point to each centroid
                for i, c in enumerate(self.__centroids):
                    d[:,i] = np.sum((self.__data-c)**2,axis=1)
                
                # Minimum squared distance for each point
                p = np.min(d, axis=1)
                
                # Select the next centroid from the data with probability proportional
                # to the squared distance to the closest centroid
                idx = self.__rng.choice(len(self.__data), p=p/sum(p))
                new_centroid = self.__data[idx,:].reshape(1,-1)
                self.__centroids = np.concatenate([self.__centroids, new_centroid], axis=0)
        else:
            # Should never happen [_]
            raise ValueError('Unknown initialization method:', self.__method)
        return
    
    def __create_figure(self) -> None:
        '''
        Initialize figure objects
        '''
        self.__fig, self.__ax = plt.subplots(figsize=self.__FIG_SIZE)

        # Initialize empty plots with display settings for data and centroids
        self.__scat_data = plt.scatter([], [], ec='k', alpha = self.__ALPHA_DATA, s = self.__SIZE_DATA)
        self.__scat_cent = plt.scatter([], [], ec='w', s = self.__SIZE_CENT, marker='*', linewidths=1.5)

        # Initialize the color arrays
        self.__color_data = np.ones([len(self.__data),3])
        self.__color_cent = self.__cluster_colors

        # Make the centroids invisible at first
        self.__scat_cent.set_alpha(0)

        self.__ax.set_xticks([])
        self.__ax.set_yticks([])
        self.__ax.set_xlim(-0.01,1.01)
        self.__ax.set_ylim(-0.01,1.01)
        return
    
    def __update_clusters(self) -> None:
        '''
        Update cluster labels according to their closest centroid
        '''
        distances = []
        for c in self.__centroids:
            d = np.sum((self.__data - c)**2,axis=1)
            distances.append(d)
        self.__labels = np.argmin(np.array(distances).T,axis=1)

        for label in np.unique(self.__labels):
            idx = np.where(self.__labels == label)
            self.__color_data[idx, :] = self.__cluster_colors[label, :]
        self.__scat_data.set_color(self.__color_data)
        return
    
    def __update_centroids(self) -> None:
        '''
        Compute new centroid positions (do not move the old ones yet)
        '''
        for i in range(self.__n_clusters_guess):
            idx = self.__labels == i
            if sum(idx) == 0:
                # Throw any unused centroids out of frame
                self.__new_centroids[i,:] = [5,5]
            else:
                self.__new_centroids[i,:] = np.mean(self.__data[idx,:], axis=0)
        return
    
    def __create_centroid_path(self) -> None:
        '''
        Create a path from the old centroids to the new ones (for animation purposes)
        '''
        self.__path_pos = 0
        self.__centroid_path = self.__centroids + np.array([si*(self.__new_centroids - self.__centroids) for si in self.__sigmoid])
        return
    
    def __move_along_centroid_path(self) -> None:
        '''
        Take a step along the path from the old centroids to the new ones (for animation purposes)
        '''
        self.__path_pos += 1
        self.__centroids = self.__centroid_path[self.__path_pos, :]
        return
    
    def __check_if_done(self) -> bool:
        '''
        Check if we are done moving from old centroid positions to new ones
        '''
        return np.max(np.abs(self.__centroids - self.__new_centroids)) == 0

    def __update(self, 
               frame_number: int
               )->tuple[matplotlib.collections.PathCollection]:
        '''
        Updater function for the animation
        '''
        if frame_number == 0:
            self.__scat_cent.set_facecolor(self.__color_cent)
            self.__scat_cent.set_alpha(self.__ALPHA_CENT)
            self.__scat_data.set_offsets(self.__data)
            self.__steps = 0
            self.__hold = True
        else:
            if self.__path_pos == self.__SIGMOID_RES-1:
                self.__steps += 1
                self.__ax.set_title('Iteration: ' + str(self.__steps))
                sleep(self.__DT)
                self.__update_clusters()
                self.__update_centroids()
                self.__create_centroid_path()
            else:
                self.__move_along_centroid_path()
                self.hold=True

        self.__scat_cent.set_offsets(self.__centroids)

        if self.__hold:
            self.__hold = False
        elif self.__check_if_done():
            self.__ax.set_title('We are done after ' + str(self.__steps-1) + ' iteration(s).')
            self.__ani.event_source.stop() 
        
        return self.__scat_data, self.__scat_cent, 

    def __go(self) -> None:
        '''
        Run the animation
        '''
        self.__ani = FuncAnimation(self.__fig, 
                                   self.__update,
                                   blit = False,  # Can't seem to get blitting to work when the figure has titles, even if using ax.text inside the plot area as many have recommended :-(
                                   interval = 10,
                                   cache_frame_data = True)
        plt.show()
        return

# In case we want to run it like a script.
if __name__ == '__main__':
    k = KMeansAnim()