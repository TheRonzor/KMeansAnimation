##########################################################################
#                                                                        #
#      May you do good and not evil                                      #
#      May you find forgiveness for yourself and forgive others          #
#      May you share freely, never taking more than you give.            #
#                                                                        #
##########################################################################

import kmaf
import rons_windows

# Create a new SimpleWindow
s = rons_windows.SimpleWindow(title='Animation Controls')

# Add the widgets:
#  - They will be added in order, left to right, then top to bottom.

#  - add_setting() takes as inputs:
#        label: The text you want to see next to each input box

#        inp_type: The datatype of the setting, e.g. int, str, float

#        default: Either a single value (which will create a tk.Entry box)
#                 or a tuple (which will crate a tk.OptionMenu, with the first value
#                 in the tuple as the default) 
#
#  - By default, the settings will be organized into 2 columns and as
#    many rows as needed. To change the number of columms, provide a value 
#    for num_columns when calling SimpleWindow() above.
#    
#  
#  - The "Go" button will be placed below these rows, centered horizontally.

#    This button is configured by calling the bind_go() function, and passing
#    in a function, followed by keyword arguments equal to the outputs of add_setting()

k           = s.add_setting('k (actual) = ', int, 3)
init_method = s.add_setting('Init. Method = ', str, kmaf.KMeansAnim.INIT_METHODS, input_width=10)
kg          = s.add_setting('k (guess) = ', int, 3)
seed        = s.add_setting('Seed = ', int, kmaf.np.random.randint(1000))
n           = s.add_setting('n = ', int, 100)
cmap        = s.add_setting('Colormap = ', str, 'rainbow', input_width=10)

# Tell the Go button what to run when pressed
s.bind_go(kmaf.KMeansAnim,
            n_clusters       = k,
            n_clusters_guess = kg,
            n_points         = n,
            method           = init_method,
            seed             = seed,
            cmap             = cmap
         )

# Show the window
s.deploy()