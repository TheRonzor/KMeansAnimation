##########################################################################
#  The author of this code claims no rights or responsibilities for it.  #
#  The code is provided as is, with the hope that you will enjoy and/or  #
#  learn from it.                                                        #
#                                                                        #
#  I would also like to share a blessing with you that was shared with   #
#  me through another random piece of code, long ago in a far away land: #
#                                                                        #
#      May you do good and not evil                                      #
#      May you find forgiveness for yourself and forgive others          #
#      May you share freely, never taking more than you give.            #
#                                                                        #
##########################################################################

import kmaf
import tkinter as tk
from tkinter import messagebox

#  Create a tkinter window with some settings and a Go button
#
#  Scroll to the bottom of the code to see just the basics of what you would need to adjust for your own needs
#
#  Use SimpleWindow.AddSetting() to create new settings:
#   - tk.Entry and tk.OptionMenu are the only options implemented (no checkboxes, etc. Slightly different logic depending on the widget)
#
#  Settings are added as a pair of labels and input boxes. They are added left to right, then top to bottom.
#  The default number of columns (2) can be modified by passing a value for num_columns when you instantiate SimpleWindow()


def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    
    # Solution from:
    # https://stackoverflow.com/questions/3352918/how-to-center-a-window-on-the-screen-in-tkinter
    
    """

    win.attributes('-alpha', 0.0)
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()
    win.attributes('-alpha', 1.0)
    
    return

class SimpleWindow():
    '''A window with a settings area and a Go button'''

    LBL_ANCHOR          = tk.E  # Labels are anchored to the west
    INP_ANCHOR          = tk.W  # Input boxes are anchored to the east

    DFLT_LBL_WIDTH      = 5     # Default width of labels
    DFLT_INP_WIDTH      = 5     # Default width of inputs

    LBL_H               = 2     # Height of labels and inputs
    PADX                = 10    # Default x padding between things

    # Keys for settings dictionary
    SD_LABEL            = 'label'
    SD_INPUT            = 'input'
    SD_VALUE            = 'value'
    SD_TYPE             = 'type'

    def __init__(self, geometry='500x150', title='A Simple Window', num_columns = 2):
        self.main = tk.Tk()
        self.main.geometry(geometry)

        self.main.title(title)

        # Partition the window into a row for the settings and a row for the go button
        # We'll call this partition the interface
        self.interface = tk.Frame(self.main)
        self.interface.rowconfigure(0, weight=1) # The Settings grid goes in this row
        self.interface.rowconfigure(1, weight=1) # The GO button gets its own row

        # The settings frame is a subset of the interface frame,
        # and starts with a single row (we'll add more rows as needed as we add widgets)
        self.settings = tk.Frame(self.interface)
        self.settings.rowconfigure(0, weight=1)
        self.settings.grid(row=0)

        # To keep track of which row and column
        self.current_row = 0
        self.current_column = 0
        self.num_columns = num_columns

        # Dictionary to store all the settings
        self.settings_dict = {}

        # The go button will go in the second row of the interface
        go = tk.Button(self.interface, text="Let's go!", command=self.Go, height=3)
        go.grid(row=1, sticky=tk.S)
        return
    
    def AddSetting(self, label, inp_type, default, label_width=DFLT_LBL_WIDTH, input_width=DFLT_INP_WIDTH):
        '''Add label and tk.Entry to the settings frame',
           and store a reference to the value in a dictionary.

           The values provided for 'label' should be unique, since they'll be the keys in the dictionary.

           For tk.Entry boxes, supply a single value for default
           for tk.OptionMenu, provide a tuple, with the default as the first element
        '''

        if type(default) != tuple:
            self.settings_dict[label] = {
                                         self.SD_LABEL: tk.Label(self.settings, text = label, height = self.LBL_H), #width=label_width),
                                         self.SD_INPUT: tk.Entry(self.settings, width = input_width),
                                         self.SD_TYPE: inp_type
                                        }
            self.settings_dict[label][self.SD_INPUT].insert(0, default)
        else:
            self.settings_dict[label] = {
                                         self.SD_LABEL: tk.Label(self.settings, text = label, height = self.LBL_H), #width=label_width),
                                         self.SD_VALUE: tk.StringVar(),
                                         self.SD_TYPE: inp_type
                                        }
            self.settings_dict[label][self.SD_VALUE].set(default[0])
            self.settings_dict[label][self.SD_INPUT] = tk.OptionMenu(self.settings, self.settings_dict[label][self.SD_VALUE], *default)
        
        # Grid settings
        self.settings_dict[label][self.SD_LABEL].grid(row=self.current_row, 
                                                      column=self.current_column*2, 
                                                      padx=self.PADX,
                                                      sticky = self.LBL_ANCHOR)
        
        self.settings_dict[label][self.SD_INPUT].grid(row=self.current_row, 
                                                      column=self.current_column*2+1, 
                                                      padx=self.PADX, 
                                                      sticky = self.INP_ANCHOR)

        self.current_column+=1
        if self.current_column == self.num_columns:
            self.current_column = 0
            self.current_row+=1
            self.settings.rowconfigure(self.current_row, weight=1)
        
        return label
    
    def GetSetting(self, label):
        s = self.settings_dict[label]
        t = s[self.SD_TYPE]
        tkt = type(s[self.SD_INPUT])

        if tkt == tk.Entry:
            return t(s[self.SD_INPUT].get())
        elif tkt == tk.OptionMenu:
            return t(s[self.SD_VALUE].get())
        else:
            raise NotImplementedError('Support for', tkt, 'not implemented.')

    def Deploy(self):
        self.interface.pack(padx=20, pady=20)
        center(self.main)                           
        self.main.mainloop()
        return
    
    def Go(self):
        try:
            kmaf.KMeansAnim(n_clusters       = self.GetSetting(k),
                            n_clusters_guess = self.GetSetting(kg),
                            n_points         = self.GetSetting(n),
                            method           = self.GetSetting(init_method),
                            seed             = self.GetSetting(seed),
                            cmap             = self.GetSetting(cmap)
                            ).Go()
        except Exception as e:
            messagebox.showwarning(title='Error', message=str(e))
        return

########################################################################## 
# To modify the code to do something other than the k-means animation,   #
# adjust the code below, as well as the body of the Go() function above  #
##########################################################################

# Create a new window
s = SimpleWindow(title='Animation Controls')

# Add the widgets
#  -They will be added in order, left to right, then top to bottom.
#  -For more columns, pass num_columns to SimpleWindow() above, the default is 2.

k           = s.AddSetting('k (actual) = ', int, 3)
init_method = s.AddSetting('Init. Method = ', str, kmaf.KMeansAnim.INIT_METHODS, input_width=10)
kg          = s.AddSetting('k (guess) = ', int, 3)
seed        = s.AddSetting('Seed = ', int, kmaf.np.random.randint(1000))
n           = s.AddSetting('n = ', int, 100)
cmap        = s.AddSetting('Colormap = ', str, 'rainbow', input_width=10)

# Show the window
s.Deploy()