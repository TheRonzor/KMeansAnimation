##########################################################################
#                                                                        #
#      May you do good and not evil                                      #
#      May you find forgiveness for yourself and forgive others          #
#      May you share freely, never taking more than you give.            #
#                                                                        #
##########################################################################

import tkinter as tk
from tkinter import messagebox
from traceback import print_exc

class SimpleWindow():
    '''
    A window with a settings area and a Go button.

    The settings are organized into 2 columns by default, this can be adjusted
    by passing in a value for num_columns.

    The geometry (size) of the window is a string, e.g. '400x200'. The window will be resized
    and centered once all the widgets have been added, but the behavior is a bit finnicky. Passing
    in a value for geometry might help if the window is not appearing as desired.
    '''

    LBL_ANCHOR          = tk.E  # Labels are anchored to the east
    INP_ANCHOR          = tk.W  # Input boxes are anchored to the west

    DFLT_LBL_WIDTH      = 5     # Default width of labels
    DFLT_INP_WIDTH      = 5     # Default width of inputs

    LBL_H               = 2     # Height of labels and inputs
    PADX                = 10    # Default x padding between things

    # Keys for settings dictionary
    SD_LABEL            = 'label'
    SD_INPUT            = 'input'
    SD_VALUE            = 'value'
    SD_TYPE             = 'type'

    def __init__(self,
                 geometry='500x150', 
                 title='A Simple Window', 
                 num_columns = 2
                 ):
        
        # Create the main window
        self.main = tk.Tk()
        self.main.geometry(geometry)
        self.main.title(title)

        # Partition the main window into a row for the settings and a row for the go button
        # We'll call this partition the "interface"
        self.interface = tk.Frame(self.main)
        self.interface.rowconfigure(0, weight=1) # The Settings grid will be the top row of the interface
        self.interface.rowconfigure(1, weight=1) # The GO button will be the bottom row of the interface

        # The settings frame is a subset of the interface frame,
        # and starts with a single row. Additional rows are added automatically as needed.
        self.settings = tk.Frame(self.interface)
        self.settings.rowconfigure(0, weight=1)
        self.settings.grid(row=0)

        # To keep track of which row and column we are on
        self.current_row = 0
        self.current_column = 0
        self.num_columns = num_columns

        # Dictionary to store all the settings
        self.settings_dict = {}

        # Place the Go button the bottom row of the interface
        go = tk.Button(self.interface, text="Let's go!", command=self.go, height=3)
        go.grid(row=1, sticky=tk.S)
        return
    
    def bind_go(self, fun, **kwargs):
        '''
        Pass in a function and keyword arguments.

        The name of the kwargs should be the name of the parameter in fun,
        and the value should be the name of the setting (these names are returned by AddSetting
        when creating the settings)
        '''
        self.go_fun = fun
        self.kwargs = kwargs
        return
    
    def add_setting(self, 
                    label, 
                    inp_type, 
                    default, 
                    label_width=DFLT_LBL_WIDTH, 
                    input_width=DFLT_INP_WIDTH
                    ):
        '''
        Create a label and input box and add them to the window.
        
        Inputs:

            label: The text which will appear in the label. These should be unique, since they
                   will also be used as the keys in the settings dictionary.

            inp_type: The datatype of the setting, e.g. int, str, float

            default: Either a single value (which will create a tk.Entry field) 
                     or a tuple (which will create a tk.OptionMenu).

            label_width: Not currently used. The labels seem to come out properly sized without it.

            input_width: Can adjust as needed if the input box is too small/large

        Output:

            label: The same value that was passed in, for convenience. This is also the key for the
                   setting in the settings dictionary.
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

        # Increment the column counter for the next setting
        self.current_column+=1
        # If the current row is full, create a new one and reset the column counter
        if self.current_column == self.num_columns:
            self.current_column = 0
            self.current_row+=1
            self.settings.rowconfigure(self.current_row, weight=1)
        
        return label
    
    def get_setting(self, label):
        '''
        Retrieves the current value for a setting.

        label is the same value that was used when the setting was created using AddSetting().
        '''

        # Get the setting from the dictionary
        s = self.settings_dict[label]
        # Get the datatype provided when the setting was created
        # This is used to convert the value into the proper data type,
        # since the values retrieved are always strings
        t = s[self.SD_TYPE]

        # Check what type of widget we are retreving values from
        # (currently, only tk.Entry and tk.OptionMenu are supported)
        tkt = type(s[self.SD_INPUT])

        if tkt == tk.Entry:
            return t(s[self.SD_INPUT].get())
        elif tkt == tk.OptionMenu:
            return t(s[self.SD_VALUE].get())
        else:
            raise NotImplementedError('Support for', tkt, 'not implemented.')
        
    def center_main(self):
        """
        Center and resize the main window. Should still set the initial geometry ahead of time, 
        and only call this function after all widgets have been added.
        
        Solution from:
            https://stackoverflow.com/questions/3352918/how-to-center-a-window-on-the-screen-in-tkinter
        """

        self.main.attributes('-alpha', 0.0)
        self.main.update_idletasks()
        width = self.main.winfo_width()
        frm_width = self.main.winfo_rootx() - self.main.winfo_x()
        win_width = width + 2 * frm_width
        height = self.main.winfo_height()
        titlebar_height = self.main.winfo_rooty() - self.main.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = self.main.winfo_screenwidth() // 2 - win_width // 2
        y = self.main.winfo_screenheight() // 2 - win_height // 2
        self.main.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.main.deiconify()
        self.main.attributes('-alpha', 1.0)
        
        return

    def deploy(self):
        '''Display the window. Call after everything else has been configured'''
        self.interface.pack(padx=20, pady=20)
        self.center_main()
        self.main.mainloop()
        return
    
    def go(self):
        '''
        Run the function assigned. This is configured by first calling bind_go().

        The kwargs are retrieved such that the key is the name of the parameter that
        was provided to bind_go(), and the value is the label provided to add_setting()
        '''
        try:
            local_kwargs = {}
            for key, value in self.kwargs.items():
                local_kwargs[key] = self.get_setting(value)
            self.go_fun(**local_kwargs)
        except Exception as e:
            print(print_exc())
            messagebox.showerror(title='Error', message=str(e))
        return