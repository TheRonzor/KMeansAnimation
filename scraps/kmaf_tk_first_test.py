import tkinter as tk
from tkinter import messagebox
import kmaf

LBL_ANCHOR = tk.E
INP_ANCHOR = tk.W

LBL_H  = 2
PADX   = 10

# The function that runs when we push the button
def Go():

    if input_cmap.get() not in kmaf.plt.colormaps():
        
        # Throws a strange warning. Also doesn't look very good on my macbook
        messagebox.showwarning(title='Invalid colormap', message='Select a valid matplotlib colormap')

    else:
        kmaf.KMeansAnim(n_clusters       = int(input_k.get()),
                        n_clusters_guess = int(input_kguess.get()),
                        n_points         = int(input_n.get()),
                        method           = method_value.get(),
                        seed             = int(input_seed.get()),
                        cmap             = input_cmap.get()
                        ).Go()
    return

# The main window
main = tk.Tk()
main.geometry('400x150')
main.title('Just what do you think you are doing, Dave?')

# Interface frame
inter = tk.Frame(main)
inter.rowconfigure(0, weight=1) # For the settings
inter.rowconfigure(1, weight=1) # For the Go button

# Settings frame goes inside interface frame
settings = tk.Frame(inter)
settings.rowconfigure(0, weight=1)
settings.rowconfigure(1, weight=1)
settings.rowconfigure(2, weight=1)
settings.grid(row=0)

# How many clusters?
label_k = tk.Label(settings, text='k (actual) = ', height=LBL_H)
label_k.grid(row=0, column=0, sticky = LBL_ANCHOR)
input_k = tk.Entry(settings, width=5)
input_k.grid(row=0, column=1, sticky = INP_ANCHOR)
input_k.insert(0, 3)

# What is our guess?
label_kguess = tk.Label(settings, text='k (guess) = ', height=LBL_H)
label_kguess.grid(row=1, column=0, sticky = LBL_ANCHOR)
input_kguess = tk.Entry(settings, width=5)
input_kguess.grid(row=1, column=1, sticky = INP_ANCHOR)
input_kguess.insert(0, 3)

# How many points?
label_n = tk.Label(settings, text='n = ', height=LBL_H)
label_n.grid(row=2, column=0, sticky = LBL_ANCHOR)
input_n = tk.Entry(settings, width=5)
input_n.grid(row=2, column=1, sticky = INP_ANCHOR)
input_n.insert(0, 100)

# Initialization method
label_meth = tk.Label(settings, text='Init. Method = ', height=LBL_H)
label_meth.grid(row=0, column=2, padx=PADX, sticky = LBL_ANCHOR)

method_value = tk.StringVar()
options_methods = ['naive', 'from_data', 'k++']
method_value.set(options_methods[-1])

input_meth = tk.OptionMenu(settings, method_value, *options_methods)
input_meth.grid(row=0, column=3,padx=PADX, sticky=INP_ANCHOR)

# Random seed
label_seed = tk.Label(settings, text='Seed = ', height=LBL_H)
label_seed.grid(row=1, column=2, padx=PADX, sticky=LBL_ANCHOR)
input_seed = tk.Entry(settings, width=5)
input_seed.grid(row=1, column=3, padx=PADX, sticky=INP_ANCHOR)

seed = kmaf.np.random.randint(1000)
input_seed.insert(0, seed)

# Colormap
label_cmap = tk.Label(settings, text='cmap = ', height=LBL_H)
label_cmap.grid(row=2, column=2, padx=PADX, sticky=LBL_ANCHOR)
input_cmap = tk.Entry(settings, width=10)
input_cmap.grid(row=2, column =3, padx=PADX, sticky=INP_ANCHOR)
input_cmap.insert(0,'rainbow')

# The Go button
go = tk.Button(inter, text="Let's go!", command=Go, height=3)
go.grid(row=1, sticky=tk.S)

# Pack it up
inter.pack(padx=20, pady=20)

# Ship it out
main.mainloop()