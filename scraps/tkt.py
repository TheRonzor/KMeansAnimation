# Figuring out the basics of tkinter...

import tkinter as tk


def DoButtonThing():
    print("I'm a real button!")

# Create the main window
main = tk.Tk()

# Size and title
main.geometry('640x640')
main.title('K-means Control Center')

label = tk.Label(main, text='Hello')
label.pack(padx=20, pady=20)

textb = tk.Text(main, height=1, width=3)
textb.pack(pady=50)

but = tk.Button(main, text='Click HERE', command=DoButtonThing)
but.pack(pady=10, padx=40)


# Make a frame to hold stuff
frame = tk.Frame(main)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

# Add stuff to the frame
btn1 = tk.Button(frame, text='abc')
btn1.grid(row=0, column=0, sticky=tk.N + tk.E)

txt1 = tk.Text(frame, height=1, width=3)
txt1.grid(row=0, column=1, sticky=tk.N + tk.E)


frame.pack()


# Do it
main.mainloop()