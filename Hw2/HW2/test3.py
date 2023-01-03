import itertools
# from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import math as m
import numpy as np
import copy
import cv2
import os


window = tk.Tk()
window.geometry("550x300+300+150")
window.resizable(width=True, height=True)

div_size = 200
img_size = div_size * 2
div1 = tk.Frame(window,  width=img_size , height=img_size , bg='blue')
div2 = tk.Frame(window,  width=div_size , height=div_size , bg='orange')
div3 = tk.Frame(window,  width=div_size , height=div_size , bg='green')

div1.grid(column=0, row=0, rowspan=2)
div2.grid(column=1, row=0)
div3.grid(column=1, row=1)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.grid(column=1, row=1)

btn = tk.Button(window, text='open image', command=open_img).grid(column=1, row=0)

window.mainloop()
'''
# create a layout
def define_layout(obj, cols=1, rows=1):
    def method(trg, col, row):

        for c in range(cols):
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj) == list:
        [method(trg, cols, rows) for trg in obj]
    else:
        trg = obj
        method(trg, cols, rows)


window = tk.Tk()
window.title('Window')
align_mode = 'nswe'
pad = 5


image_before = cv2.imread('pic4PR2\signs.jpg')
img_before_gray = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
img_before_temp = copy.deepcopy(image_before)
arr = np.array(img_before_gray)
img_before_size = arr.shape

image_after = cv2.imread('pic4PR2\signs.jpg')
img_after_gray = cv2.cvtColor(image_after, cv2.COLOR_RGB2GRAY)
img_after_temp = copy.deepcopy(image_after)
arr = np.array(img_after_gray)
img_after_size = arr.shape
#cv2.imwrite("signs_gray.png", img_after_gray)

#print(img_before_size, img_after_size)
div_size = 200
div1 = tk.Frame(window,  width=div_size , height=div_size , bg='blue')
div2 = tk.Frame(window,  width=img_before_size[1], height=img_before_size[0], bg='orange')
div3 = tk.Frame(window,  width=img_after_size[1], height=img_after_size[0], bg='green')

window.update()
win_size = min( window.winfo_width(), window.winfo_height())
#print(win_size)

div1.grid(column=0, row=0, padx=pad, pady=pad, sticky=align_mode)
div2.grid(column=0, row=1, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
div3.grid(column=1, row=1, padx=pad, pady=pad, rowspan=2, sticky=align_mode)

define_layout(window, cols=2, rows=2)
define_layout([div1, div2, div3])

im = Image.open('./signs.jpg') # read original image
imTK_L = ImageTk.PhotoImage( im.resize( (img_before_size[1], img_before_size[0]) ) )

image_main = tk.Label(div2, image=imTK_L)
image_main['height'] = img_before_size[0]
image_main['width'] = img_before_size[1]

image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)

# create a scalebar
value1 = tk.IntVar()
value2 = tk.IntVar()
value3 = tk.IntVar()

lbl_title1 = tk.Scale(div1, label='GaussianFilter', from_=2, to=9, orient=tk.HORIZONTAL,
         resolution=1, variable=value1, command=main)

lbl_title2 = tk.Scale(div1, label='GaussianFilter2', from_=1, to=100, orient=tk.HORIZONTAL,
         resolution=10, variable=value2, command=s_print)

lbl_title3 = tk.Scale(div1, label='GaussianFilter3', from_=1, to=100, orient=tk.HORIZONTAL,
         resolution=10, variable=value3, command=s_print)


lbl_title1.grid(column=0, row=1, sticky=align_mode)
lbl_title2.grid(column=0, row=2, sticky=align_mode)
lbl_title3.grid(column=0, row=3, sticky=align_mode)

define_layout(window, cols=3, rows=3)
define_layout(div1)
define_layout(div2, rows=2)
define_layout(div3, rows=2)


# read image in right hand side
#im = Image.open('./img_after_temp.png')
imTK_R = ImageTk.PhotoImage( im.resize((img_after_size[1], img_after_size[0]) ))

image_main = tk.Label(div3, image=imTK_R)
image_main['height'] = img_after_size[0]
image_main['width'] = img_after_size[1]
image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)
print('initial')
'''

root.mainloop()