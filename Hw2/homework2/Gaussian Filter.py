import itertools
import tkinter as tk
from PIL import Image, ImageTk
import math as m
import numpy as np
import copy
import cv2
from tkinter import filedialog

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


image_before = cv2.imread('./Pic2_before.png')
img_before_gray = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
img_before_temp = copy.deepcopy(image_before)
arr = np.array(img_before_gray)
img_before_size = arr.shape
cv2.imwrite("signs_gray.png", img_before_gray)

image_after = copy.deepcopy(image_before)
img_after_gray = copy.deepcopy(img_before_gray)
img_after_temp = copy.deepcopy(img_before_temp)
img_after_size = img_before_size

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

im = Image.open('./signs_gray.png') # read original image
imTK_L = ImageTk.PhotoImage( im.resize( (img_before_size[1], img_before_size[0]) ) )

image_main = tk.Label(div2, image=imTK_L)
image_main['height'] = img_before_size[0]
image_main['width'] = img_before_size[1]

image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)


# read image in right hand side
im = Image.open('./img_after_temp.png')
imTK_R = ImageTk.PhotoImage( im.resize((img_after_size[1], img_after_size[0]) ))

image_main = tk.Label(div3, image=imTK_R)
image_main['height'] = img_after_size[0]
image_main['width'] = img_after_size[1]
image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)



# scaler and filter design


def s_print(text):		# print scale variable
    print(value1.get(), text)
    val1 = m.exp(value1.get())
    print(val1)
    val2 = m.exp(value2.get())
    print(val2)
    val3 = m.exp(value3.get())
    print(val3)


def intial_filter(fNum):
    filter = []
    for i in range(fNum):
        filter.append([0.0] * fNum)
    fil = np.array(filter)
    fil_size = fil.shape
    #print('filter: ', fil)
    #print('filSize: ', fil_size)
    return fil


# create a gaussian filter
def gaussian_filter_creator(guassian_size):
    i = 0
    j = 0
    sum_of_filter = 0
    gaussian_size = int(value1.get())
    matrix_var = int(gaussian_size / 2)
    intial_gaussian_filter = intial_filter(gaussian_size)
    normalize_gaussian_filter = copy.deepcopy(intial_gaussian_filter)
    #print('matrix_var:', matrix_var, 'intial_gaussian_filter:', intial_gaussian_filter)
    for i,j in itertools.product(range(gaussian_size), range(gaussian_size)):
        intial_gaussian_filter[i][j] = m.exp(-((-matrix_var + i)**2+(-matrix_var + j)**2))
        sum_of_filter = intial_gaussian_filter[i][j] + sum_of_filter
        #print('i:',i,'j:',j, 'intial_gaussian_filter:', intial_gaussian_filter[i][j],
        #      'normalize_gaussian_filter:', normalize_gaussian_filter[i][j], 'sum', np.sum(normalize_gaussian_filter))
    normalize_gaussian_filter = intial_gaussian_filter / sum_of_filter
    #print('normalize_gaussian_filter:', normalize_gaussian_filter, 'sum', np.sum(normalize_gaussian_filter))
    return normalize_gaussian_filter


def conv_full_image(img, fil):
    temp = []
    img_temp = copy.deepcopy(img)
    img_size = img.shape
    fil_size = fil.shape
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            img_temp = conv(img, img_temp, img_size, fil, fil_size, temp, i, j)

    return img_temp


def conv(img_new_c, img_temp_c, img_size_c, fil, fil_size, temp, i, j):
    if fil_size[0] % 2 != 0:
        for ii in range(fil_size[0]):
            if i + fil_size[0] < img_size_c[0]:
                for jj in range(fil_size[1]):
                    if j + fil_size[1] < img_size_c[1]:
                        temp.append(img_temp_c[i + ii][j + jj] * fil[ii][jj])
                    else:
                        break
            else:
                break
        #print(temp)
        temp_sum = np.sum(temp)
        cen_pointx = int(fil_size[0] / 2)
        cen_pointy = int(fil_size[1] / 2)
        num = int(temp_sum)
        if (i + fil_size[0] < img_size_c[0]) and (j + fil_size[1] < img_size_c[1]):
            #print('img_temp_c[', i + 1, '][', j + 1, ']: ', img_temp_c[i + 1][j + 1], 'temp_sum: ', num)
            img_new_c[i + cen_pointx][j + cen_pointy] = num
        temp.clear()
    return img_new_c


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((img_after_size[1], img_after_size[0]), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(div3, image=img)
    panel.image = img
    panel.grid(column=1, row=1, sticky=align_mode)


# main body
def main(value1):
    gaussian_filter = gaussian_filter_creator(value1)
    gaussian_filter_size = gaussian_filter.shape
    print('gaussian_filter_size:', gaussian_filter_size, 'sum', np.sum(gaussian_filter))
    img_after_temp = conv_full_image(img_before_gray, gaussian_filter)
    cv2.imwrite("img_after_temp.png", img_after_temp)

    print('done, please open image')


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

print('initial')

# open after image
btn = tk.Button(window, text='open after image', command=open_img).grid(column=1, row=0)

window.mainloop()