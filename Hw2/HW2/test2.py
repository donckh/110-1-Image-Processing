import itertools
import math
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


# open file
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


# open image
def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((img_after_size[1], img_after_size[0]), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(div3, image=img)
    panel.image = img
    panel.grid(column=1, row=1, sticky=align_mode)


# test function
def s_print(text):		# print scale variable
    print(value1.get(), text)
    val1 = m.exp(value1.get())
    print(val1)
    val2 = m.exp(value2.get())
    print(val2)
    val3 = m.exp(value3.get())
    print(val3)


# create a window
window = tk.Tk()
window.title('Window')
align_mode = 'nswe'
pad = 5

# create a scale bar value
value1 = tk.IntVar()
value2 = tk.IntVar()
value3 = tk.IntVar()
value4 = tk.IntVar()

image_before = cv2.imread('./signs.jpg')
img_before_gray = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
arr = np.array(img_before_gray)
img_before_size = arr.shape
cv2.imwrite("signs_gray.png", img_before_gray)

image_after = copy.deepcopy(image_before)
img_after_temp = copy.deepcopy(image_before)
img_after_size = img_before_size

# print(img_before_size, img_after_size)
div_size = 200
div1 = tk.Frame(window,  width=div_size , height=div_size , bg='blue')
div2 = tk.Frame(window,  width=img_before_size[1], height=img_before_size[0], bg='orange')
div3 = tk.Frame(window,  width=img_after_size[1], height=img_after_size[0], bg='green')

window.update()
win_size = min( window.winfo_width(), window.winfo_height())
# print(win_size)

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

define_layout(window, cols=3, rows=3)
define_layout(div1)
define_layout(div2, rows=2)
define_layout(div3, rows=2)

# read image in right hand side
im = Image.open('./signs_gray.png')
imTK_R = ImageTk.PhotoImage( im.resize((img_after_size[1], img_after_size[0]) ))

image_main = tk.Label(div3, image=imTK_R)
image_main['height'] = img_after_size[0]
image_main['width'] = img_after_size[1]
image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)


# filter design
def initial_filter(fNum):
    filter = []
    for i in range(fNum):
        filter.append([0.0] * fNum)
    fil = np.array(filter)
    # fil_size = fil.shape
    # print('filter: ', fil)
    # print('filSize: ', fil_size)
    return fil


# create a gaussian filter
def gaussian_filter_creator():
    i = 0
    j = 0
    sum_of_filter = 0
    gaussian_size = int(value1.get())  # gaussain filter size(3) can be changed to value1.get()
    matrix_var = int(gaussian_size / 2)
    initial_gaussian_filter = initial_filter(gaussian_size)
    normalize_gaussian_filter = copy.deepcopy(initial_gaussian_filter)
    # print('matrix_var:', matrix_var, 'intial_gaussian_filter:', intial_gaussian_filter)
    for i,j in itertools.product(range(gaussian_size), range(gaussian_size)):
        initial_gaussian_filter[i][j] = m.exp(-((-matrix_var + i)**2 + (-matrix_var + j)**2))
        sum_of_filter = initial_gaussian_filter[i][j] + sum_of_filter
        # print('i:',i,'j:',j, 'intial_gaussian_filter:', intial_gaussian_filter[i][j],
        #      'normalize_gaussian_filter:', normalize_gaussian_filter[i][j], 'sum', np.sum(normalize_gaussian_filter))
    normalize_gaussian_filter = initial_gaussian_filter / sum_of_filter
    # print('normalize_gaussian_filter:', normalize_gaussian_filter, 'sum', np.sum(normalize_gaussian_filter))
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

        temp_sum = np.sum(temp)
        # print('sum:', int(temp_sum))
        cen_pointx = int(fil_size[0] / 2)
        cen_pointy = int(fil_size[1] / 2)
        num = int(temp_sum)
        if (i + fil_size[0] < img_size_c[0]) and (j + fil_size[1] < img_size_c[1]):
            #print('img_temp_c[', i + 1, '][', j + 1, ']: ', img_temp_c[i + 1][j + 1], 'temp_sum: ', num)
            # if num > 255:
            #     img_new_c[i + cen_pointx][j + cen_pointy] = 255
            # elif num < 0:
            #     img_new_c[i + cen_pointx][j + cen_pointy] = 0
            # else:
                img_new_c[i + cen_pointx][j + cen_pointy] = num
        temp.clear()
    return img_new_c


def sobel_filter(phase):
    if phase > 112.5:
        sobel_filter = [[2, 1, 0], [1, 0, -1], [0, -2, -1]]
    elif phase > 67.5:
        sobel_filter = [[1, 0, -1], [2, 0, -2], [1, 0, 1]]
    elif phase > 22.5:
        sobel_filter = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    else:
        sobel_filter = [[1, 2, 1],[0, 0, 0], [-1, -2, -1]]

    sobel_filter = np.array(sobel_filter)
    print('sobel_filter:', sobel_filter)
    # sobel_filter_size = sobel_filter.shape
    # print('sobel_filter:', sobel_filter, 'sobel_filter_size:', sobel_filter_size)
    return sobel_filter


def sobel_calculation(image_x, image_y):
    image = np.zeros(image_x.shape[0] * image_x.shape[1])
    image = image.reshape(image_x.shape[0], image_x.shape[1])
    for i, j in itertools.product(range(image_x.shape[0]), range(image_x.shape[1])):
        # if image[i][j] > 255:
        #     image[i][j] = 255
        # elif image[i][j] < 0:
        #     image[i][j] = 0
        # else:
            image[i][j] = int(math.sqrt(image_x[i][j] ** 2 + image_y[i][j] ** 2))
    for i, j in itertools.product(range(image_x.shape[0]), range(image_x.shape[1])):
        image[i][j] = int(image[i][j] / np.max(image) * 255)
        # if i == (image_x.shape[0] - 1) and j < (image_x.shape[1] - 1):
        #     print('image[i:][j]:', image[:][j])
    print('np.max(image):', np.max(image))
    return image


# main body
def main(value1):
    gaussian_filter = gaussian_filter_creator()
    gaussian_filter_size = gaussian_filter.shape
    print('gaussian_filter_size:', gaussian_filter_size, 'sum', np.sum(gaussian_filter))
    img_after_temp = conv_full_image(img_before_gray, gaussian_filter)
    cv2.imwrite("img_after_gaussian.png", img_after_temp)

    img_after_temp_y = np.zeros(img_after_temp.shape[0] * img_after_temp.shape[1])
    img_after_temp_y = img_after_temp_y.reshape(img_after_temp.shape[0] * img_after_temp.shape[1])
    img_after_temp_x = np.zeros(img_after_temp.shape[0] * img_after_temp.shape[1])
    img_after_temp_x = img_after_temp_x.reshape(img_after_temp.shape[0] * img_after_temp.shape[1])
    img_after_temp_combine = np.zeros(img_after_temp.shape[0] * img_after_temp.shape[1])
    img_after_temp_combine = img_after_temp_combine.reshape(img_after_temp.shape[0] * img_after_temp.shape[1])

    img_after_temp_x = conv_full_image(img_after_temp, sobel_filter(0))
    img_after_temp_y = conv_full_image(img_after_temp, sobel_filter(90))
    img_after_temp_combine = sobel_calculation(img_after_temp_x, img_after_temp_y)
    cv2.imwrite("img_after_sobel_x.png", img_after_temp_x)
    cv2.imwrite("img_after_sobel_y.png", img_after_temp_y)
    cv2.imwrite("img_after_sobel_combine.png", img_after_temp_combine)

    print('done, please open image')

# scale bar function
lbl_title1 = tk.Scale(div1, label='Gaussian Filter', from_=2, to=3, orient=tk.HORIZONTAL,
         resolution=1, variable=value1, command=main)

# lbl_title2 = tk.Scale(div1, label='Sobel Filter2', from_=2, to=9, orient=tk.HORIZONTAL,
#          resolution=1, variable=value2, command=s_print)

lbl_title3 = tk.Scale(div1, label='GaussianFilter3', from_=1, to=100, orient=tk.HORIZONTAL,
         resolution=10, variable=value3, command=s_print)

lbl_title4 = tk.Scale(div1, label='GaussianFilter4', from_=1, to=100, orient=tk.HORIZONTAL,
         resolution=10, variable=value4, command=s_print)

lbl_title1.grid(column=0, row=1, sticky=align_mode)
# lbl_title2.grid(column=0, row=2, sticky=align_mode)
lbl_title3.grid(column=0, row=3, sticky=align_mode)
lbl_title4.grid(column=0, row=4, sticky=align_mode)

print('initial')

# open after image
btn = tk.Button(window, text='open after image', command=open_img).grid(column=1, row=0)

window.mainloop()