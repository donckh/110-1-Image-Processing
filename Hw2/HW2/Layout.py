import tkinter as tk
from PIL import Image, ImageTk
import math as m

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


def s_print(text):		# print scale variable
    print(value1.get(), text)
    val1 = m.exp(value1.get())
    print(val1)
    val2 = m.exp(value2.get())
    print(val2)


window = tk.Tk()
window.title('Window')
align_mode = 'nswe'
pad = 5

div_size = 200
img_size = div_size * 2
div1 = tk.Frame(window,  width=div_size , height=div_size , bg='blue')
div2 = tk.Frame(window,  width=img_size , height=img_size , bg='orange')
div3 = tk.Frame(window,  width=img_size , height=img_size , bg='green')

window.update()
win_size = min( window.winfo_width(), window.winfo_height())
print(win_size)

div1.grid(column=0, row=0, padx=pad, pady=pad, sticky=align_mode)
div2.grid(column=0, row=1, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
div3.grid(column=1, row=1, padx=pad, pady=pad, rowspan=2, sticky=align_mode)

define_layout(window, cols=2, rows=2)
define_layout([div1, div2, div3])

im = Image.open('./signs.jpg')
imTK_L = ImageTk.PhotoImage( im.resize( (img_size, img_size) ) )

image_main = tk.Label(div2, image=imTK_L)
image_main['height'] = img_size
image_main['width'] = img_size

image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)

im = Image.open('./balls.jpg')
imTK_R = ImageTk.PhotoImage( im.resize( (img_size, img_size) ) )

image_main = tk.Label(div3, image=imTK_R)
image_main['height'] = img_size
image_main['width'] = img_size
image_main.grid(column=0, row=1, sticky=align_mode)
image_main.grid(column=1, row=1, sticky=align_mode)

#create a scalebar

value1 = tk.IntVar()
value2 = tk.IntVar()
value3 = tk.IntVar()

lbl_title1 = tk.Scale(div1, label='GaussianFilter', from_=3, to=9, orient=tk.HORIZONTAL,
         resolution=2, show=0, variable=value1, command=s_print)

lbl_title2 = tk.Scale(div1, label='GaussianFilter2', from_=3, to=9, orient=tk.HORIZONTAL,
         resolution=2, show=0, variable=value2, command=s_print)

lbl_title3 = tk.Scale(div1, label='GaussianFilter3', from_=3, to=9, orient=tk.HORIZONTAL,
         resolution=2, show=0, variable=value3, command=s_print)


lbl_title1.grid(column=0, row=1, sticky=align_mode)
lbl_title2.grid(column=0, row=3, sticky=align_mode)
lbl_title3.grid(column=0, row=2, sticky=align_mode)

define_layout(window, cols=3, rows=3)
define_layout(div1)
define_layout(div2, rows=2)
define_layout(div3, rows=2)

window.mainloop()