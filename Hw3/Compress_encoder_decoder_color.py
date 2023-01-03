import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from heapq import heappush, heappop, heapify
from collections import defaultdict
from bitarray import bitarray
import ast
import copy

img = cv2.imread('face_color.png')  # ,0: gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print('read image')
# img =[[62,55,55,54,49,48,47,55],  #test matrix
#     [62,57,54,52,48,47,48,53],
#     [61,60,52,49,48,47,49,54],
#     [63,61,60,60,63,65,68,65],
#     [67,67,70,74,79,85,91,92],
#     [82,95,101,106,114,115,112,117],
#     [96,111,115,119,128,128,130,127],
#     [109,121,127,133,139,141,140,133]]

img = np.array(img)
img1 = img.astype(np.float64)
# img1 = img


def color_cvt_matrix():
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    # transform_matrix = np.array([[0.299, 0.587, 0.114],
    #                              [-0.169, -0.331, 0.500],
    #                              [0.500, -0.419, -0.081]])
    # shift_matrix = np.array([0, 128, 128])
    # transform_matrix = np.array([[0.183, 0.614, 0.062],
    #                              [-0.101, -0.339, 0.439],
    #                              [0.439, -0.399, -0.040]])
    # shift_matrix = np.array([0, 128, 128])
    return transform_matrix, shift_matrix


def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        print("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float64)
    rgb_image_size = rgb_image.shape
    # RGB to YCbCr matrix
    ycbcr2rgb_matrix, shift_matrix = color_cvt_matrix()
    ycbcr_image = np.zeros(rgb_image.shape)

    # run all pixel with RGB and convert to YCbCr
    for i, j in itertools.product(range(rgb_image_size[0]), range(rgb_image_size[1])):
        ycbcr_image[i, j, :] = np.dot(ycbcr2rgb_matrix, rgb_image[i, j, :]) + shift_matrix
        # ycbcr_image[i, j, :] = shift_matrix - np.dot(ycbcr2rgb_matrix, rgb_image[i, j, :])
    # ycbcr_image = ycbcr_image / 255
    return ycbcr_image


def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    # ycbcr_image = ycbcr_image * 255
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        print("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float64)
    ycbcr_image_size = ycbcr_image.shape
    rgb2ycbcr_matrix, shift_matrix = color_cvt_matrix()

    rgb2ycbcr_matrix_inv = np.linalg.inv(rgb2ycbcr_matrix)
    rgb_image = np.zeros(ycbcr_image_size)

    for i,j in itertools.product(range(ycbcr_image_size[0]),range(ycbcr_image_size[1])):
        rgb_image[i, j, :] = np.dot(rgb2ycbcr_matrix_inv, ycbcr_image[i, j, :]) - np.dot(rgb2ycbcr_matrix_inv, shift_matrix)
        # rgb_image[i, j, :] = np.dot(rgb2ycbcr_matrix_inv, ycbcr_image[i, j, :]) + np.dot(rgb2ycbcr_matrix_inv,
        #                                                                                  shift_matrix)
    return rgb_image.astype(np.uint8)


def subsampling_420(image):
    image_after_sampling = image.copy()  # 4:2:0
    image_after_sampling[1::2, :] = image_after_sampling[::2, :]  # 0
    # Vertically, every 2nd element equals to element above itself.
    image_after_sampling[:, 1::2] = image_after_sampling[:, ::2]  # 2
    # Horizontally, every 2nd element equals to the element on its left side.
    # print(image_after_sampling)
    return image_after_sampling


def subsampling_422(image):
    image_after_sampling = image.copy()  # 4:2:2
    # Horizontally, every 2nd element equals to the element on its left side.
    image_after_sampling[:, 1::2] = image_after_sampling[:, ::2]  # 4:2:2
    # print(image_after_sampling)
    return image_after_sampling


def dct_tsf(image_block):
    dct_block = np.zeros(image_block.shape)
    image_block_new = np.zeros(image_block.shape)
    m, n = image_block.shape
    dct_block[0, :] = 1 * np.sqrt(1 / n)
    for u in range(1, m):
        for x in range(n):
            dct_block[u, x] = np.cos(np.pi * u * (x + 0.5) / n) * np.sqrt(2 / n)

    image_block_new = np.dot(dct_block, image_block)
    image_block_new = np.dot(image_block_new, np.transpose(dct_block))
    return image_block_new, dct_block


def idct_tsf(image_block, dct_block):
    image_block_new = np.dot(np.transpose(dct_block), image_block)
    image_block_new = np.dot(image_block_new, dct_block)
    return image_block_new


def padding(image):
    image_temp = np.array(image) - 128
    image_size = image_temp.shape
    # print('gray size: ', image_size)
    # print('col: ', image_size[0], 'row: ', image_size[1])
    col_mod = image_size[0] % 8
    row_mod = image_size[1] % 8
    pad_col = 8 - col_mod
    pad_row = 8 - row_mod
    image_new = np.array(image)
    # print('col_mod: ', col_mod, 'row_mod: ', row_mod, 'pad_col: ', pad_col)
    if pad_col != 8 or pad_row != 8:
        image_new = cv2.copyMakeBorder(image_temp, 0, pad_col, 0, pad_row, cv2.BORDER_CONSTANT, 0)
    image_new_size = image_new.shape
    # print('col: ', image_new_size[0], 'row: ', image_new_size[1])
    return image_new


def remove_padding(image_size, image_new):
    print('gray size: ', image_size)
    crop_image = image_new[:image_size[0], :image_size[1]]
    crop_image_size = crop_image.shape
    print('col: ', crop_image_size[0], 'row: ', crop_image_size[1])
    crop_image = np.array(crop_image)  # + 128
    return crop_image


def luminance_matrix(size):
    lum_matrix = [[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]]
    lum_matrix = np.array(lum_matrix)
    lum_matrix_size = lum_matrix.shape
    lum_matrix_update = lum_matrix[:size,:size]
    lum_matrix_update_size = lum_matrix_update.shape
    # print('lum_matrix_update_size: ', lum_matrix_update_size, 'type lum: ', type(lum_matrix_update))
    qlt = 1
    lum_matrix_update = lum_matrix_update * qlt
    return lum_matrix_update


def quantization(image_block):
    image_block_size = image_block.shape
    lum_matrix = luminance_matrix(image_block_size[0])
    image_block_new = np.divide(image_block, lum_matrix)
    image_block_new_size = image_block_new.shape
    # print('image_block_new_size: ', image_block_new_size, 'type image_block_new: ', type(image_block_new))
    return image_block_new


def inv_quantization(image_block):  # invert quantization
    image_block_size = image_block.shape
    lum_matrix = luminance_matrix(image_block_size[0])
    image_block_new = np.multiply(image_block, lum_matrix)
    return image_block_new


def zigzag(block, size):
    n = size
    i, j = 0, 0
    zz_block = []
    zz_loc = []
    while j != (n - 1):
        # print(i, j)
        zz_block.append(block[j][i])
        zz_loc.append([j, i])
        if i == 0 and (j & 1):  # i=0, j=1 -> 1
            j += 1
            continue
        if j == 0 and (i & 1) == 0:  # j=0, i!=1 -> 1
            i += 1
            continue
        if (i ^ j) & 1:  # i!=j -> 1
            i -= 1
            j += 1
            continue
        if (i ^ j) & 1 == 0:  # [(i!=j -> 1) = 0] -> 1
            i += 1
            j -= 1
            continue
    while i != (n - 1) or j != (n - 1):
        # print(i, j)
        zz_block.append(block[j][i])
        zz_loc.append([j, i])
        if i == (n - 1) and (j & 1):
            j += 1
            continue
        if j == (n - 1) and (i & 1) == 0:
            i += 1
            continue
        if (i ^ j) & 1 == 0:
            i += 1
            j -= 1
            continue
        if (i ^ j) & 1:
            i -= 1
            j += 1
            continue
    # print(i, j)
    zz_block.append(block[j][i])
    zz_loc.append([j, i])
    return zz_block, zz_loc


def zagzig(block, size, zz_loc):
    loc_count = 0
    zz_block = np.zeros((size * size))
    zz_block = zz_block.reshape(size, size)
    block = np.array(block)
    block = block.reshape(size, size)
    for i, j in itertools.product(range(size), range(size)):
        zz_block[zz_loc[loc_count][0]][zz_loc[loc_count][1]] = block[i][j]
        loc_count += 1
    # print(i, j)
    return zz_block


def predictive_code(arr):
    arr = np.around(arr, decimals=0)
    arr_list = arr.tolist()
    text = arr_list
    # text = ','.join('%s' %id for id in arr)
    print('zero_type: ', type(text), 'len: ', len(text), text)
    count = 0
    # if len(text) > 128:
    for i in range(1,len(arr)):
        # if len(text) > 120:
        if arr[len(arr)-i] == 0:
            count += 1
        else:
            break
    # print(count)
    arr_zero = [count, 0]

    arr_new = arr_list[:(len(arr_list)-count)] + arr_zero
    # print('zero_type: ', type(arr_new), 'len: ', len(arr_new), 'arr: ', arr_new)
    for j in range(len(arr_new)):
        if arr_new[j] == 0:
            arr_new[j] = 0
        else:
            continue
    # print('zero_after_replace: ', type(arr_new), 'len: ', len(arr_new), 'arr: ', arr_new)
    text = ','.join('%s' %id for id in arr_new)
    # print('after_zero_type: ', type(text), 'len: ', len(text), 'arr: ', text)
    return text


def zero_remove(arr):
    arr = np.around(arr, decimals=0)
    arr_list = arr.tolist()
    text = ','.join('%s' %id for id in arr)
    # print('zero_type: ', type(text), 'len: ', len(text), text)
    count = 0
    if len(text) > 128:
        for i in range(1,len(arr)):
            if len(text) > 120:
                if arr[len(arr)-i] == 0:
                    count += 1
                else:
                    break
    # print(count)
    arr_zero = [count, 0]

    arr_new = arr_list[:(len(arr_list)-count)] + arr_zero
    # print('zero_type: ', type(arr_new), 'len: ', len(arr_new), 'arr: ', arr_new)
    for j in range(len(arr_new)):
        if arr_new[j] == 0:
            arr_new[j] = 0
        else:
            continue
    # print('zero_after_replace: ', type(arr_new), 'len: ', len(arr_new), 'arr: ', arr_new)
    text = ','.join('%s' %id for id in arr_new)
    # print('after_zero_type: ', type(text), 'len: ', len(text), 'arr: ', text)
    return text


def inv_zero_add(arr):
    list2 = arr
    zero_add = [0] * int(list2[-2])
    # print('list2: ', type(int(list2[-2])))
    # print('list2: ', type(list2), 'len: ', len(list2), 'arr: ', list2)
    del list2[-2:]
    # print('zero_add: ', type(zero_add), 'len: ', len(zero_add), 'arr: ', zero_add)
    # print('list2: ', type(list2), 'len: ', len(list2), 'arr: ', list2)
    list2.extend(zero_add)
    # print('list2_after_add_zero: ', type(list2), 'len: ', len(list2), 'recovered: ', list2)
    return list2


def huffman_coding(arr,file_num):
    # print('hum_type: ', type(arr), arr)
    text = zero_remove(arr)
    # text = arr
    # text = ','.join('%s' %id for id in arr)
    # print('hum_text_type: ', type(text), text)
    file_num = str(file_num)
    freq_lib = defaultdict(int)  # generate a default library
    for ch in text:  # count each letter and record into the frequency library
        freq_lib[ch] += 1
    heap = [[fq, [sym, ""]] for sym, fq in freq_lib.items()]  # '' is for entering the huffman code later
    # print(heap)
    heapify(heap)  # transform the list into a heap tree structure
    # print(heap)
    # print(freq_lib)

    while len(heap) > 1:
        right = heappop(heap)  # heappop - Pop and return the smallest item from the heap
        # print('right = ', right)
        left = heappop(heap)
        # print('left = ', left)

        for pair in right[1:]:
            pair[1] = '0' + pair[1]  # add zero to all the right note
        for pair in left[1:]:
            pair[1] = '1' + pair[1]  # add one to all the left note
        heappush(heap, [right[0] + left[0]] + right[1:] + left[1:])
        # add values onto the heap. Eg. h = []; heappush(h, (5, 'write code')) --> h = [(5, 'write code')]

    huffman_list = right[1:] + left[1:]
    # print(huffman_list)
    huffman_dict = {a[0]: bitarray(str(a[1])) for a in huffman_list}
    # print(huffman_dict)
    encoded_text = bitarray()
    encoded_text.encode(huffman_dict, text)
    # print('encoded_text: ',encoded_text)
    pad = 8 - (len(encoded_text) % 8)
    with open('./compress/compressed_file'+file_num+'.bin', 'wb') as w:
        encoded_text.tofile(w)

    # decode
    decoded_text = bitarray()
    with open('./compress/compressed_file'+file_num+'.bin', 'rb') as r:
        decoded_text.fromfile(r)
    # print('decoded_text: ', decoded_text, 'padding: ', pad)
    if len(decoded_text) != len(encoded_text):
        decoded_text = decoded_text[:-pad]  # remove padding
    decoded_text = decoded_text.decode(huffman_dict)
    decoded_text = ''.join(decoded_text)
    # print('decoded_text_rm_pad: ', decoded_text)

    with open('./uncompress/uncompress'+file_num+'.bin', 'w') as w:
        w.write(text)
        list1 = ast.literal_eval(decoded_text)
        # print(list1)
        # print('type: ', type(list1))
        list2 = list(list1)
        # print(list2)
        list3 = inv_zero_add(list2)
    #     print('type: ', type(list2))
    # print('before_len: ', len(text), 'type: ', type(text), 'before_text: ', text)
    # print('after_len: ', len(decoded_text), 'type: ', type(decoded_text) , 'after_text: ', decoded_text)
    return list3


def image_process(image, stride):
    image_size = image.shape  # padding
    image_after_pad = padding(image)  #
    image_after_pad_size = image_after_pad.shape
    image_new = np.zeros((image_after_pad_size[0],image_after_pad_size[1]))
    image_block = np.zeros((stride,stride))
    image_block = image_block.reshape(stride,stride)
    for i, j in itertools.product(range(0,image_after_pad_size[0],stride), range(0,image_after_pad_size[1],stride)):  # split by block
        for x, y in itertools.product(range(stride), range(stride)):  # inside block computation
            image_block[x][y] = image_after_pad[i + x][j + y]
        image_block_after_dct, dct_mtx = dct_tsf(image_block)  # DCT transform
        # image_block_after_dct = cv2.dct(image_block)  # Library DCT
        image_block_after_qtz = quantization(image_block_after_dct)  # Quantization
        # print('image_block_after_qtz: ', image_block_after_qtz)
        # print('image_block_after_qtz_type: ', type(image_block_after_qtz))
        image_block_after_zza, pix_loc = zigzag(image_block_after_qtz, stride)  # Zigzag
        file_name = str(i) + '_' + str(j)
        image_block_after_hfc = huffman_coding(image_block_after_zza, file_name)  # Huffman coding
        image_block_after_zzi = zagzig(image_block_after_hfc, stride, pix_loc)  # Zagzig

        # image_block_after_zza_size = image_block_after_zza.shape
        # print('image_block_after_zza_size: ', image_block_after_zza_size)
        # image_block_after_zzi_size = image_block_after_zzi.shape
        # print('image_block_after_zzi_size: ', image_block_after_zzi_size)
        # print('image_block_after_dct: ', image_block_after_qtz)
        # print('image_block_after_zza: ', image_block_after_zza)
        # print('image_block_after_zzi: ', image_block_after_zzi)

        image_block_after_iqtz = inv_quantization(image_block_after_zzi)  # inverse Quantization
        image_block_after_idct = idct_tsf(image_block_after_iqtz, dct_mtx)  # iDCT transform
        # image_block_after_idct = cv2.dct(image_block_after_dct)  # Library iDCT

        for m, n in itertools.product(range(stride), range(stride)):
            image_new[i + m][j + n] = image_block_after_idct[m][n]
    image_new = remove_padding(image_size, image_new)
    print('image_new_max: ', np.max(image_new), 'image_new_min: ', np.min(image_new))
    return image_new


n = 8
# image_after_idct = img
# image_after_idct = np.array(image_after_idct)
# print('type: ', image_after_idct.shape)

# block_after_dct, image_after_idct = image_process(img1, n)

img1_size = img1.shape
print('img1_size: ', img1_size)
image_after_idct = np.empty(img1_size)
image_cvt2ycc_420 = np.empty(img1_size)
image_cvt2ycc_422 = np.empty(img1_size)

image_cvt2ycc = rgb2ycbcr(img1)  # cvt to ycc
# image_after_idct = image_process(img1, n)

image_cvt2ycc_420[:,:,0] = subsampling_420(image_cvt2ycc[:,:,0])
image_cvt2ycc_420[:,:,1] = subsampling_420(image_cvt2ycc[:,:,1])
image_cvt2ycc_420[:,:,2] = subsampling_420(image_cvt2ycc[:,:,2])

image_cvt2ycc_422[:,:,0] = subsampling_422(image_cvt2ycc[:,:,0])  # [:,:,1]
image_cvt2ycc_422[:,:,1] = subsampling_422(image_cvt2ycc[:,:,1])  # [:,:,1]
image_cvt2ycc_422[:,:,2] = subsampling_422(image_cvt2ycc[:,:,2])  # [:,:,2]

image_after_idct[:,:,0] = image_process(image_cvt2ycc[:,:,0], n)
image_after_idct[:,:,1] = image_process(image_cvt2ycc[:,:,1], n)
image_after_idct[:,:,2] = image_process(image_cvt2ycc[:,:,2], n)

image_cvt2ycc_420_cvt2rgb = ycbcr2rgb(image_cvt2ycc_420)
image_cvt2ycc_422_cvt2rgb = ycbcr2rgb(image_cvt2ycc_422)
image_cvt2rgb = ycbcr2rgb(image_after_idct)

print('image_cvt2r_max: ', np.max(image_after_idct[:,:,0]), 'image_cvt2rgb_min: ', np.min(image_after_idct[:,:,0]))
print('image_cvt2g_max: ', np.max(image_after_idct[:,:,1]), 'image_cvt2rgb_min: ', np.min(image_after_idct[:,:,1]))
print('image_cvt2b_max: ', np.max(image_after_idct[:,:,2]), 'image_cvt2rgb_min: ', np.min(image_after_idct[:,:,2]))

# image_after_idct = (image_after_idct - np.min(image_after_idct)) / (np.max(image_after_idct) - np.min(image_after_idct))

# image_after_idct[:,:,0] = (image_after_idct[:,:,0] - np.min(image_after_idct[:,:,0])) / (np.max(image_after_idct[:,:,0]) - np.min(image_after_idct[:,:,0]))
# image_after_idct[:,:,1] = (image_after_idct[:,:,1] - np.min(image_after_idct[:,:,1])) / (np.max(image_after_idct[:,:,1]) - np.min(image_after_idct[:,:,1]))
# image_after_idct[:,:,2] = (image_after_idct[:,:,2] - np.min(image_after_idct[:,:,2])) / (np.max(image_after_idct[:,:,2]) - np.min(image_after_idct[:,:,2]))

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(121)  # 231
plt.imshow(img, 'gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])

plt.subplot(122)  # 231
plt.imshow(image_after_idct[:,:,0], 'gray')
plt.title('uncompress image after compressed')
plt.xticks([]), plt.yticks([])

# plt.subplot(122)  # 233
# plt.imshow(image_after_idct)  # IDCT
# plt.title('image_after_idct')
# plt.xticks([]), plt.yticks([])

# plt.subplot(131)  # 233
# plt.imshow(image_cvt2rgb/255)  # IDCT
# plt.title('444')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(132)  # 233
# plt.imshow(image_cvt2ycc_420_cvt2rgb/255)  # IDCT
# plt.title('420')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(133)  # 233
# plt.imshow(image_cvt2ycc_422_cvt2rgb/255)  # IDCT
# plt.title('422')
# plt.xticks([]), plt.yticks([])

# plt.subplot(236)  # 233
# plt.imshow(image_cvt2ycc_422_cvt2rgb/255)  # IDCT
# plt.title('image_cvt2ycc_422')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(234)  # 233
# plt.imshow(image_cvt2ycc[:,:,0], 'gray')  # IDCT
# plt.title('image_cvt2ycc[:,:,0]')
# plt.xticks([]), plt.yticks([])
# plt.subplot(235)  # 233
# plt.imshow(image_cvt2ycc[:,:,1])  # IDCT
# plt.title('image_cvt2ycc_cb')
# plt.xticks([]), plt.yticks([])
# plt.subplot(236)  # 233
# plt.imshow(image_cvt2ycc[:,:,2])  # IDCT
# plt.title('image_cvt2ycc_cr')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222)
# plt.hist(img1.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('histogram'), loc = 'upper left')
print('done')
plt.show()
