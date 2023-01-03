import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from heapq import heappush, heappop, heapify
from collections import defaultdict
from bitarray import bitarray
import ast

img = cv2.imread('face_color.png', 0)

# img =[[62,55,55,54,49,48,47,55],  #test matrix
#     [62,57,54,52,48,47,48,53],
#     [61,60,52,49,48,47,49,54],
#     [63,61,60,60,63,65,68,65],
#     [67,67,70,74,79,85,91,92],
#     [82,95,101,106,114,115,112,117],
#     [96,111,115,119,128,128,130,127],
#     [109,121,127,133,139,141,140,133]]

img = np.array(img)
img1 = img.astype('float')


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
    print('col_mod: ', col_mod, 'row_mod: ', row_mod, 'pad_col: ', pad_col)
    if pad_col != 8 or pad_row != 8:
        image_new = cv2.copyMakeBorder(image_temp, 0, pad_col, 0, pad_row, cv2.BORDER_CONSTANT, 0)
    image_new_size = image_new.shape
    print('col: ', image_new_size[0], 'row: ', image_new_size[1])
    return image_new


def remove_padding(image_size, image_new):
    print('gray size: ', image_size)
    crop_image = image_new[:image_size[0], :image_size[1]]
    crop_image_size = crop_image.shape
    print('col: ', crop_image_size[0], 'row: ', crop_image_size[1])
    crop_image = np.array(crop_image) + 128
    return crop_image


def luminance_matrix():
    lum_matrix = [[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]]
    qlt = 1
    lum_matrix = np.array(lum_matrix) * qlt
    return lum_matrix


def quantization(image_block):
    lum_matrix = luminance_matrix()
    image_block_new = np.divide(image_block, lum_matrix)
    return image_block_new


def inv_quantization(image_block):  # invert quantization
    lum_matrix = luminance_matrix()
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


def zero_remove(arr):
    arr = np.around(arr, decimals=0)
    arr_list = arr.tolist()
    text = ','.join('%s' %id for id in arr)
    # print('zero_type: ', type(text), 'len: ', len(text), text)
    count = 0
    if len(text) > 128:
        for i in range(1,len(arr)):
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
    print('hum_text_type: ', type(text), text)
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
    print('encoded_text: ',encoded_text)
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
    print('decoded_text_rm_pad: ', decoded_text)

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
    image_size = image.shape
    image_after_pad = padding(image)
    image_after_pad_size = image_after_pad.shape
    image_new = np.zeros((image_after_pad_size[0],image_after_pad_size[1]))
    image_block = np.zeros((stride,stride))
    image_block = image_block.reshape(stride,stride)
    for i, j in itertools.product(range(0,image_after_pad_size[0],stride), range(0,image_after_pad_size[1],stride)):  # split by block
        for x, y in itertools.product(range(stride), range(stride)):  # inside block computation
            image_block[x][y] = image_after_pad[i + x][j + y]
        image_block_after_dct, dct_mtx = dct_tsf(image_block)  # DCT transform
        image_block_after_qtz = quantization(image_block_after_dct)
        print('image_block_after_qtz: ', image_block_after_qtz)
        print('image_block_after_qtz_type: ', type(image_block_after_qtz))
        image_block_after_zza, pix_loc = zigzag(image_block_after_qtz, stride)

        file_name = str(i) + '_' + str(j)
        image_block_after_hfc = huffman_coding(image_block_after_zza, file_name)

        image_block_after_zzi = zagzig(image_block_after_hfc, stride, pix_loc)
        # image_block_after_zza_size = image_block_after_zza.shape
        # print('image_block_after_zza_size: ', image_block_after_zza_size)
        # image_block_after_zzi_size = image_block_after_zzi.shape
        # print('image_block_after_zzi_size: ', image_block_after_zzi_size)
        # print('image_block_after_dct: ', image_block_after_qtz)
        # print('image_block_after_zza: ', image_block_after_zza)
        # print('image_block_after_zzi: ', image_block_after_zzi)

        image_block_after_iqtz = inv_quantization(image_block_after_zzi)
        image_block_after_idct = idct_tsf(image_block_after_iqtz, dct_mtx)  # iDCT transform
        # image_block_after_inv_dct = cv2.idct(image_block_after_dct)  # Library iDCT
        for m, n in itertools.product(range(stride), range(stride)):
            image_new[i + m][j + n] = image_block_after_idct[m][n]
    image_new = remove_padding(image_size, image_new)
    print('image_new_max: ', np.max(image_new), 'image_new_min: ', np.min(image_new))

    return image_block_after_dct, image_new


n = 8
block_after_dct, image_after_idct = image_process(img1, n)

plt.subplot(121)  # 231
plt.imshow(img1, 'gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])

# plt.subplot(222)
# plt.hist(img1.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('histogram'), loc = 'upper left')

plt.subplot(122)  # 233
plt.imshow(image_after_idct, 'gray')  # IDCT
plt.title('IDCT1')
plt.xticks([]), plt.yticks([])
print('done')
plt.show()
