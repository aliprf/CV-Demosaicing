import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import os.path
from tqdm import tqdm
import math


def load_images():
    img_path = './imgs/'
    b_imgs = []
    inp_imgs = []
    for file in tqdm(os.listdir(img_path)):
        if str(file).startswith("b"):
            b_imgs.append(np.array(Image.open(img_path + file)))
            inp_imgs.append(np.array(Image.open(img_path + str(file).split('_')[1])))
    return b_imgs, inp_imgs


def show_img(img, prefix):
    p_img = Image.fromarray(img.astype(np.uint8), 'RGB')
    p_img.save(prefix + '.png')
    # img.show()


def get_x_at_y(rgb_d, x_pixel, y_pixel, c):
    channels = [0, 1, 2]
    channels.remove(c)
    if rgb_d[x_pixel, y_pixel, channels[0]] == -1 and rgb_d[x_pixel, y_pixel, channels[1]] == -1:
        print('both are -1')
    for ch in channels:
        if rgb_d[x_pixel, y_pixel, ch] != -1:
            return c, ch


def calc_green_at_red_or_blue(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel):
    w = rgb_d.shape[0]
    h = rgb_d.shape[1]
    main_channel = rgb_d[:, :, pixel]
    opposite_channel = rgb_d[:, :, at_pixel]
    value = b_pixel
    if x_pixel - 2 >= 0 and x_pixel + 2 < w and y_pixel - 2 >= 0 and y_pixel + 2 < h:
        main_ch_data = 2 * np.sum([main_channel[x_pixel - 1, y_pixel], main_channel[x_pixel + 1, y_pixel],
                                   main_channel[x_pixel, y_pixel - 1], main_channel[x_pixel, y_pixel + 1]])
        opposite_ch_data = 4 * opposite_channel[x_pixel, y_pixel] - (
                opposite_channel[x_pixel - 2, y_pixel] + opposite_channel[x_pixel + 2, y_pixel] +
                opposite_channel[x_pixel, y_pixel - 2] + opposite_channel[x_pixel, y_pixel + 2])
        value = (main_ch_data + opposite_ch_data) / 9
    return value


def calc_red_blue_at_versa(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel):
    w = rgb_d.shape[0]
    h = rgb_d.shape[1]
    opposite_channel = rgb_d[:, :, at_pixel]
    value = b_pixel
    main_channel = rgb_d[:, :, pixel]

    if x_pixel - 2 >= 0 and x_pixel + 2 < w and y_pixel - 2 >= 0 and y_pixel + 2 < h:
        main_ch_data = 2 * np.sum([main_channel[x_pixel - 1, y_pixel - 1], main_channel[x_pixel + 1, y_pixel + 1],
                                   main_channel[x_pixel + 1, y_pixel - 1], main_channel[x_pixel - 1, y_pixel + 1]])

        opposite_ch_data = 6 * opposite_channel[x_pixel, y_pixel] - (3 / 2) * (opposite_channel[x_pixel - 2, y_pixel] +
                                                                               opposite_channel[x_pixel + 2, y_pixel] +
                                                                               opposite_channel[x_pixel, y_pixel - 2] +
                                                                               opposite_channel[x_pixel, y_pixel + 2])
        value = (main_ch_data + opposite_ch_data) / 9
    return value


def calc_red_blue_at_green(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel):
    w = rgb_d.shape[0]
    h = rgb_d.shape[1]
    main_channel = rgb_d[:, :, pixel]
    opposite_channel = rgb_d[:, :, at_pixel]
    value = b_pixel

    if x_pixel - 2 >= 0 and x_pixel + 2 < w and y_pixel - 2 >= 0 and y_pixel + 2 < h:
        '''calc cross diagonal'''
        cross_data = - 1 * (opposite_channel[x_pixel - 1, y_pixel - 1] + opposite_channel[x_pixel + 1, y_pixel + 1] +
                            opposite_channel[x_pixel - 1, y_pixel + 1] + opposite_channel[x_pixel + 1, y_pixel - 1])

        if main_channel[x_pixel - 1, y_pixel] != -1 and main_channel[x_pixel + 1, y_pixel] != -1:
            vert_hor_data = - 1 * (opposite_channel[x_pixel, y_pixel - 2] + opposite_channel[x_pixel, y_pixel + 2]) \
                    + 0.5 * (opposite_channel[x_pixel - 2, y_pixel] + opposite_channel[x_pixel + 2, y_pixel])
            main_ch_data = 4 * np.sum([main_channel[x_pixel-1, y_pixel], main_channel[x_pixel+1, y_pixel]])

        elif main_channel[x_pixel, y_pixel - 1] != -1 and main_channel[x_pixel, y_pixel + 1] != -1:
            vert_hor_data = - 1 * (opposite_channel[x_pixel - 2, y_pixel] + opposite_channel[x_pixel + 2, y_pixel]) \
                    + 0.5 * (opposite_channel[x_pixel, y_pixel - 2] + opposite_channel[x_pixel, y_pixel + 2])
            main_ch_data = 4 * np.sum([main_channel[x_pixel, y_pixel-1], main_channel[x_pixel, y_pixel+1]])
        else:
            assert 'error'

        value = (5 * opposite_channel[x_pixel, y_pixel] + vert_hor_data + main_ch_data + cross_data) / 11
    return value


def gradient(b_pixel, rgb_d, x_pixel, y_pixel, c):
    R = 0
    G = 1
    B = 2
    pixel, at_pixel = get_x_at_y(rgb_d, x_pixel, y_pixel, c)

    if pixel == G and (at_pixel == R or at_pixel == B):
        _val = calc_green_at_red_or_blue(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel)
    elif (pixel == R and at_pixel == B) or (pixel == B and at_pixel == R):
        _val = calc_red_blue_at_versa(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel)
    elif (pixel == R or pixel == B) and at_pixel == G:
        _val = calc_red_blue_at_green(b_pixel, rgb_d, x_pixel, y_pixel, pixel, at_pixel)
    else:
        assert "error"
    return _val


def bayer(rgb_d, x_pixel, y_pixel, c):
    R = 0
    G = 1
    B = 2

    w = rgb_d.shape[0]
    h = rgb_d.shape[1]
    values = []
    '''figure out what channel we wanna find'''
    if c == G:
        if x_pixel - 1 >= 0 and rgb_d[x_pixel - 1, y_pixel, c] != -1:
            values.append(rgb_d[x_pixel - 1, y_pixel, c])
        if x_pixel + 1 < w and rgb_d[x_pixel + 1, y_pixel, c] != -1:
            values.append(rgb_d[x_pixel + 1, y_pixel, c])
        if y_pixel - 1 >= 0 and rgb_d[x_pixel, y_pixel - 1, c] != -1:
            values.append(rgb_d[x_pixel, y_pixel - 1, c])
        if y_pixel + 1 < h and rgb_d[x_pixel, y_pixel + 1, c] != -1:
            values.append(rgb_d[x_pixel, y_pixel + 1, c])
        _avg = sum(values) // len(values)

    elif c == B or c == R:  # we need cross diagonal
        '''cross diagonal:'''
        if x_pixel - 1 >= 0 and y_pixel - 1 >= 0 and rgb_d[x_pixel - 1, y_pixel - 1, c] != -1:  # [-1,-1]
            values.append(rgb_d[x_pixel - 1, y_pixel - 1, c])
        if x_pixel + 1 < w and y_pixel + 1 < h and rgb_d[x_pixel + 1, y_pixel + 1, c] != -1:  # [+1,+1]
            values.append(rgb_d[x_pixel + 1, y_pixel + 1, c])
        if x_pixel + 1 < w and y_pixel - 1 >= 0 and rgb_d[x_pixel + 1, y_pixel - 1, c] != -1:  # [+1,-1]
            values.append(rgb_d[x_pixel + 1, y_pixel - 1, c])
        if x_pixel - 1 >= 0 and y_pixel + 1 < h and rgb_d[x_pixel - 1, y_pixel + 1, c] != -1:  # [-1,+1]
            values.append(rgb_d[x_pixel - 1, y_pixel + 1, c])
        ''''''
        if x_pixel - 1 >= 0 and rgb_d[x_pixel - 1, y_pixel, c] != -1:
            values.append(rgb_d[x_pixel - 1, y_pixel, c])
        if x_pixel + 1 < w and rgb_d[x_pixel + 1, y_pixel, c] != -1:
            values.append(rgb_d[x_pixel + 1, y_pixel, c])
        if y_pixel - 1 >= 0 and rgb_d[x_pixel, y_pixel - 1, c] != -1:
            values.append(rgb_d[x_pixel, y_pixel - 1, c])
        if y_pixel + 1 < h and rgb_d[x_pixel, y_pixel + 1, c] != -1:
            values.append(rgb_d[x_pixel, y_pixel + 1, c])
    _avg = sum(values) // len(values)
    # if math.isnan(_avg):
    return _avg


def demosaicing(b_imgs, inp_imgs):
    """
    :param b_imgs: list of a 2d images {w*h}
    :param inp_imgs: list of rgb images {w*h*3}
    :return:input_img_arr, bl_img_arr, hq_img_arr
    """
    input_img_arr = []
    bl_img_arr = []
    hq_img_arr = []

    R = 0
    G = 1
    B = 2
    index = 0
    for img in tqdm(b_imgs):
        # if len(img.shape) == 3:
        #     img = img[:, :, 0]
        '''Demosaicing:'''
        rgb_d = np.zeros([img.shape[0], img.shape[1], 3]) - 1
        for x_pixel in range(0, rgb_d.shape[0], 2):
            for y_pixel in range(0, rgb_d.shape[1], 2):
                pixel_value = img[x_pixel, y_pixel]
                rgb_d[x_pixel, y_pixel, B] = pixel_value
                if x_pixel + 1 < rgb_d.shape[0]:
                    rgb_d[x_pixel + 1, y_pixel, G] = img[x_pixel + 1, y_pixel]
                if y_pixel + 1 < rgb_d.shape[1]:
                    rgb_d[x_pixel, y_pixel + 1, G] = img[x_pixel, y_pixel + 1]
                if x_pixel + 1 < rgb_d.shape[0] and y_pixel + 1 < rgb_d.shape[1]:
                    rgb_d[x_pixel + 1, y_pixel + 1, R] = img[x_pixel + 1, y_pixel + 1]

        rgb_d_imp = rgb_d.copy()
        rgb_d_base = rgb_d.copy()
        '''save biLinear results'''
        show_img(rgb_d, "0_dem_" + str(index))

        for x_pixel in range(rgb_d.shape[0]):  # '''width'''
            for y_pixel in range(rgb_d.shape[1]):  # '''height'''
                for c in range(rgb_d.shape[2]):  # '''channel'''
                    if rgb_d[x_pixel, y_pixel, c] == -1:
                        '''biLinear interpolation:'''
                        b_pixel = bayer(rgb_d, x_pixel, y_pixel, c)
                        rgb_d[x_pixel, y_pixel, c] = b_pixel
                        '''improved Linear interpolation'''
                        rgb_d_imp[x_pixel, y_pixel, c] = gradient(b_pixel, rgb_d_base, x_pixel, y_pixel, c)
        '''save biLinear/hq results'''
        show_img(rgb_d, "1_bay_" + str(index))
        show_img(rgb_d_imp, "2_imp_" + str(index))

        '''saving imgs in mem'''
        input_img_arr.append(inp_imgs[index])
        bl_img_arr.append(rgb_d)
        hq_img_arr.append(rgb_d_imp)
        index += 1
    return input_img_arr, bl_img_arr, hq_img_arr


def cal_loss(i_img_arr, o_img_arr):
    R_2 = 255.0 * 255.0
    mse_arr = []
    psnr_arr = []
    for i in range(len(i_img_arr)):
        '''MSE'''
        mse = np.square(i_img_arr[i].flatten() - o_img_arr[i].flatten()).mean(axis=-1)
        mse_arr.append(mse)
        '''PSNR'''
        psnr = 10 * np.log10(R_2/mse)
        psnr_arr.append(psnr)
    return mse_arr, psnr_arr


if __name__ == '__main__':
    '''we first load the images in the path'''

    b_imgs, inp_imgs = load_images()
    input_img_arr, bl_img_arr, hq_img_arr = demosaicing(b_imgs, inp_imgs)
    mse_bl, psnr_bl = cal_loss(input_img_arr, bl_img_arr)
    mse_hg, psnr_hq = cal_loss(input_img_arr, hq_img_arr)
    print("BL->MSE: " + str(mse_bl) + "AVG: " + str(np.mean(mse_bl)) + "     BL->PSNR: " + str(psnr_bl) + "     AVG: " + str(np.mean(psnr_bl)))
    print("HQ->MSE: " + str(mse_hg) + "AVG: " + str(np.mean(mse_hg)) + "     HQ->PSNR: " + str(psnr_hq) + "     AVG: " + str(np.mean(psnr_hq)))
