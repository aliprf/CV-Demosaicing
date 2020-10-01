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

def show_img(img, prefix):
    p_img = Image.fromarray(img.astype(np.uint8), 'RGB')
    p_img.save(prefix + '.png')
    # img.show()


def gradient(rgb_d, x_pixel, y_pixel, c):
    weights = [1/2, 5/8, 3/4]
    indices = [0, 1, 2]
    indices.remove(c)
    _avg = 0
    for index in indices:
        if rgb_d[x_pixel, y_pixel, index] != 0:
            weight = weights[index]
            v1 = v2 = v3 = v4 = 0
            w = rgb_d.shape[0]
            h = rgb_d.shape[1]
            if x_pixel - 2 >= 0:
                v1 = rgb_d[x_pixel - 1, y_pixel, c]
            if x_pixel + 2 < w:
                v2 = rgb_d[x_pixel + 1, y_pixel, c]
            if y_pixel - 2 >= 0:
                v3 = rgb_d[x_pixel, y_pixel - 1, c]
            if y_pixel + 2 < h:
                v4 = rgb_d[x_pixel, y_pixel + 1, c]
            _avg = weight * (rgb_d[x_pixel, y_pixel, index] - (v1 + v2 + v3 + v4) / 4)
            break
    return _avg


def bayer(rgb_d, x_pixel, y_pixel, c):
    v1 = v2 = v3 = v4 =0
    w = rgb_d.shape[0]
    h = rgb_d.shape[1]
    if x_pixel-1 >= 0:
        v1 = rgb_d[x_pixel-1, y_pixel, c]
    if x_pixel+1 < w:
        v2 = rgb_d[x_pixel+1, y_pixel, c]
    if y_pixel-1 >= 0:
        v3 = rgb_d[x_pixel, y_pixel-1, c]
    if y_pixel+1 < h:
        v4 = rgb_d[x_pixel, y_pixel+1, c]
    _avg = (v1 + v2 + v3 + v4)//4
    return _avg


def bilinear_interpolation(b_imgs, inp_imgs):
    """
    :param b_imgs: list of a 2d images {w*h}
    :param inp_imgs: list of rgb images {w*h*3}
    :return:
    """
    R = 0
    G = 1
    B = 2
    index = 0
    for img in tqdm(b_imgs):
        if len(img.shape) == 3:
            img = img[:, :, 0]
        rgb_d = np.zeros([img.shape[0], img.shape[1], 3])
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

        '''save biLinear results'''
        show_img(rgb_d, "dec_" + str(index))
        '''Demosaicing:'''
        rgb_d_imp = rgb_d.copy()
        rgb_d_base = rgb_d.copy()
        for x_pixel in range(rgb_d.shape[0]):  # '''width'''
            for y_pixel in range(rgb_d.shape[1]):  # '''height'''
                for c in range(rgb_d.shape[2]):  # '''channel'''
                    if rgb_d[x_pixel, y_pixel, c] == 0:
                        '''biLinear interpolation'''
                        b_pixel = bayer(rgb_d, x_pixel, y_pixel, c)
                        rgb_d[x_pixel, y_pixel, c] = b_pixel
                        '''improved Linear interpolation:'''
                        rgb_d_imp[x_pixel, y_pixel, c] = b_pixel + gradient(rgb_d_base, x_pixel, y_pixel, c)
        '''save biLinear results'''
        show_img(rgb_d, "base_" + str(index))
        '''save improved results'''
        show_img(rgb_d_imp, "imp_" + str(index))
        index += 1


def load_images():
    img_path = './imgs/'
    b_imgs = []
    inp_imgs = []
    for file in tqdm(os.listdir(img_path)):
        if str(file).startswith("b"):
            b_imgs.append(np.array(Image.open(img_path + file)))
            inp_imgs.append(np.array(Image.open(img_path + str(file).split('_')[1])))
    return b_imgs, inp_imgs


if __name__ == '__main__':
    '''we first load the images in the path'''

    b_imgs, inp_imgs = load_images()
    bilinear_interpolation(b_imgs, inp_imgs)
