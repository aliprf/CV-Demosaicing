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


def avg(rgb_d, x_pixel, y_pixel, c):
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
    for img in b_imgs:
        rgb_d = np.zeros([img.shape[0], img.shape[1], 3])
        # tmp_img = np.zeros([img.shape[0], img.shape[1], 3])
        # for y_pixel in range(rgb_d.shape[1]):  # '''height'''
        #     for x_pixel in range(rgb_d.shape[0]):  # '''width'''
        #         tmp_img[x_pixel, y_pixel, 0] = img[x_pixel, y_pixel]
        # fft_p = tmp_img.astype(np.uint8)
        # p_img = Image.fromarray(fft_p)
        # p_img.save('my' + str(index + 100) + '.jpg')
        # return 9

        for y_pixel in range(0, rgb_d.shape[1], 2):
            for x_pixel in range(0, rgb_d.shape[0], 2):
                rgb_d[x_pixel, y_pixel, B] = img[x_pixel, y_pixel]
                # tmp_img[x_pixel, y_pixel, R] = img[x_pixel, y_pixel]
                if x_pixel + 1 < rgb_d.shape[0]:
                    rgb_d[x_pixel, y_pixel, G] = img[x_pixel + 1, y_pixel]
                if y_pixel + 1 < rgb_d.shape[1]:
                    rgb_d[x_pixel, y_pixel, G] = img[x_pixel, y_pixel + 1]
                if x_pixel + 1 < rgb_d.shape[0] and y_pixel + 1 < rgb_d.shape[1]:
                    rgb_d[x_pixel, y_pixel, R] = img[x_pixel + 1, y_pixel + 1]
        # p_img = Image.fromarray(rgb_d.astype(np.uint8), 'RGB')
        # p_img.save('my' + str(index+100) + '.png')
        # img.show()
        '''create correct img'''
        for y_pixel in range(rgb_d.shape[1]):  # '''height'''
            for x_pixel in range(rgb_d.shape[0]):  # '''width'''
                for c in range(rgb_d.shape[2]):  # '''channel'''
                    if rgb_d[x_pixel, y_pixel, c] == 0:
                        rgb_d[x_pixel, y_pixel, c] = avg(rgb_d, x_pixel, y_pixel, c)

        p_img = Image.fromarray(rgb_d.astype(np.uint8), 'RGB')
        p_img.save('my' + str(index) + '.png')
        # img.show()
        index += 1
    return 0


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
