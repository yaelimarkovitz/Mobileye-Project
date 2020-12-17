from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import random
import os
import zipfile
import glob


def open_zipfile(labals_path, pictuers_path):
    set_ = zipfile.ZipFile(labals_path, 'r')
    set_.extractall(path="labels/")
    set_ = zipfile.ZipFile(pictuers_path, 'r')
    set_.extractall(path="pic/")


def rand_next_point(pic, x, y, un_tfl_points):
    numx = random.randint(0, x)
    numy = random.randint(0, y)
    while pic[numy, numx] == 19:
        numx = random.randint(0, x)
        numy = random.randint(0, y)
    un_tfl_points.append((numx, numy))


def analyze_label(label):
    # print(label)
    im = Image.open(label)
    width, height = im.size
    pic = np.asarray(im, dtype='uint8')

    tfl_points = []
    un_tfl_points = []
    i, j = 0, 0
    while i < height:
        while j < width:
            if pic[i][j] == 19:
                tfl_points.append((j, i))
                rand_next_point(pic, j, i, un_tfl_points)
                j += 200
                i += 200
            else:
                j += 1
        j = 0
        i += 1
    return tfl_points, un_tfl_points


def crop_image(path, x, y):
    im = Image.open(path)
    im = im.crop((x, y, x + 81, y + 81))
    croped_image = np.asarray(im, dtype='uint8')
    return croped_image


def load_to_bin(croped_image, set_, is_tfl):
    filedata = 'Data_dir/' + set_ + '/data.bin'
    with open(filedata, mode='ab+') as fileobj:
        np.array(croped_image, dtype=np.uint8).tofile(fileobj)
    filelabel = 'Data_dir/' + set_ + '/labels.bin'
    with open(filelabel, mode='ab+') as fileobj:
        np.array(np.array([is_tfl]), dtype=np.uint8).tofile(fileobj)


def load_city_pic(labels_path, pictures_path, city, set_):
    city_labels = glob.glob(labels_path + '/' + city + '/' + '*labelIds.png')
    for label in city_labels:
        tfl_list, un_tfl_list = analyze_label(label)
        if tfl_list:
            for tfl_point, untfl_point in zip(tfl_list, un_tfl_list):
                pic_path = pictures_path + '/' + (label.split("/")[-1]).split("gtFine")[0] + "leftImg8bit.png"
                croped_im = crop_image(pic_path, tfl_point[0], tfl_point[1])
                load_to_bin(croped_im, set_, 1)
                croped_im = crop_image(pic_path, untfl_point[0], untfl_point[1])
                load_to_bin(croped_im, set_, 0)


def build_trainset(set_):
    labels_path = './labels/gtFine/' + set_
    pictures_path = './pic/leftImg8bit/' + set_
    cities = os.listdir(labels_path)
    for city in cities:
        print(city)
        load_city_pic(labels_path, pictures_path, city, set_)


def check_dataset(file, index):
    im = crop_image('pic/leftImg8bit/train/aachen/aachen_000002_000019_leftImg8bit.png', 0, 0)
    size = bytearray(im)
    image = np.memmap('../Data_dir/train/data.bin', dtype='uint8', mode='r', offset=index * len(size), shape=(81, 81, 3))
    plt.imshow(image)
    plt.show()
    flag = np.memmap('../Data_dir/train/labels.bin', dtype='uint8', mode='r', offset=index * 1, shape=(1))
    print(flag)


def main():
    open_zipfile('C:/Users/RENT/Desktop/sets/gtFine_trainvaltest.zip',
                 'C:/Users/RENT/Desktop/sets/leftImg8bit_trainvaltest.zip')
    build_trainset('train')
    build_trainset('val')
    check_dataset('../Data_dir/train/data.bin', 190)


if __name__ == '__main__':
    main()
