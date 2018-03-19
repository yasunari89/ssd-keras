import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

"""
拡張子に注意！
このコードは'.jpg'と'.JPG'に対応
.jpegではないことに注意

コマンドライン引数
0: グレースケール[rgb,bgrではないので注意]
1: 閾値処理
2: ガウスぼかし
3: 左右反転
"""

argv = int(sys.argv[1])
path = 'barcode/data'
lists = os.listdir(path)
#print(lists)
filenames = []
for one_list in lists:
    if one_list[-4:] == '.jpg' or one_list[-4:] == '.JPG':
        filenames.append(one_list)
#print(filenames)

for filename in filenames:
    path = path + '/' + filename
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if argv == 0:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        print(path, 'をグレースケール化しています...')
    if argv == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        print('閾値: ', _)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print(path, 'を閾値処理しています...')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if argv == 2:
        img = cv2.GaussianBlur(img, (55, 55), 0)
        print(path, 'をガウスぼかししています...')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if argv == 3:
        img = cv2.flip(img, 1)
        print(path, 'を左右反転させています...')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    cv2.imwrite('barcode/create_images/' + str(np.random.randint(10000000)) + '.jpg', img)

    path = 'barcode/data'
