import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import random
import os

from ssd import SSD300
from ssd_utils import BBoxUtility

# matplotlib表示画像の拡大　大した意味なし
plt.rcParams['figure.figsize'] = (8, 8)
# 画像を見やすいようにするかみたいな　ここではしない設定
plt.rcParams['image.interpolation'] = 'nearest'

# npの行列の指数表示の禁止　大した意味なし
np.set_printoptions(suppress=True)

# tensorflowの設定クラス
config = tf.ConfigProto()
# GPUのメモリの何％を使うか否か　あまり関係ない
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

#voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
voc_classes = ['barcode', 'tag', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
# (300, 300)では自動的に画像を(300, 300)に拡大・縮小している
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('weights_SSD300.hdf5', by_name=True)
model.load_weights('./checkpoints/third_weights_barcode.09-1.46.hdf5', by_name=True)
model.compile('sgd','mse')
bbox_util = BBoxUtility(NUM_CLASSES)


for path in os.listdir('pics'):
    inputs = []
    images = []
    img_path = './pics/' + path
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    #rand = random.randint(1,10**10)
    #pre_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('pre_pics/'+str(rand)+'.jpg',pre_img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    # preprocess_input()は平均値を引いてRGB→BGRにしているだけ
    # 0-1にしているわけではないから注意
    inputs = preprocess_input(np.array(inputs))

    # verbose=1はログを出す設定
    preds = model.predict(inputs, batch_size=1, verbose=1)
    #print(preds[0][4500])
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        # おそらく予測確率
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(img / 255.)
        # 矩形の座標軸(?)
        currentAxis = plt.gca()


        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

            # barcodeのみの切り取り
            if label == 1:
                print(coords)
                # area = [xmin, ymin, xmax, ymax]
                area = [coords[0][0], coords[0][1], coords[0][0] + coords[1], coords[0][1] + coords[2]]
                print(area)
                im = cv2.imread(img_path, 1)
                dst = im[area[1]: area[3], area[0]: area[2]]
                rand = random.randint(1,1000)
                filepath = 'PASCAL_VOC/barcode/cut_images/' + path + str(rand) + '.jpg'
                cv2.imwrite(filepath, dst)
        print(path, 'の切り取り完了')

    #plt.show()
