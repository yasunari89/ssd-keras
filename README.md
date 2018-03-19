SSDの実行手順は<code>SSD.ipynb</code>学習手順は<code>SSD_training.ipynb</code>をGitHub上で参照

<strong>https://github.com/rykov8/ssd_keras を基に作成</strong><br/>

## ssd_layers.pyのv2.xに伴うコード変更
<pre>
get_output_shape_for(self, input_shape)
</pre>
<pre>
compute_output_shape(self, input_shape) 
</pre>

## run.pyのv2.xに伴うコード変更
<pre>
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
</pre>
<pre>
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
model.compile('sgd','mse')
</pre>

## run.pyの画像trimming codeの追加
<pre><code>
    print(coords)
    # area = [xmin, ymin, xmax, ymax]
    area = [coords[0][0], coords[0][1], coords[0][0] + coords[1], coords[0][1] + coords[2]]
    print(area)
    im = cv2.imread(img_path, 1)
    dst = im[area[1]: area[3], area[0]: area[2]]
    rand = random.randint(1,10**10)
    filepath = 'PASCAL_VOC/barcode/cut_images/' + str(rand) + '.jpg'
    cv2.imwrite(filepath, dst)
print(path, 'の切り取り完了')
</code></pre>

## 複数枚の検出
<pre>
img_path = './pics/名前.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
</pre>
を追加

## 実行時の参考
https://qiita.com/slowsingle/items/64cc927bb29a49a7af14
