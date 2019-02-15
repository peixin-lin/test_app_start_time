#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.externals import joblib
from PIL import Image
import numpy as np
import os
from sklearn import svm
import re
import time
import sys
import shutil

def get_image_datas(img_dir):
    label_list = []
    image_list = []
    image_classes = os.listdir(img_dir)
    for classes in image_classes:
        image_dir = img_dir + classes
        for image_path in os.listdir(image_dir)[:-1]:
            if image_path.endswith(".jpeg"):
                img = Image.open(image_dir+"/"+image_path)
                # 获得图像尺寸:
                w, h = img.size
                # 缩放到1/8:
                img.thumbnail((w // 8, h // 8))
                image_list.append(np.asarray(img).flatten())
                label_list.append(classes)
    return image_list, label_list

def train_model(X_train, Y_train):
    linear_svcClf = svm.LinearSVC()
    linear_svcClf.fit(X_train, Y_train)
    joblib.dump(linear_svcClf, "svm_trained.model")
    print('model trained.')

def numerical_sort(value):   
    ''' 用以将图片根据文件名排序的函数 '''
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_test_images(folder):
    image_list = []
    image_dir = os.getcwd() + '/' + folder + '/'
    file_names = sorted((fn for fn in os.listdir(image_dir) if fn.endswith(".jpeg")), key=numerical_sort)
    file_names = [image_dir + fn for fn in file_names]
    for image_path in file_names:
        if image_path.endswith(".jpeg"):
            img = Image.open(image_path)
            # 获得图像尺寸:
            w, h = img.size
            # 缩放到1/8:
            img.thumbnail((w // 8, h // 8))
            image_list.append(np.asarray(img).flatten())
    return image_list

def predict_from_model(img_list, model="svm_trained.model"):
    clf = joblib.load(model)
    predicts = clf.predict(img_list)
    return predicts

def calculate(predictions):
    ''' 用以计算启动时间的函数 '''
    start_index = None
    end_index = None
    for index, result in enumerate(predictions):
            # 以第一帧被识别为‘"activated"的前一帧作为起始帧
            if result == 'activated' and start_index is None:
                start_index = index - 1
            # 以第一帧被识别为"frontpage"的作为结束帧
            if result == 'frontpage' and end_index is None:
                end_index = index

    index_different = end_index - start_index + 1
    # 帧率以图片数量除以录制时间计算
    frame_rate = len(predictions) / 10
    print('index different: ', index_different)
    print('frame rate: ', frame_rate)
    print('start time: ', index_different / frame_rate)
    pass

def evaluate(X_train, Y_train, X_test, Y_test):
    linSVC = svm.LinearSVC()
    linSVC.fit(X_train, Y_train)
    from sklearn.metrics import accuracy_score
    Y_test_pred = linSVC.predict(X_test)
    print('Evaluate result: ', accuracy_score(Y_test, Y_test_pred))

def record(folder, pkg):
    ''' 用以自动录制启动视频并生成图片的函数，此方法省略了手指点击按钮的过程，误差可能较大 '''
    if folder in os.listdir('.'):
        shutil.rmtree(folder)
    os.mkdir(folder)
    os.system('cd '+folder)
    os.system('cd '+folder+'\n'+'adb shell screenrecord --verbose --time-limit 10 /sdcard/'+folder+'.mp4 &')
    time.sleep(1)
    os.system('adb shell am start -W '+pkg)
    print('Recorded. Wait for pulling.')
    time.sleep(10)
    os.system('cd '+folder+'\n'+'adb pull sdcard/'+folder+'.mp4')
    print('Pulled.')
    os.system('cd '+folder+'\n'+'ffmpeg -i '+folder+'.mp4 -r 30 -t 100 %d.jpeg')
    print('Converted. Finished.')
    os.system('rm '+folder+'/'+folder+'.mp4')
    
def main(args):
    operation = args[0]
    if operation == 'r':
        folder = args[1]
        pkg = args[2]
        record(folder, pkg)
    elif operation == 'c':
        folder = args[1]
        img_list = get_test_images(folder)
        predictions = predict_from_model(img_list)
        calculate(predictions)
    elif operation == 't':
        folder = args[1]
        X_train, Y_train = get_image_datas(folder)
        train_model(X_train, Y_train)
        
if __name__ == '__main__':
    main(sys.argv[1:])


# In[ ]:




