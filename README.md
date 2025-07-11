# **Proyek Klasifikasi Gambar: Sampah Plastik, Kertas, dan Kantong Sampah**
Dataset yang digunakan memiliki 3 kelas jenis tas yaitu, **Garbage Bag Images**, **Paper Bag Images**, **Plastic Bag Images**. Masing-masing kelas berisi 5000 data gambar.
<br>
Sumber Dataset: [Plastic - Paper - Garbage Bag Images](https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images)

Project ini menggunakan arsitektur CNN untuk membuat model klasifikasi gambar tas. Hasil model disimpan menjadi 3 format yaitu TFLite, SavedModel, dan TFJS.

## **1. Instalasi & Impor Modul**
Unduh modul yang dibutuhkan untuk menjalankan project ini menggunakan perintah berikut:
```
!pip install -r requirements.txt

```
```python
!pip install kaggle
!pip install split-folders
!pip install tensorflowjs

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import splitfolders
import shutil
import subprocess
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

## **2. Persiapan Data**
* Dataset diunduh dari Kaggle menggunakan API Kaggle. Dataset kemudian diekstrak ke direktori yang ditentukan di Google Drive.
* Dataset dibagi menjadi set data latih (80%) dan validasi (20%) menggunakan pustaka `splitfolders`
* Augmentasi gambar diterapkan pada set data latih untuk meningkatkan variasi data dan mencegah overfitting. Augmentasi yang digunakan meliputi rotasi, pergeseran, shear, zoom, dan flip horizontal.

## **3. Pengembangan Model**
Model yang digunakan adalah Convolutional Neural Network (CNN) sekuensial dengan arsitektur sebagai berikut:
* Lapisan Konvolusi 1: 32 filter, ukuran kernel 3x3, fungsi aktivasi ReLU
* Lapisan Max Pooling 1: ukuran pool 2x2
* Lapisan Konvolusi 2: 64 filter, ukuran kernel 3x3, fungsi aktivasi ReLU
* Lapisan Max Pooling 2: ukuran pool 2x2
* Lapisan Konvolusi 3: 128 filter, ukuran kernel 3x3, fungsi aktivasi ReLU
* Lapisan Max Pooling 3: ukuran pool 2x2
* Lapisan Konvolusi 4: 512 filter, ukuran kernel 3x3, fungsi aktivasi ReLU
* Lapisan Max Pooling 4: ukuran pool 2x2
* Lapisan Flatten: Mengubah matriks fitur menjadi vektor
* Lapisan Dropout: 0.5 untuk regularisasi
* Lapisan Dense: 512 neuron, fungsi aktivasi ReLU
* Lapisan Dense Output: 3 neuron (sesuai jumlah kelas), fungsi aktivasi softmax
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

## **4. Training Model**
Model dilatih selama 20 epoch dengan learning rate 1e-4. Callback `ModelCheckpoint` digunakan untuk menyimpan model terbaik, dan EarlyStopping untuk menghentikan pelatihan jika tidak ada peningkatan setelah 5 epoch.

## **5. Evaluasi Model**
Akurasi dan loss dari model divisualisasikan untuk melihat performa selama pelatihan. Laporan klasifikasi menunjukkan presisi, recall, dan F1-score untuk setiap kelas.
[Classification Report](evaluation/image.png)

## **6. Menyimpan Model**
Model yang telah ditrainin dan dievaluasi kemudian disimpan dalam 3 format, yaitu:
* Saved Model
* TFLite
* TFJS
