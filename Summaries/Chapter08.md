# 케라스 창시자에게 배우는 딥러닝
## 8장, *컴퓨터 비전을 위한 딥러닝*

**합성곱 신경망**(convolutional neural network)은 2011~2015년 사이의 초기 딥러닝의 부흥을 이끈 종류의 딥러닝 모델이다. 이 장에서 소개할 합성곱 신경망은 컨브넷(convnet)이라고도 부른다.

## 8.1 합성곱 신경망 소개

MNIST 숫자 이미지 분류에 컨브넷을 사용해 보겠다. 이 예제는 2장에서 밀집 연결 신경망(densely connected network)으로 해결했던 문제이다.

먼저 함수형 API를 통해 모델을 만들어 보고 이후에 이론적 배경 및 정의 등에 대해 알아보자.

**코드 8-1. 간단한 컨브넷 만들기**
```
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

컨브넷이 배치 차원을 제외하고 (image_height, image_width, image_channels) 크기의 입력 텐서를 사용한다는 점이 중요하다. 이 예제에서는 MNIST 이미지 포맷인 (28, 28, 1) 크기의 입력을 처리하도록 컨브넷을 설정해야 한다.

이제 이 컨브넷의 구조를 출력해 보자.

**코드 8-2. 모델의 summary() 메소드 출력**
```
>>> model.summary()
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2  (None, 13, 13, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 5, 5, 64)          0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 128)         73856     
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 10)                11530     
                                                                 
=================================================================
Total params: 104202 (407.04 KB)
Trainable params: 104202 (407.04 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

`Conv2D`, `MaxPooling2D` 층의 출력은 (height, width, channels) 크기의 랭크-3 텐서이다. 높이와 너비 차원은 모델이 깊어질수록 작아지는 경향이 있다. 채널의 수는 `Conv2D` 층에 전달된 첫 번째 매개변수인 `filters`에 의해 32개, 64개 또는 128개로 조절된다.

마지막 `Conv2D` 층은 128개의 채널을 가진 3X3 크기의 특성 맵(feature map)이다. 다음 단계는 이 출력을 밀집 연결 분류기로 주입하는 것인데, 1D 벡터를 처리하지만 이전 층의 출력이 랭크-3 텐서이기 때문에 `Dense` 층 이전에 `Flatten` 층으로 먼저 3D 출력을 1D 텐서로 펼친다.

마지막으로 10개의 클래스 분류를 위해 출력 크기를 10으로 하고 소프트맥스 활성화 함수를 사용한다.