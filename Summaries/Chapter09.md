# 케라스 창시자에게 배우는 딥러닝
## 9장, *컴퓨터 비전을 위한 고급 딥러닝*

## 9.1 세 가지 주요 컴퓨터 비전 작업

컴퓨터 비전 분야에서는 다음과 같은 세 가지의 주요한 작업들이 있다.

- **이미지 분류**(image classification): 이미지에 하나 이상의 레이블을 할당하는 것이 목표이다. 한 범주만 선택하는 단일 레이블 분류, 또는 여러 범주를 선택할 수 있는 다중 레이블 분류가 있다.
- **이미지 분할**(image segmentation): 이미지를 다른 영역으로 나누거나 분할하는 것이 목표이다. 각 영역이 일반적으로 하나의 범주를 나타낸다.
- **객체 탐지**(object detection): 이미지에 있는 관심 객체 주변에 **바운딩 박스**(bounding box)라고 부르는 사각형을 그리는 것이 목표이다. 각 사각형이 하나의 클래스에 연관된다.

객체 탐지는 너무 전문적이고 복잡하기 때문에 이 책에는 담기지 않았다. 이번 장에서는 이미지 분할에 대한 예제를 살펴본다.



## 9.2 이미지 분할 예제

딥러닝을 사용한 이미지 분할은 모델을 사용하여 이미지 각 픽셀에 클래스를 할당한다. 이미지 분할에는 두 종류가 있다.

- **시맨틱 분할**(semantic segmentation): 각 픽셀이 독립적으로 하나의 의미를 가진 범주로 분류된다. 예를 들어 이미지에 고양이가 두 마리 있어도 둘 다 단순히 'cat'으로 분류된다.
- **인스턴스 분할**(instance segmentation): 이미지 픽셀을 범주로 분류하는 것에 더해 개별 객체 인스턴스를 구분한다. 예를 들어 이미지에 고양이가 두 마리 있으면 하나는 'cat 1', 다른 하나는 'cat 2'로 분류된다.

이 예제에서는 시맨틱 분할에 초점을 맞춘다.

실습을 위해 Oxford-IIIT Pets 데이터셋을 사용한다. 이 데이터셋은 다양한 품종의 고양이, 강아지 사진 7,390개와 각 사진의 전경-배경 분할 마스크를 포함한다. **분할 마스크**(segmentation mask)는 이미지 분할에서 레이블에 해당한다. 입력 이미지와 동일한 크기의 이미지이고 컬러 채널은 하나이다. 픽셀에 담긴 정수값은 입력 이미지에서 해당 픽셀의 클래스를 나타낸다. 이 데이터셋의 분할 마스크 픽셀 값은 다음과 같은 의미를 갖는다.

- 1(전경)
- 2(배경)
- 3(윤곽)

먼저 wget, tar 셀 명령어로 데이터셋을 내려받고 압축을 해제한다.

```
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```

입력 파일 경로와 분할 마스크 파일 경로를 각각 리스트로 구성한다.

```
import os

input_dir = "images/"
target_dir = "annotations/trimaps/"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".jpg")]
)

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith(".png") and not fname.startswith(".")]
)
```

입력과 분할 마스크를 출력해 보자.

```
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

plt.axis("off")
plt.imshow(load_img(input_img_paths[9]))
```

![샘플 이미지](image-59.png)

이 이미지에 해당하는 타깃(분할 마스크)은 다음과 같다.

```
def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    
img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
display_target(img)
```

![타깃 마스크](image-60.png)

그 다음 입력과 타깃을 2개의 넘파이 배열로 로드하고 이 배열을 훈련과 검증 세트로 나눌 것이다.

```
import numpy as np
import random

img_size = (200, 200)
num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))
def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img

# 전체 이미지를 input_imgs에 float32 배열로 로드하고 타깃 마스크는 순서를 유지한 채 targets에 uint8로 로드한다
input_imgs = np.zeros((num_imgs, ) + img_size + (3, ), dtype="float32")
targets = np.zeros((num_imgs, ) + img_size + (1, ), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])
    
num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]
```

이제 모델을 정의한다.

```
from tensorflow import keras
from tensorflow.keras import layers

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3, ))
    x = layers.Rescaling(1./255)(inputs)
    # 패딩이 특성 맵 크기에 영향을 미치지 않도록 padding="same"으로 설정한다
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    
    # 각 출력 픽셀을 3개의 범주 중 하나로 분류하기 위해 3개의 필터와 소프트맥스 활성화 함수를 가진 Conv2D 층으로 모델을 종료한다
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size=img_size, num_classes=3)
model.summary()
```

`summary()` 출력 결과는 다음과 같다.

```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 200, 200, 3)]     0         
                                                                 
 rescaling (Rescaling)       (None, 200, 200, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 100, 100, 64)      1792      
                                                                 
 conv2d_1 (Conv2D)           (None, 100, 100, 64)      36928     
                                                                 
 conv2d_2 (Conv2D)           (None, 50, 50, 128)       73856     
                                                                 
 conv2d_3 (Conv2D)           (None, 50, 50, 128)       147584    
                                                                 
 conv2d_4 (Conv2D)           (None, 25, 25, 256)       295168    
                                                                 
 conv2d_5 (Conv2D)           (None, 25, 25, 256)       590080    
                                                                 
 conv2d_transpose (Conv2DTr  (None, 25, 25, 256)       590080    
 anspose)                                                        
                                                                 
 conv2d_transpose_1 (Conv2D  (None, 50, 50, 256)       590080    
 Transpose)                                                      
                                                                 
 conv2d_transpose_2 (Conv2D  (None, 50, 50, 128)       295040    
 Transpose)                                                      
                                                                 
 conv2d_transpose_3 (Conv2D  (None, 100, 100, 128)     147584    
 Transpose)                                                      
                                                                 
 conv2d_transpose_4 (Conv2D  (None, 100, 100, 64)      73792     
 Transpose)                                                      
                                                                 
 conv2d_transpose_5 (Conv2D  (None, 200, 200, 64)      36928     
 Transpose)                                                      
                                                                 
 conv2d_6 (Conv2D)           (None, 200, 200, 3)       1731      
                                                                 
=================================================================
Total params: 2880643 (10.99 MB)
Trainable params: 2880643 (10.99 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

모델의 처음 절반은 이미지 분류에서 사용하는 컨브넷과 닮아 있다. 3번의 다운샘플링(downsampling)을 통해 마지막 합성곱 층의 활성화 출력은 (25, 25, 256) 크기로 끝난다. 처음 절반 부분은 이미지를 작은 특성 맵으로 인코딩한다. 각 위치가 원본 이미지의 더 큰 영역에 대한 정보를 담고 있다. 일종의 압축으로 해석이 가능하다.

이 모델의 처음 절반은 또한 `MaxPooling2D` 층을 사용하지 않고 **스트라이드**(stride)를 추가하여 다운샘플링하고 있다. 이미지 분할의 경우 모델의 출력으로 픽셀별 타깃 마스크를 생성해야 하므로 정보의 공간상 위치에 많은 관심을 두기 때문에 스트라이드를 사용한 것이다. 풀링 방식 사용 시 풀링 윈도우 안의 위치 정보가 완전히 삭제된다.

따라서 최대 풀링은 분류 작업에는 잘 맞지만 분할 작업에는 상당히 부적합하다. 반면 스트라이드 합성곱은 위치 정보를 유지하면서 특성 맵을 다운샘플링하는 작업에 더 잘 맞는다.

모델의 나머지 절반은 `Conv2DTranspose` 층을 쌓은 것이다. 이 층은 지금까지 적용한 변환을 거꾸로 적용하여 특성 맵을 업샘플링(upsampling)한다. 따라서 특성 맵이 압축되더라도 다시 원본 크기로 복원할 수 있다.

이제 모델을 컴파일하고 훈련한다.

```
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
]
history = model.fit(train_input_imgs, train_targets,
                    epochs=50,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets))
```

훈련과 검증 손실을 그래프로 나타낸다.

```
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
```

![훈련과 검증 손실 곡선](image-61.png)

검증 손실이 가장 작은 최상의 모델을 다시 로드하여 분할 마스크를 예측하는 방법을 알아본다.

```
from tensorflow.keras.utils import array_to_img

model = keras.models.load_model("oxford_segmentation.keras")

i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)

display_mask(mask)
```

![테스트 이미지](image-62.png)
![예측된 분할 마스크](image-63.png)

이제 최고 수준의 모델을 만들기 위해 전문가가 빠르고 정확한 결정을 내릴 수 있도록 만드는 멘탈 모델과 사고 과정이 필요하다. 이를 위해 **아키텍처 패턴**(architecture pattern)에 대해 알아보자.



## 9.3 최신 컨브넷 아키텍처 패턴

모델의 아키텍처(architecture)는 사용할 층, 층의 설정, 층을 연결하는 방법 등 모델을 만드는 데 사용된 일련의 선택이다. 이런 선택은 모델의 **가설 공간**(hypothesis space)을 정의한다. 경사 하강법이 검색할 수 있는 가능한 함수의 공간으로 파라미터는 모델의 가중치이다. 좋은 가설 공간은 현재 문제와 솔루션에 대한 **사전 지식**(prior knowledge)을 인코딩한다. 예를 들어 합성곱 층을 사용하는 것은 입력 이미지에 있는 패턴이 이동 불변성이 있음을 미리 알고 있다는 의미이다.

컨브넷 아키텍처의 모범 사례들은 대표적으로 **잔차 연결**(residual connection), **배치 정규화**(batch normalization), **분리 합성곱**(separable convolution)이 있다.

먼저 거시적인 관점에서 시스템 아키텍처에 대한 모듈화-계층화-재사용(Modularity-Hierarchy-Reuse, MHR) 공식을 알아보자.

### 9.3.1 모듈화, 계층화 그리고 재사용

복잡한 시스템을 단순화하기 위해 복잡한 구조를 **모듈화**(modularity)하고, 모듈을 **계층화**(hierarchy)하고, 같은 모듈을 적절하게 여러 곳에서 **재사용**(reuse)한다. 재사용은 또한 **추상화**(abstraction)의 다른 말이다. 이것이 MHR(Modularity-Hierarchy-Reuse) 공식이다.

딥러닝 자체는 경사 하강법을 통한 연속적인 최적화에 이 공식을 적용한 것이다. 탐색 공간을 모듈(층)로 구조화하여 깊게 계층을 구성한다. 여기에서 모든 것을 재사용할 수 있다. 예를 들어 합성곱은 다른 공간 위치에서 동일한 정보를 재사용하는 것이다.

딥러닝 모델 아키텍처는 이러한 모듈화, 계층화, 재사용을 영리하게 활용하는 것이다. 인기 있는 모든 컨브넷 아키텍처는 단순히 층으로만 구성되어 있지 않고 반복되는 층 그룹(블록(block) 또는 모듈(module))으로 구성되어 있다. 예를 들어 VGG16 구조는 [합성곱, 합성곱, 최대 풀링] 블록이 반복되는 구조이다.

또한, 대부분의 컨브넷은 피라미드와 같은 계층 구조를 가지는 경우가 많다. 합성곱 필터 개수가 점차 커지면서 특성 맵 크기는 줄어든다.

계층 구조가 깊으면 특성 재사용과 이로 인한 추상화를 장려하기 때문에 본질적으로 좋다. 일반적으로 작은 층을 깊게 쌓은 모델이 큰 층을 얕게 쌓은 모델보다 좋다. 그러나 **그레이디언트 소실**(vanishing gradient) 문제 때문에 층을 쌓을 수 있는 정도에 한계가 있다. 이 문제는 핵심 아키텍처 패턴 중 하나인 잔차 연결을 탄생시킨다.

### 9.3.2 잔차 연결

정보가 잡음이 있는 채널을 통해 순차적으로 전달될 때 일어나는 에러 누적은 옮겨 말하기(Telephone) 게임으로 은유할 수 있다.

순차적인 딥러닝 모델에서 역전파는 옮겨 말하기 게임과 매우 비슷하다. 예를 들어 다음과 같이 함수가 연결되어 있다고 가정하자.

```
y = f4(f3(f2(f1(x))))
```

이 게임은 f4의 출력에 기록된 오차(모델의 손실)를 기반으로 연결된 각 함수의 파라미터를 조정하는 방식이다. f1을 조정하기 위해선 f2, f3, f4에 오차 정보를 통과시켜야 한다. 하지만 연속적으로 놓인 함수들 각각에는 일정량의 잡음이 포함되어 있다. 함수의 연결이 너무 깊으면 잡음이 그레이디언트 정보를 압도하기 시작하고 역전파가 동작하지 않게 된다. 즉, 모델이 전혀 훈련되지 않는다. 이를 **그레이디언트 소실**(vanishing gradient) 문제라고 한다.

이는 연결된 각각의 함수를 비파괴적으로 만듦으로써 해결할 수 있다. 즉, 이전 입력에 담긴 잡음 없는 정보를 유지시킨다. 이를 구현하는 가장 쉬운 방법이 **잔차 연결**(residual connection)이다.

![처리 블록을 돌아가는 잔차 연결](image-64.png)

정말 간단하게 층이나 블록의 입력을 출력에 더하기만 하면 된다. 잔차 연결은 relu 활성화 함수, 드롭아웃 층처럼 파괴적이거나 잡음이 있는 블록들을 돌아가는 정보의 지름길(information shortcut)이다. 이전 층의 오차 그레이디언트 정보가 잡음 없이 네트워크 깊숙히 전파되게 만든다.

잔차 연결은 다음과 같이 구현할 수 있다.

**코드 9-1. 잔차 연결 의사 코드**
```
# 입력 텐서
x = ...
# 원본 입력을 별도로 저장한다. 이를 잔차라고 부른다.
residual = x
# 이 계산 블록은 파괴적이거나 잡음이 있을 수 있다.
x = block(x)
# 원본 입력을 층의 출력에 더한다. 최종 출력은 항상 원본 입력의 전체 정보를 보존한다.
x = add([x, residual])
```

입력을 블록의 출력에 다시 더한다는 것은 입력과 출력의 크기가 같아야 함을 의미한다. 출력 크기가 다를 경우엔 활성화 함수가 없는 1X1 Conv2D 층을 사용하여 잔차를 원하는 출력 크기로 선형적으로 투영할 수 있다. 블록에 있는 합성곱 층은 패딩 때문에 공간 방향으로 다운샘플링되지 않도록 padding="same" 옵션을 사용한다. 또는 최대 풀링 층으로 인한 다운샘플링에 맞추기 위해 잔차 투영에 스트라이드를 사용할 수 있다.

**코드 9-2. 필터 개수가 변경되는 잔차 블록**
```
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
residual = layers.Conv2D(64, 1)(residual)
x = layers.add([x, residual])
```

**코드 9-3. 최대 풀링 층을 가진 잔차 블록**
```
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
residual = layers.Conv2D(64, 1, strides=2)(residual)
x = layers.add([x, residual])
```

조금 더 구체적으로 여러 개의 블록으로 구성된 간단한 컨브넷을 구성해 보자. 각 블록은 2개의 합성곱 층과 하나의 선택적인 최대 풀링 층으로 이루어져 있고 각 블록마다 잔차 연결을 가진다.

```
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)

def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x

x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

```
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_4 (InputLayer)        [(None, 32, 32, 3)]          0         []                            
                                                                                                  
 rescaling_1 (Rescaling)     (None, 32, 32, 3)            0         ['input_4[0][0]']             
                                                                                                  
 conv2d_13 (Conv2D)          (None, 32, 32, 32)           896       ['rescaling_1[0][0]']         
                                                                                                  
 conv2d_14 (Conv2D)          (None, 32, 32, 32)           9248      ['conv2d_13[0][0]']           
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 16, 16, 32)           0         ['conv2d_14[0][0]']           
 g2D)                                                                                             
                                                                                                  
 conv2d_15 (Conv2D)          (None, 16, 16, 32)           128       ['rescaling_1[0][0]']         
                                                                                                  
 add_2 (Add)                 (None, 16, 16, 32)           0         ['max_pooling2d_1[0][0]',     
                                                                     'conv2d_15[0][0]']           
                                                                                                  
 conv2d_16 (Conv2D)          (None, 16, 16, 64)           18496     ['add_2[0][0]']               
                                                                                                  
 conv2d_17 (Conv2D)          (None, 16, 16, 64)           36928     ['conv2d_16[0][0]']           
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 8, 8, 64)             0         ['conv2d_17[0][0]']           
 g2D)                                                                                             
                                                                                                  
 conv2d_18 (Conv2D)          (None, 8, 8, 64)             2112      ['add_2[0][0]']               
                                                                                                  
 add_3 (Add)                 (None, 8, 8, 64)             0         ['max_pooling2d_2[0][0]',     
                                                                     'conv2d_18[0][0]']           
                                                                                                  
 conv2d_19 (Conv2D)          (None, 8, 8, 128)            73856     ['add_3[0][0]']               
                                                                                                  
 conv2d_20 (Conv2D)          (None, 8, 8, 128)            147584    ['conv2d_19[0][0]']           
                                                                                                  
 conv2d_21 (Conv2D)          (None, 8, 8, 128)            8320      ['add_3[0][0]']               
                                                                                                  
 add_4 (Add)                 (None, 8, 8, 128)            0         ['conv2d_20[0][0]',           
                                                                     'conv2d_21[0][0]']           
                                                                                                  
 global_average_pooling2d (  (None, 128)                  0         ['add_4[0][0]']               
 GlobalAveragePooling2D)                                                                          
                                                                                                  
 dense (Dense)               (None, 1)                    129       ['global_average_pooling2d[0][
                                                                    0]']                          
                                                                                                  
==================================================================================================
Total params: 297697 (1.14 MB)
Trainable params: 297697 (1.14 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```

이렇듯 잔차 연결을 사용하면 그레이디언트 소실에 대해 걱정하지 않고 원하는 깊이의 네트워크를 만들 수 있다.

### 9.3.3 배치 정규화

**정규화**(normalization)는 머신 러닝 모델에 주입되는 샘플들을 균일하게 만드는 광범위한 방법이다. 이 방법은 모델이 학습하고 새로운 데이터에 잘 일반화되도록 돕는다. 가장 일반적인 형태는 데이터가 정규 분포(가우스 분포)를 따른다고 가정하고 분포를 원점에 맞춘 후 분산이 1이 되도록 조정하는 것이다.

```
normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)
```

Dense나 Conv2D 층에 들어가는 입력 데이터가 정규화되어 있더라도 출력 데이터가 동일한 분포를 가질 것이라고 기대하긴 어렵다. 따라서 활성화 함수의 출력을 정규화하는 아이디어가 제안된다. 이것이 배치 정규화(batch normalization)의 역할이다. 훈련하는 동안 현재 배치 데이터의 평균과 분산을 사용하여 샘플을 정규화한다. 대표성을 가질 만큼 충분히 크지 않다면 훈련에서 본 배치 데이터에서 구한 평균과 분산의 지수 이동 평균을 사용한다.

2015년 아이오페와 세게디가 배치 정규화를 처음 제안했을 때는 내부 공변량 변화(internal covariate shift)를 감소시키기 때문에 도움이 된다고 언급되었지만, 누구도 왜 배치 정규화가 도움이 되는지를 확실히 알지 못한다. 가설은 많지만 정확한 근거는 존재하지 않는다.

실제로 배치 정규화의 주요 효과는 잔차 연결과 매우 흡사하게 그레이디언트의 전파를 도와주는 것으로 보인다. 따라서 더 깊은 네트워크를 구성할 수 있게 된다. 매우 깊은 네트워크라면 여러 개의 `BatchNormalization` 층을 포함해야 훈련할 수 있다. 케라스에 포함된 ResNet50, EfficientNet, Xception 등의 고급 컨브넷 구조는 배치 정규화를 많이 사용한다.

`BatchNormalization` 층은 어떤 층 다음에도 사용할 수 있다.

아직 논란의 여지가 있지만, 일반적으로 활성화 층 이전에 배치 정규화 층을 놓는 것이 권장된다. 다음 두 코드의 차이점에 유의한다.

**코드 9-4. 피해야 할 배치 정규화 사용법**
```
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
```

**코드 9-5. 배치 정규화 사용법: 활성화 층이 마지막에 온다**
```
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
```

이렇게 하는 이유는 배치 정규화가 입력 평균을 0으로 만들지만 relu 활성화 함수는 0을 기준으로 값을 필터링하기 때문이다. 활성화 함수 이전에 정규화를 수행하면 활성화 함수의 활용도가 극대화된다. 물론 순서가 바뀐다고 해도 모델은 여전히 훈련될 것이며, 항상 더 나쁜 결과가 되지는 않을 것이다.

`BatchNormalization` 층이 있는 모델을 미세 조정할 때 이 층들을 동결하는 것이 좋다. 그렇지 않으면 내부 평균과 분산이 계속 업데이트되어 Conv2D 층에 적용할 매우 작은 업데이트를 방해할 수 있다.

### 9.3.4 깊이별 분리 합성곱

**깊이별 분리 합성곱**(depthwise separable convolution) 층은 `Conv2D`를 대체하면서 훈련할 모델 파라미터가 더 적고 부동 소수점 연산이 더 적고 모델의 성능을 몇 퍼센트 높일 수 있다. 케라스에서는 `SeparableConv2D`에 구현되어 있다. 이 층은 입력 채널별로 따로따로 공간 방향의 합성곱을 수행한다. 그 다음 점별 합성곱(pointwise convolution)(1X1 합성곱)을 통해 출력 채널을 합친다.

![깊이별 분리 합성곱: 깊이별 합성곱 다음에 점별 합성곱이 뒤따른다](image-65.png)

이는 공간 특성의 학습과 채널 방향 특성의 학습을 분리하는 효과를 낸다. 합성곱은 이미지상의 패턴이 특정 위치에 묶여 있지 않다는 가정에 의존한다. 마찬가지로 깊이별 분리 합성곱은 중간 활성화에 있는 **공간상의 위치**가 **높은 상관관계**를 가지지만 채널 간에는 **매우 독립적**이라는 가정에 의존한다. 심층 신경망에 의해 학습되는 이미지 표현의 경우 이 가정이 일반적으로 맞기 때문에 모델이 훈련 데이터를 더 효율적으로 사용할 수 있도록 도와준다. 처리할 정보 구조에 대한 강한 가정을 가진 모델은 이 가정이 맞는 한 더 좋은 모델이다.

깊이별 분리 합성곱은 일반 합성곱보다 훨씬 적은 개수의 파라미터를 사용하고 더 적은 수의 연산을 수행하면서 유사한 표현 능력을 가지고 있다. 수렴이 더 빠르고 쉽게 과대적합되지 않는 작은 모델을 만든다. 이런 장점은 제한된 데이터로 밑바닥부터 모델을 훈련할 때 특히 중요하다.

케라스에 포함된 고성능 컨브넷 구조인 Xception의 기반으로 깊이별 분리 합성곱이 사용되었다. 더 자세한 이론적 배경은 "Xception: Deep Learning with Depthwise Separable Convolutions" 논문에 소개되어 있다.

### 9.3.5 Xception 유사 모델에 모두 적용하기

지금까지 배운 컨브넷 아키텍처 원칙을 정리하면 다음과 같다.

- 모델은 반복되는 층 **블록**으로 조직되어야 한다. 블록은 일반적으로 여러 개의 합성곱 층과 최대 풀링 층으로 구성된다.
- 특성 맵의 공간 방향 크기가 줄어듦에 따라 필터 개수는 증가해야 한다.
- 깊고 좁은 아키텍처가 넓고 얕은 것보다 낫다.
- 층 블록에 잔차 연결을 추가하면 깊은 네트워크를 훈련하는 데 도움이 된다.
- 합성곱 층 다음에 배치 정규화 층을 추가하면 도움이 될 수 있다.
- Conv2D 층을 파라미터 효율성이 더 좋은 SeparableConv2D 층으로 바꾸면 도움이 될 수 있다.

이런 아이디어를 하나의 모델에 적용해 보자. 이 모델은 작은 버전의 Xception 모델과 비슷하다. 이 모델을 이전 장에서 본 강아지 vs 고양이 데이터셋에 적용한다. 데이터 로딩과 모델 훈련 방식은 동일하게 하고 컨브넷 구조만 다음과 같이 변경한다.

```
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
# 분리 합성곱의 기본 가정은 RGB 이미지에는 맞지 않으므로 첫 번째 층은 일반적인 Conv2D이다다
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

for size in [32, 64, 128, 256, 512]:
    residual = x
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    
    x = layers.add([x, residual])
    
# Flatten 층 대신 GlobalAveragePooling2D 층을 사용한다
x = layers.GlobalAveragePooling2D()(x)
# 규제를 위해 드롭아웃 층을 사용한다
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks,
)
```