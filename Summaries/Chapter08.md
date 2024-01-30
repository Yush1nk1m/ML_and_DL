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

이제 MNIST 숫자 이미지에 이 컨브넷을 훈련한다.

**코드 8-3. MNIST 이미지에서 컨브넷 훈련하기**
```
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=64)
```

테스트 데이터에서 모델을 평가하면 다음과 같은 결과가 나온다.

**코드 8-4. 컨브넷 평가하기**
```
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"테스트 정확도: {test_acc:.3f}")
313/313 [==============================] - 3s 8ms/step - loss: 0.0302 - accuracy: 0.9901
테스트 정확도: 0.990
```

2장의 완전 연결 네트워크보다 높은 성능을 기록하였다. 다음부터는 왜 컨브넷이 완전 연결 네트워크보다 잘 작동하는 것인지에 대해 알아보자.

### 8.1.1 합성곱 연산

완전 연결 층(Dense)은 입력 특성 공간의 전역 패턴(모든 픽셀에 걸친 패턴)을 학습한다. 그러나 합성곱 층은 작은 2D 윈도우와 같은 것으로 지역 패턴을 학습한다. 앞선 예제에서의 윈도우는 모두 3X3 크기였다.

이러한 핵심적인 특징은 컨브넷에 두 가지 흥미로운 성질을 제공한다.

- **학습된 패턴은 평행 이동 불변성(translation invariant)을 가진다.** 예를 들어 컨브넷은 오른쪽 아래 모서리에서 학습한 어떤 패턴을 다른 모서리에서도 인식할 수 있다. 그러나 완전 연결 네트워크는 새로운 위치에서 나타난 패턴은 완전히 새로운 패턴으로 인식한다. 이는 컨브넷이 적은 수의 훈련 샘플만으로 일반화 능력을 가진 표현을 학습할 수 있도록 해준다.
- **컨브넷은 패턴의 공간적 계층 구조를 학습할 수 있다.** 첫 번째 합성곱 층은 에지와 같은 작은 지역 패턴을 학습하고, 두 번째 합성곱 층은 첫 번째 층의 특성으로 구성된 더 큰 패턴을 학습하는 식이다. 이런 방식을 사용하여 컨브넷은 매우 복잡하고 추상적인 시각적 개념을 효과적으로 학습할 수 있다.

합성곱 연산은 **특성 맵**(feature map)이라는 랭크-3 텐서에 적용된다. 이 텐서는 2개의 **공간** 축(**높이**, **너비**)과 **깊이** 축(**채널**)으로 구성된다. 합성곱 연산은 입력 특성 맵에서 작은 패치(patch)들을 추출하고 모든 패치에 같은 변환을 적용해 **출력 특성 맵**(output feature map)을 만든다.

출력 특성 맵도 높이와 너비를 가진 랭크-3 텐서이지마니 깊이는 층의 매개변수로 결정되기 때문에 상황에 따라 상태가 다르다. 이 경우에 깊이 축의 채널은 더 이상 RGB, 흑백과 같이 특정 컬러를 의미하지 않는다. 대신 일종의 **필터**(filter)를 의미한다. 필터는 입력 데이터의 어떤 특성을 인코딩하는데, 예를 들어 입력에 얼굴이 있는지를 필터가 반영하고 있을 수 있다.

MNIST 예제에서는 첫 번째 합성곱 층이 (28, 28, 1) 크기의 특성 맵을 입력으로 받아 (26, 26, 32) 크기의 특성 맵을 출력한다. 즉, 입력에 대해 32개의 필터를 적용하는 것이다. 32개의 출력 채널 각각이 26 X 26 크기의 배열 값을 가진다. 이 값은 입력에 대한 필터의 **응답 맵**(response map)으로 입력 각 위치에서 필터 패턴에 대한 응답을 나타낸다.

깊이 축에 있는 각 차원은 하나의 **특성**(또는 필터)이고, 랭크-2 텐서인 output[:, :, n]은 입력에 대한 이 필터의 응답을 나타내는 2D 공간상의 **맵**이 된다.

합성곱은 다음 2개의 핵심적인 파라미터가 있다.

- **입력으로부터 뽑아낼 패치의 크기**: 전형적으로 3X3 또는 5X5 크기를 사용한다.
- **특성 맵의 출력 깊이**: 합성곱으로 계산할 필터 개수이다.

케라스의 `Conv2D` 층에서 이 파라미터들은 `Conv2D(output_depth, (window_height, window_width))`처럼 첫 번째, 두 번째 매개변수로 전달된다.

이제 3D 입력 특성 맵 위를 3X3 또는 5X5 크기의 윈도우가 **슬라이딩**(sliding)하면서 모든 위치에서 3D 특성 패치((window_height, window_width, input_depth) 크기)를 추출하는 방식으로 합성곱이 작동한다. 이런 3D 패치는 **합성곱 커널**(convolution kernel)이라고 불리는 하나의 학습된 가중치 행렬과의 텐서 곱셈으로 (output_depth, ) 크기의 1D 벡터로 변환된다. 동일한 커널이 모든 패치에 걸쳐서 재사용되고 변환된 모든 벡터는 (height, width, output_depth) 크기의 3D 특성 맵으로 재구성된다. 출력 특성 맵의 공간상 위치는 입력 특성 맵의 같은 위치에 대응된다.

두 가지 이유로 출력의 높이, 너비는 입력의 높이, 너비와 다를 수 있다.

- 경계 문제. 입력 특성 맵에 패딩을 추가하여 대응할 수 있다.
- **스트라이드**(stride) 사용 여부에 따라 다르다.

#### 경계 문제와 패딩 이해하기

5X5 입력 특성 맵에 3X3 윈도우를 사용하는 상황을 떠올려 보자. 윈도우의 중앙에 맞추어 출력 특성 맵을 구성하면 출력 특성 맵은 3X3 크기로 줄어든다. 이처럼 윈도우의 크기가 1X1이 아니기 때문에 입력 특성맵 가장자리 부분이 적게 추출되고 제외되는 것을 경계 문제라고 부른다.

입력과 동일한 높이, 너비의 출력 특성 맵을 얻고 싶다면 **패딩**(padding)을 사용할 수 있다. 이는 입력 특성 맵의 가장자리에 적절한 개수의 행과 열을 추가하는 기법으로, 3X3 윈도우의 경우 가장자리마다 1개의 행 또는 열, 5X5 윈도우의 경우 가장자리마다 2개의 행 또는 열을 추가하면 출력 특성 맵이 같은 크기를 갖게 된다.

`Conv2D` 층에서 패딩은 `padding` 매개변수로 설정할 수 있다. `valid` 값을 전달하면 패딩을 하지 않겠다는 의미이고, `same` 값을 전달하면 패딩을 하여 출력 특성 맵의 크기를 **같게** 만들겠다는 의미이다. 기본값은 valid이다.

#### 합성곱 스트라이드 이해하기

매 반복마다의 윈도우의 이동량을 **스트라이드**라고 정의할 수 있다. 스트라이드의 기본값은 1이고, 스트라이드가 1보다 큰 **스트라이드 합성곱**도 가능하다.

스트라이드 2를 사용했다는 것은 특성 맵의 너비와 높이가 2의 배수로 다운샘플링되었다는 뜻이다. 스트라이드 합성곱은 분류 모델에서 드물게 사용되지만 일부 유형의 모델에서는 유용하다.

분류 모델에서는 특성 맵을 다운샘플링하기 위해 스트라이드 대신에 첫 번째 컨브넷 예제에 사용된 최대 풀링(max pooling) 연산을 사용하는 경우가 많다.

### 8.1.2 최대 풀링 연산

앞선 컨브넷 예제에서 `MaxPooling2D` 층마다 특성 맵 크기가 절반으로 줄어들었다. 최대 풀링은 이처럼 스트라이드와 비슷하게 특성 맵을 강제적으로 다운샘플링한다.

최대 풀링은 입력 특성 맵에서 윈도우 크기만큼의 패치를 추출하고, 각 채널별로 최댓값을 출력한다. 합성곱과 개념적으론 비슷하지만, 추출된 패치에 학습된 선형 변환을 적용하지 않고 하드코딩된 최댓값 추출 연산을 사용한다. 최대 풀링은 보통 2X2 윈도우와 스트라이드 2를 사용하여 특성 맵을 절반 크기로 다운샘플링한다. 이에 반해 합성곱은 전형적으로 3X3 윈도우와 스트라이드 1을 사용한다.

왜 특성 맵을 다운샘플링하는지 이해하기 위해 다운샘플링이 없는 컨브넷을 정의해 보자.

**코드 8-5. 최대 풀링 층이 빠진 잘못된 구조의 컨브넷**
```
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs)
```

모델 구조는 다음과 같다.

```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d_6 (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 conv2d_7 (Conv2D)           (None, 24, 24, 64)        18496     
                                                                 
 conv2d_8 (Conv2D)           (None, 22, 22, 128)       73856     
                                                                 
 flatten_2 (Flatten)         (None, 61952)             0         
                                                                 
 dense_2 (Dense)             (None, 10)                619530    
                                                                 
=================================================================
Total params: 712202 (2.72 MB)
Trainable params: 712202 (2.72 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

이 설정에는 두 가지 문제가 있다.

- 특성의 공간적 계층 구조 학습에 도움이 되지 않는다. 세 번째 층의 윈도우는 단순히 초기 입력의 7X7 윈도우 영역에 대한 정보만 담고 있을 뿐이다.
- 너무 많은 파라미터가 존재한다. 그렇게 때문에 심각한 과대적합이 발생하고 학습 시간도 느려질 수밖에 없다.

다운샘플링을 하는 이유는 간단히 처리할 특성 맵의 가중치 개수를 줄이기 위함이다. 또한 연속적인 합성곱 층이 점점 커진 윈도우를 통해 바라보도록 만들어 필터의 공간적인 계층 구조를 구성한다.

최대 풀링 대신 입력 패치의 채널별 평균값을 계산하여 변환하는 **평균 풀링**(average pooling)을 사용할 수도 있다. 그러나 최대 풀링이 다른 방법들보다 잘 작동하는 편이다. 특성이 특성 맵의 각 타일에서 어떤 패턴이나 개념의 존재 여부를 인코딩하는 경향이 있기 때문이다. 그래서 특성의 지도라고 부르는 것이다.

가장 나은 서브샘플링 방법은 스트라이드가 없는 합성곱으로 조밀한 특성 맵을 만들고 작은 패치에 대해 최대로 활성화된 특성을 고르는 것이다.



## 8.2 소규모 데이터셋에서 밑바닥부터 컨브넷 훈련하기

이 절에서는 강아지와 고양이 이미지 분류 문제를 다룰 것이다. 소규모 데이터셋을 이용해 2,000개의 훈련 샘플에서 작은 컨브넷을 규제 없이 훈련하여 기준이 되는 기본 성능을 만든다. 이 방법의 주요 이슈는 과대적합이 될 것이다. 그 다음 컴퓨터 비전 분야에서 과대적합을 줄이기 위해 사용되는 강력한 방법인 **데이터 증식**(data augmentation)을 사용하여 네트워크의 성능을 개선할 것이다.

다음 절에서는 소규모 데이터셋에 딥러닝을 적용하기 위한 핵심적인 기술 두 가지를 살펴본다. **사전 훈련된 네트워크로 특성을 추출**하는 것, **사전 훈련된 네트워크를 세밀하게 튜닝**하는 것이다.

소규모 데이터셋에서 이미지 분류 문제를 수행할 때는 위 세 가지 전략을 포함해야 한다.

- 처음부터 작은 모델 훈련하기
- 사전 훈련된 모델을 사용하여 특성 추출하기
- 사전 훈련된 모델을 세밀하게 튜닝하기

### 8.2.1 작은 데이터셋 문제에서 딥러닝의 타당성

모델이 작고 규제가 잘 되어 있으며 간단한 작업이라면 수백 개의 샘플로도 충분한 샘플이라고 할 수 있다. 컨브넷은 지역적이고 평행 이동으로 변하지 않는 특성을 학습하기 때문에 지각에 관한 문제에서 매우 효율적으로 데이터를 사용한다.

또한 딥러닝 모델은 매우 다목적으로 사용되며, 모델을 조금만 변경하면 다른 문제에 충분히 사용할 수 있다. 이것이 딥러닝의 가장 큰 장점들 중 하나인 특성 재사용이다.

### 8.2.2 데이터 내려받기

사용할 강아지 vs 고양이 데이터셋(Dogs vs Cats dataset)은 케라스에 포함되어 있지 않으므로 캐글에서 구한다.

```
!kaggle competitions download -c dogs-vs-cats
!unzip -qq dogs-vs-cats.zip
!unzip -qq train.zip
```

그리고 클래스별로 훈련 세트 1,000개, 검증 세트 500개, 테스트 세트 1,000개로 서브셋을 만들 것이다. 전체 데이터는 25,000개지만 작은 데이터셋에서의 성능 개선을 위해 크기를 줄인다.

**코드 8-6. 이미지를 훈련, 검증, 테스트 디렉터리로 복사하기**
```
import os, shutil, pathlib

original_dir = pathlib.Path("train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)
            
make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)
```

이제 이 문제는 균형 잡힌 이진 분류 문제가 되었다.

### 8.2.3 모델 만들기

