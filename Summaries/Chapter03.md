# 케라스 창시자에게 배우는 딥러닝
## 3장, *케라스와 텐서플로 소개*

## 3.1 텐서플로란?

텐서플로(Tensorflow)는 구글에서 만든 파이썬 기반의 무료 오픈 소스 머신 러닝 플랫폼이다. 수치 텐서에 대한 수학적 표현을 적용할 수 있도록 하는 것을 목적으로 한다. 다음과 같이 넘파이 이상의 기능을 제공한다.

- 미분 가능한 모든 표현식에 대해 자동으로 그레이디언트를 계산할 수 있다.
- CPU뿐만 아니라 고도로 병렬화된 하드웨어 가속기인 GPU와 TPU에서도 실행할 수 있다.
- 텐서플로에서 정의한 계산은 여러 머신에 쉽게 분산시킬 수 있다.
- 텐서플로 프로그램은 C++, 자바스크립트, 텐서플로 라이트 등과 같은 다른 런타임에 맞게 변환할 수 있다. 따라서 텐서플로 애플리케이션을 실전 환경에 쉽게 배포할 수 있다.



## 3.2 케라스란?

케라스는 텐서플로 위에 구축된 파이썬용 딥러닝 API로 어떤 종류의 딥러닝 모델도 쉽게 만들고 훈련할 수 있는 방법을 제공한다.

텐서플로를 통해 다양한 하드웨어(CPU, GPU, TPU 등)에서 실행하고 수천 대의 머신으로 매끄럽게 확장할 수 있다.



## 3.3 케라스와 텐서플로의 간략한 역사

생략



## 3.4 딥러닝 작업 환경 설정하기

생략



## 3.5 텐서플로 시작하기

신경망 훈련은 다음과 같은 개념을 중심으로 진행된다.

- 저수준 텐서 연산, 다음과 같은 텐서플로 API로 변환될 수 있다.
  - **텐서**(신경망의 상태를 저장하는 특별한 텐서인 **변수**도 포함)
  - 덧셈, relu, matmul과 같은 **텐서 연산**
  - 수학 표현식의 그레이디언트를 계산하는 방법인 **역전파**(텐서플로의 `GradientTape` 객체를 통해 처리됨)
- 고수준 딥러닝 개념, 다음과 같은 케라스 API로 변환될 수 있다.
  - **모델**을 구성하는 **층**
  - 학습에 사용하는 피드백 신호를 정의하는 **손실 함수**
  - 학습 진행 방법을 결정하는 **옵티마이저**
  - 정확도처럼 모델의 성능을 평가하는 **측정 지표**
  - 미니 배치 확률적 경사 하강법을 수행하는 **훈련 루프**

### 3.5.1 상수 텐서와 변수

텐서플로에서 어떤 작업을 하기 위해선 텐서가 필요하다. 텐서를 만들려면 초깃값이 필요하다. 예를 들어 모두 1이거나 0인 텐서, 또는 랜덤한 분포에서 뽑은 값으로 텐서를 만들 수 있다.

**코드 3-1. 모두 1 또는 0인 텐서**
```
import tensorflow as tf

x = tf.ones(shape=(2, 1))
print(x)

x = tf.zeros(shape=(2, 1))
print(x)
```

**코드 3-2. 랜덤 텐서**
```
import tensorflow as tf

# 정규 분포(normal distribution)
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
print(x)

# 균등 분포(uniform distribution)
x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
print(x)
```

넘파이 배열에는 값을 할당할 수 있지만 텐서플로 텐서는 상수이므로 값을 할당할 수 없다.

**코드 3-3. 넘파이 배열에 값 할당하기**
```
import numpy as np

x = np.ones(shape=(2, 2))
x[0, 0] = 0.
```

**코드 3-4. 텐서플로 텐서에 값을 할당하지 못함**
```
x = tf.ones(shape=(2, 2))
x[0, 0] = 0.
# TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
```

**변수**(`tf.Variable`)를 사용하면 텐서플로에서 수정 가능한 상태를 관리할 수 있다. 이를 통해 모델을 훈련 가능하다.

변수를 만들려면 랜덤 텐서와 같이 초기값을 제공해야 한다.

**코드 3-5. 텐서플로 변수 만들기**
```
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)
```

변수의 상태는 `assign` 메소드로 수정할 수 있다.

**코드 3-6. 텐서플로 변수에 값 할당하기**
```
v.assign(tf.ones((3, 1)))
```

일부 원소에만 적용할 수도 있다.

**코드 3-7. 변수 일부에 값 할당하기**
```
v[0, 0].assign(3.)
```

`assign_add()`와 `assign_sub()`은 각각 `+=`, `-=`과 동일하다.

**코드 3-8. assign_add() 사용하기**
```
v.assign_add(tf.ones((3, 1)))
```

### 3.5.2 텐서 연산: 텐서플로에서 수학 계산하기

**코드 3-9. 기본적인 수학 연산**
```
a = tf.ones((2, 2))
b = tf.square(a)        # 제곱
c = tf.sqrt(a)          # 제곱근
d = b + c               # 두 텐서의 합(원소별 연산)
e = tf.matmul(a, b)     # 두 텐서의 점곱
e *= d                  # 두 텐서의 곱(원소별 연산)
```

중요한 점은 연산이 모두 바로 실행된다는 점이다. 이를 **즉시 실행**(eager execution) 모드라고 부른다.

### 3.5.3 GradientTape API 다시 살펴보기

미분 가능한 모든 표현에 대해 그레이디언트를 계산할 수 있다는 점이 텐서플로와 넘파이의 중요한 차이점이다.

**코드 3-10. GradientTape 사용하기**
```
input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
```

`gradient = tape.gradient(loss, weights)`와 같은 표현식은 가중치에 대한 모델 손실의 그레이디언트를 계산하는 데 가장 널리 사용되는 방법이다.

지금까지 `tape.gradient()`의 입력 텐서가 텐서플로 변수인 경우만 살펴 보았다. 실제로 입력은 어떤 텐서라도 가능하다. 하지만 텐서플로는 기본적으로 훈련 가능한 변수만 추적한다. 상수 텐서의 경우 `tape.watch()`를 호출하여 추적한다는 것을 수동으로 알려주어야 한다.

**코드 3-11. 상수 텐서 입력과 함께 GradientTape 사용하기**
```
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
```

모든 텐서에 대한 모든 그레이디언트를 계산하기 위해 필요한 정보를 미리 저장하는 것은 비용이 너무 많이 들기 때문에 필요한 방법이다. 훈련 가능한 변수는 기본적으로 감시 대상이고, 이 외에 자원 낭비를 막기 위해 테이프는 감시할 대상을 알아야 한다.

그레이디언트 테이프를 사용해 이계도(second-order) 그레이디언트도 계산할 수 있다. 예를 들어 시간에 대한 물체 위치의 그레이디언트는 물체의 속도고, 이계도 그레이디언트는 가속도이다.

수직 방향으로 낙하하는 사과의 위치를 시간에 따라 측정하고 `position(time) = 4.9 * time ** 2`임을 알았을 때의 가속도를 측정하는 예제를 살펴본다.

**코드 3-12. 그레이디언트 테이프를 중첩하여 이계도 그레이디언트 계산하기**
```
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
```

### 3.5.4 엔드-투-엔드 예제: 텐서플로 선형 분류기

먼저 선형적으로 잘 구분되는 합성 데이터를 만든다. 2D 평면의 포인트로 2개의 클래스를 가진다. 특정한 평균과 공분산 행렬(covariance matrix)을 가진 랜덤 분포에서 좌표 값을 추출한다.

공분산 행렬은 포인트 클라우드의 모양을 정하고, 평균은 포인트 클라우드의 위치를 정한다. 따라서 같은 공분산 행렬을 사용하고 평균값만 다르게 조정하여 두 개의 포인트 클라우드를 생성한다.

**코드 3-13. 2D 평면에 두 클래스의 랜덤한 포인트 생성하기**
```
num_samples_per_class = 1000
# 왼쪽 아래에서 오른쪽 위로 향하는 타원형의 포인트 클라우드에 첫 번째 클래스의 포인트 1,000개 생성
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
# 동일한 공분산 행렬에 다른 평균을 사용하여 다른 클래스의 포인트 생성
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
```

**코드 3-14. 두 클래스를 (2000, 2) 크기의 한 배열로 쌓기**
```
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
```

(2000, 1) 크기의 0 배열과 1 배열을 합쳐 타깃 레이블을 생성한다. `inputs[i]`가 클래스 0에 속하면 `targets[i, 0]`은 0이다.

**코드 3-15. 0과 1로 구성된 타깃 생성하기**
```
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))
```

**코드 3-16. 두 클래스의 포인트를 그래프로 그리기**
```
import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()
```

![합성 데이터: 2D 평면에 놓인 두 클래스의 랜덤한 포인트](image-7.png)

이제 두 포인트 클라우드를 구분할 수 있는 선형 분류기를 만든다. 선형 분류기는 하나의 아핀 변환(`prediction = W · input + b`)이며, 예측과 타깃 사이의 차이를 제곱한 값을 최소화하도록 훈련된다.

먼저 랜덤한 값과 0으로 초기화한 변수 `W`, `b`를 만든다.

**코드 3-17. 선형 분류기의 변수 만들기**
```
input_dim = 2
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim, )))
```

**코드 3-18. 정방향 패스 함수**
```
def model(inputs):
    return tf.matmul(inputs, W) + b
```

이 선형 분류기는 2D 입력을 다루기 때문에 W는 2개의 스칼라 가중치 w1, w2로 이루어진다. (`W = [[w1], [w2]]`) 반면 b는 하나의 스칼라 값이다. 따라서 어떤 입력 포인트 [x, y]가 주어지면 예측 값은 `prediction = [[w1], [w2]] · [x, y] + b = w1 * x + x2 * y + b`가 된다.

다음 코드는 손실 함수를 보여준다.

**코드 3-19. 평균 제곱 오차 손실 함수**
```
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)
```

다음은 훈련 스탭으로 훈련 데이터를 받아 손실을 최소화하도록 가중치 `W`, `b`를 업데이트한다.

**코드 3-20. 훈련 스탭 함수**
```
learning_rate = 0.1

def training_step(inputs, targets):
    # 정방향 패스
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    # 가중치에 대한 손실의 그레이디언트 계산
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    # 가중치 업데이트
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    
    return loss
```

구현을 간단히 하기 위해 미니 배치 훈련 대신 한 번에 전체 데이터를 사용하는 배치 훈련을 사용한다. 이렇게 하면 결과적으로 미니 배치 훈련 때보다 일반적으로 큰 학습률을 사용할 수 있다.

**코드 3-21. 배치 훈련 루프**
```
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"{step}번째 스텝의 손실: {loss:.4f}")
```

다음 코드는 이 선형 모델이 훈련 데이터 포인트를 어떻게 분류하는지 그린다.

```
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```

![훈련 입력에 대한 모델의 예측: 훈련 타깃과 매우 비슷하다](image-8.png)

포인트 [x, y]에 대한 예측 값은 `prediction == [[w1], [w2]] · [x, y] + b == w1 * x + w2 * y + b`이다. 따라서 클래스 0은 `w1 * x + w2 * y + b < 0.5`이고, 클래스 1은 `w1 * x + w2 * y + b > 0.5`으로 정의할 수 있다. 그리고 이것은 2D 평면 위의 직선의 방정식 `w1 * x + w2 * y + b = 0.5`이다. 이 직선보다 위에 있으면 클래스 1, 이 직선보다 아래에 있으면 클래스 0으로 분류된다. 이 직선을 그려보자.

```
import numpy as np

x = np.linspace(-1, 4, 1000)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
```

![직선으로 나타낸 선형 모델](image-9.png)

선형 분류기란 즉, 데이터에 있는 두 클래스를 잘 구분하는 직선(고차원 공간의 경우 초평면(hyperplane))의 파라미터를 찾는 것이다.



## 3.6 신경망의 구조: 핵심 Keras API 이해하기

### 3.6.1 층: 딥러닝의 구성 요소

신경망의 기본 데이터 구조는 **층**(layer)이다. 층은 하나 이상의 텐서를 입력으로 받고, 하나 이상의 텐서를 출력하는 데이터 처리 모듈이다. 대부분의 경우 **가중치**(weight)라는 층의 상태를 가진다. 가중치는 확률적 경사 하강법으로 학습되는 하나 이상의 텐서이며 신경망이 학습한 지식이 축적되어 있다.

층마다 적절한 텐서 포맷과 데이터 처리 방식이 다르다. **밀집 연결 층**(densely connected layer), **밀집 층**(dense layer) 또는 **완전 연결 층**(fully connected layer)이라고 불리는 층은 케라스의 `Dense` 클래스에 해당하고, 랭크-2 텐서에 저장된 (samples, features)와 같은 간단한 데이터에 사용 가능하다. **순환 층**(recurrent layer)이나 **1D 합성곱 층**(convolution layer, Conv1D)으로는 일반적으로 랭크-3 텐서에 저장된 시퀀스 데이터를 처리한다. **2D 합성곱 층**(Conv2D)으로는 일반적으로 랭크-4 텐서에 저장된 이미지 데이터를 처리한다.

케라스에서 딥러닝 모델을 만드는 것은 호환되는 층을 연결하여 유용한 데이터 변환 파이프라인을 구성하는 것이다.

#### 케라스의 Layer 클래스

케라스에서는 `Layer` 또는 `Layer`와 밀접하게 상호작용하는 것이 전부이다.

`Layer`는 상태(가중치), 연산(정방향 패스)을 캡슐화한 객체이다. 가중치는 `build()` 메소드에서 정의하고, 연산은 `call()` 메소드에서 정의한다.

2개의 상태 `W`, `b`를 가지고 `output = activation(dot(input, W) + b)` 연산을 수행하는 케라스 층을 구현해보자.

**코드 3-22. Layer의 서브클래스(subclass)로 구현한 Dense 층**
```
from tensorflow import keras

# 모든 케라스 층은 Layer 클래스를 상속한다다
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
        
    # build 메소드에서 가중치를 생성한다
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer="zeros")
        
    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
```

클래스 인스턴스를 생성하면 텐서를 입력으로 받는 함수처럼 사용할 수 있다.

#### 자동 크기 추론: 동적으로 층 만들기

**층 호환**(layer compatibility) 개념은 모든 층이 특정 크기의 입력 텐서만 받고 특정 크기의 출력 텐서만 반환한다는 것이다.

케라스를 사용할 때 대부분의 경우 모델에 추가하는 층은 앞선 층의 출력 크기에 맞도록 입력 크기가 동적으로 만들어진다.

Layer 클래스의 `__call__()` 메소드는 자동 크기 추론을 위해 층이 처음 본 입력 크기를 매개변수로 받는다.

### 3.6.2 층에서 모델로

케라스의 `Model` 클래스는 딥러닝 모델로, 층으로 구성된 그래프이다. 이전에 본 `Sequential` 모델은 단순히 층을 쌓은 것으로 하나의 입력을 하나의 출력으로 매핑한다. 이외에도 자주 등장하는 네트워크는 다음과 같은 것들이 있다.

- 2개의 가지(two-batch)를 가진 네트워크
- 멀티헤드(multihead) 네트워크
- 잔차 연결(residual connection)

네트워크 구조(topology)는 꽤 복잡할 수 있다. 케라스에서는 `Model` 클래스의 서브클래스를 직접 만들거나 함수형 API를 사용해 이러한 모델들을 만들 수 있다.

모델의 구조는 **가설 공간**(hypothesis space)을 정의한다. 머신 러닝이란, 사전에 정의된 **가능성 있는 공간**(space of possibility) 안에서 피드백 신호의 도움을 받아 입력 데이터의 유용한 표현을 찾는 것이다. 네트워크 구조를 선택하면 입력 데이터를 출력 데이터로 매핑하는 일련의 텐서 연산들로 가설 공간이 제한된다.

### 3.6.3 '컴파일' 단계: 학습 과정 설정

모델 구조 정의 후엔 다음의 세 가지를 선택해야 한다.

- **손실 함수**(loss function) 또는 **목적 함수**(objective function): 훈련 과정에서 최소화해야 할 값, 현재 작업에 대한 성공의 척도
- **옵티마이저**(optimizer): 손실 함수를 기반으로 네트워크의 업데이트 방향을 결정, 특정 종류의 확률적 경사 하강법(SGD)으로 구현
- **측정 지표**(metric): 분류의 정확도 등 훈련과 검증 과정에서 모니터링할 성공의 척도, 손실과 달리 훈련이 최적화하지 않으므로 미분 불가능해도 되는 것

위 세 가지를 선택한 이후엔 모델에 내장된 `compile()`과 `fit()` 메소드를 사용해 모델 훈련을 시작할 수 있다. 또는 사용자 정의 루프를 만들 수도 있다.

`compile()` 메소드는 훈련 과정을 설정한다. 매개변수로는 `optimizer`, `loss`, `metrics`(리스트)가 있다.

```
# 선형 분류기 정의
model = keras.Sequential([keras.layers.Dense(1)])

model.compile(optimizer="rmsprop",          # 옵티마이저 지정
              loss="mean_squared_error",    # 손실 함수 지정
              metrics=["accuracy"])         # 측정 지표 지정
```

**옵티마이저**
- SGD(모멘텀 선택 가능)
- RMSprop
- Adam
- Adagrad
- ...

**손실 함수**
- CategoricalCrossentropy
- SparseCategoricalCrossentropy
- BinaryCrossentropy
- MeanSquaredError
- KLDivergence
- ConsineSimilarity
- ...

**측정 지표**
- CategoricalAccuracy
- SparseCategoricalAccuracy
- BinaryAccuarcy
- AUC
- Precision
- Recall
- ...

### 3.6.4 손실 함수 선택하기

네트워크가 손실을 최소화하기 위한 편법을 사용할 수 있기 때문에 문제에 적합한 손실 함수를 선택하는 것은 중요하다. 예를 들어 2개의 클래스가 있는 분류 문제에는 이진 크로스엔트로피(binary crossentropy), 여러 개의 클래스가 있는 분류 문제에는 범주형 크로스엔트로피(catogorical crossentropy)를 사용하는 것 등이다.

### 3.6.5 fit() 메소드 이해하기

`fit()` 메소드는 `compile()` 다음에 호출되어 훈련 루프를 구현한다. 다음과 같은 매개변수가 있다.

- **훈련할 데이터(입력과 타깃)**: 일반적으로 넘파이 배열이나 텐서플로 Dataset 객체로 전달한다.
- **훈련할 에포크**(epoch) **횟수**: 전달한 데이터에서 훈련 루프를 몇 번이나 반복할지 알려준다.
- **미니 배치 경사 하강법의 각 에포크에서 사용할 배치 크기**: 가중치 업데이트 단계에서 그레이디언트를 계산하는 데 사용될 훈련 샘플 개수를 의미한다.

**코드 3-23. 넘파이 데이터로 fit() 메소드 호출하기**
```
history = model.fit(
    inputs,
    targets,
    epochs=5,
    batch_size=128
)
```

`fit()` 메소드는 `History` 객체를 반환한다. 이 객체는 딕셔너리인 `history` 속성을 가지고 있다. 이 딕셔너리는 `loss` 또는 특정 측정 지표의 이름의 키와 각 에포크 값의 리스트를 매핑한다.

### 3.6.6 검증 데이터에서 손실과 측정 지표 모니터링하기

머신 러닝의 목표는 범용적으로 잘 동작하는 모델을 얻는 것이다. 새로운 데이터에 모델이 어떻게 동작하는지 예상하기 위해 훈련 데이터의 일부를 **검증 데이터**(validation data)로 떼어 놓는 것이 표준적인 방법이다. 이 데이터를 사용하여 손실과 측정 지표를 계산할 수 있다. 이를 위해 `fit()` 메소드의 `validation_data` 매개변수를 사용한다. 검증 데이터는 훈련 데이터처럼 전달할 수 있다.

**코드 3-24. validation_data 매개변수 사용하기**
```
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]

model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)
```

검증 데이터의 손실 값을 검증 손실(validation loss)이라고 한다. 검증 목적은 모델이 학습한 것이 실제 데이터에 유용한지 모니터링하는 것이므로 훈련 데이터와 검증 데이터는 철저히 분리되어야 한다.

훈련이 끝난 후 검증 손실과 측정 지표를 계산하고 싶다면 `evaluate()` 메소드를 사용한다.

```
loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)
```

스칼라 값의 리스트가 반환되는데, 첫 번째 항목이 검증 손실이고 두 번째 항목이 검증 데이터에 대한 측정 지표 값이다.

### 3.6.7 추론: 훈련한 모델 사용하기

모델을 훈련하고 나면 이 모델을 사용하여 새로운 데이터에서 예측을 만들게 된다. 이를 **추론**(inference)이라고 한다. 간단한 방법은 모델의 `__call__()` 메소드를 호출하는 것이다.

```
predictions = model(new_inputs)
```

이 방법은 모든 입력을 한 번에 처리하기 때문에 데이터가 많다면 가능하지 않을 수 있다.

따라서 `predict()` 메소드를 사용해 작은 배치로 데이터를 순환해 넘파이 배열로 예측을 반환받을 수 있다. `__call__()` 메소드와 달리 텐서플로 Dataset 객체도 처리할 수 있다.

```
predictions = model.predict(new_inputs, batch_size=128)
```

반환되는 배열에는 모든 입력 데이터 각각에 대한 예측이 담겨 있다.

## 3.7 요약

- 텐서플로는 CPU, GPU, TPU에서 실행할 수 있는 업계 최강의 수치 컴퓨팅 프레임워크이다. 미분 가능한 어떤 표현식의 그레이디언트도 자동으로 계산할 수 있다. 여러 가지 장치에 배포할 수 있고, 자바스크립트를 포함하여 다양한 종류의 런타임에 맞도록 프로그램을 변환할 수 있다.
- 케라스는 텐서플로에서 딥러닝을 수행하기 위한 표준 API로, 이 책에서 사용하는 라이브러리이다.
- 텐서플로의 핵심 객체는 텐서, 변수, 텐서 연산, 그레이디언트 테이프이다.
- 케라스의 핵심 클래스는 Layer이다. 층은 가중치와 연산을 캡슐화한다. 이런 층들을 조합하여 모델을 만든다.
- 모델을 훈련하기 전에 옵티마이저, 손실 함수, 측정 지표를 선택하여 `model.compile()` 메소드에 지정해야 한다.
- 미니 배치 경사 하강법을 실행하는 `fit()` 메소드로 모델을 훈련할 수 있다. 또한, 이 메소드를 사용하여 모델의 훈련 과정에서 본 적 없는 검증 데이터에 대한 손실과 측정 지표를 모니터링할 수 있다.
- 모델을 훈련하고 나면 `model.predict()` 메소드를 사용하여 새로운 입력에 대한 예측을 만든다.