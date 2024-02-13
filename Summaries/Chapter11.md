# 케라스 창시자에게 배우는 딥러닝
## 11장, *텍스트를 위한 딥러닝*

## 11.1 자연어 처리 소개

컴퓨터 과학에서는 사람의 언어를 기계어와 구분하기 위해 자연어(natural language)라고 부른다. 모든 기계어는 설계된 것으로, 규칙이 먼저 완성된 후 언어를 사용한다. 그러나 자연어는 반대로 먼저 사용되고 나중에 규칙이 생긴다. 자연어는 유기체와 매우 비슷하게 진화 과정에 의해 형성되기 때문에 자연이라는 단어가 붙는다. 결과적으로 기계어는 매우 구조적이고 엄격하며 고정된 어휘에서 정확하게 정의된 개념을 표현하기 위해 정확한 문법 규칙을 사용한다. 반면 자연어는 복잡하고 모호하고 혼란스럽고 불규칙하고 끊임없이 변화한다.

자연어 처리(Natural Language Processing, NLP) 시스템은 응용 언어학(applied linguistics)의 입장에서 가장 먼저 시도되었다. 엔지니어와 언어학자는 복잡한 언어 규칙을 직접 만들어 기초적인 기계 번역(machine translation)을 수행하거나 간단한 챗봇(chatbot)을 만들었다. 하지만 언어는 항상 규칙에 맞을 수 없고 쉽게 체계화할 수 없다.

수동으로 규칙을 만드는 방식은 1990년대까지 지배적이었다. 하지만 컴퓨팅 파워의 증가에 따라 1980년대 후반부터 자연어 처리에 머신 러닝 방법이 적용되기 시작한다. 머신 러닝 방법의 초기 시도는 결정 트리 기반이었다. 이전 시스템의 if/then/else 같은 규칙을 자동으로 찾도록 한 것에 지나지 않았다. 그 다음에는 로지스틱 회귀와 같은 통계적 방법이 빠르게 퍼졌다. 이후 학습된 파라미터를 가진 모델이 완전히 자리를 잡아감에 따라 언어학자의 필요성이 점차 사라졌다.

이것이 현대적인 NLP이다. 머신 러닝과 대규모 데이터셋을 사용해서 컴퓨터에 언어를 이해하는 능력을 부여하는 것이 아니라, 입력으로 언어를 받아 유용한 결과를 반환하도록 하는 것이다. 예를 들어 다음과 같은 것들이 있다.

- "이 글의 주제는 무엇인가요?" (텍스트 분류)
- "이 텍스트에 부적절한 내용이 들어 있나요?" (콘텐츠 필터링)
- "이 텍스트가 긍정적인가요? 아니면 부정적인가요?" (감성 분석)
- "문장을 완성하기 위해 다음 단어는 무엇이 되어야 하나요?" (언어 모델링)
- "이 문장은 독일어로 어떻게 되나요?" (번역)
- "이 글을 하나의 문단으로 요약하면 어떻게 되나요?" (요약)
- 기타

텍스트 처리 모델은 사람처럼 언어를 이해하는 것이 아니라 입력에 있는 통계적인 규칙성을 찾는 것뿐이다. 이는 여러 가지 간단한 작업을 수행하는 데 충분하다. 컴퓨터 비전이 픽셀에 적용하는 패턴 인식인 것처럼 NLP는 단어, 문장, 문단에 적용되는 패턴 인식이다.

NLP 도구(결정 트리, 로지스틱 회귀)는 1990년대부터 2010년대까지 느리게 발전했다. 대부분의 연구자들의 관심사는 특성 공학이었고 2014~2015년 즈음부터 마침내 많은 연구자가 순환 신경망, 특히 LSTM의 언어 이해 능력을 분석하기 시작한다. 2015년부터 2017년까지 순환 신경망이 급성장하는 NLP 분야를 지배했다. 이때 LSTM 모델이 요약(summarization)에서 질문-대답(question-answering), 기계 번역까지 많은 중요 작업에서 최고 수준의 성능을 달성한다.

마지막으로 2017~2018년 즈음에 새로운 아키텍처인 트랜스포머(Transformer)가 등장해 RNN을 대체하였다. 트랜스포머가 짧은 기간 동안 자연어 처리 분야에서 상당한 진전을 이루어 내었고, 오늘날 대부분의 NLP 시스템이 트랜스포머를 기반으로 한다.



## 11.2 텍스트 데이터 준비

미분 가능한 함수인 딥러닝 모델은 수치 텐서만 처리할 수 있다. 따라서 원시 텍스트는 입력으로 사용할 수 없다. **텍스트 벡터화**(vectorization)는 텍스트를 수치 텐서로 바꾸는 과정이다. 벡터화 과정은 다양한 모양과 형식을 띠지만 동일한 템플릿을 따른다.

- 처리하기 쉽도록 텍스트를 **표준화**(standardization)한다. 소문자로 바꾸거나 구두점을 제거하는 등의 작업이다.
- 텍스트를 **토큰**(token)이라고 부르는 단위로 분할한다. 문자, 단어, 단어의 그룹이 토큰이 될 수 있다. 이를 **토큰화**(tokenization)이라고 한다.
- 각 토큰을 수치 벡터로 바꾼다. 일반적으로 데이터에 등장하는 모든 토큰을 우선 **인덱싱**(indexing)한다.

### 11.2.1 텍스트 표준화

다음과 같은 두 문장을 생각해 보자.

- "sunset came. i was staring at the Mexico sky. isnt nature splendid??"
- "Sunset came; I stared at the México sky. isn't nature splendid?"

두 문장은 거의 동일함에도 바이트 문자열로 바꾸면 매우 다른 표현이 된다. 

텍스트 표준화는 모델이 인코딩 차이를 고려하지 않도록 두 문장 간의 차이점과 같은 것들을 제거하기 위한 기초적인 특성 공학의 한 형태이다. 이는 머신 러닝에만 국한된 것이 아니고, 검색 엔진과 같은 것을 만들 때도 동일한 작업이 필요하다.

가장 간단하고 널리 사용되는 텍스트 표준화 방법 중 하나는 소문자로 바꾸고 구두점 문자를 삭제하는 것이다. 앞의 두 문장에 이를 적용하면 다음과 같이 바뀐다.

- "sunset came i was staring at the mexico sky isnt nature splendid"
- "sunset came i stared at the méxico sky isnt nature splendid"

두 문장이 훨씬 비슷해졌다. 이 외에도 'é'와 같은 특수문자를 'e'처럼 표준 형태로 바꾸는 것이 자주 사용된다.

마지막으로 머신 러닝에서 드물게 사용되는 **어간 추출**(stemming)이라는 고급 표준화 패턴이 있다. 동사의 시제 표현 차이처럼 같은 단어가 형태만 변형된 경우 공통된 하나의 표현으로 바꾸는 것이다. 예를 들어 "was staring", "stared"가 둘 다 "[stare]"가 된다. 결국 두 문장이 동일한 인코딩을 갖게 될 수 있다.

- "sunset came i [stare] at the mexico sky isnt nature splendid"

이런 표준화 기법을 사용함으로써 모델에 필요한 훈련 데이터가 줄어들고 일반화가 더 잘 된다. 그러나 표준화를 함으로써 일정량의 정보가 삭제될 수도 있다는 점에 유의한다. 예를 들어 질문을 추출하는 모델을 만들고 싶다면 '?'는 유용한 신호가 될 수 있으므로 반드시 별도의 토큰으로 다루어야 한다.

### 11.2.2 텍스트 분할(토큰화)

텍스트 표준화 이후엔 **토큰화**(tokenization)를 통해 벡터화할 단위인 토큰으로 나누어야 한다. 세 가지 방법으로 이를 수행할 수 있다.

- **단어 수준 토큰화**: 토큰이 공백 또는 구두점으로 구분된 부분 문자열이다. 비슷한 다른 방법으로 가능한 경우 단어를 부분 단어(subword)로 더 나눌 수 있다. 예를 들어 "staring"을 "star+ing"으로, "called"를 "call+ed"로 다루는 것이 있다.
- **N-그램(N-gram) 토큰화**: 토큰이 N개의 연속된 단어 그룹이다. 예를 들어 "the cat" 또는 "he was"는 2-그램 또는 바이그램(bigram) 토큰이다.
- **문자 수준 토큰화**: 각 문자가 하나의 토큰이다. 실제로 잘 쓰이지 않고, 텍스트 생성이나 음성 인식과 같은 특별한 작업에서만 사용된다.

일반적으로 단어 수준 토큰화나 N-그램 토큰화를 항상 사용할 것이다.

텍스트 처리 모델에는 단어의 순서를 고려하는 **시퀀스 모델**(sequence model)과 입력 단어의 순서를 무시하고 집합으로 다루는 **BoW 모델**(Bag-of-Words model)이 있다. 시퀀스 모델을 만들 때는 단어 수준 토큰화를 사용하고, BoW 모델을 만들 때는 N-그램 토큰화를 사용한다. N-그램은 인공적으로 모델에 국부적인 단어 순서에 대한 소량의 정보를 주입한다.

### 11.2.3 어휘 사전 인덱싱

텍스트를 토큰으로 나눈 후 각 토큰을 수치 표현으로 인코딩해야 한다. 토큰을 해싱(hashing)하여 고정 크기의 이진 벡터로 바꾸는 것처럼 상태가 없는 방식을 사용할 수 있다. 하지만 실전에서는 훈련 데이터에 있는 모든 토큰의 인덱스(어휘 사전(vocabulary))를 만들어 어휘 사전의 각 항목에 고유한 정수를 할당하는 방법을 사용한다.

예를 들어 다음과 같다.

```
vocabulary = {}
for text in dataset:
    text = standardize(text)
    tokens = tokenize(text)
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
```

그 다음 이 정수를 신경망이 처리할 수 있도록 원-핫 벡터 같은 벡터 인코딩으로 바꿀 수 있다.

```
def one_hot_encode_token(token):
    vector = np.zeros((len(vocabulary), ))
    token_index = vocabulary[token]
    vector[token_index] = 1
    return vector
```

이 단계에서는 보통 훈련 데이터상의 최빈 단어 2~3만 개 정도로만 어휘 사전을 제한한다.

어휘 사전 인덱스에서 새로우 토큰을 찾을 때 토큰이 항상 존재하리란 법은 없다. 따라서 이런 상황을 다루기 위해 예외 어휘(out of vocabulary) 인덱스를 사용한다. 약어로 OOV 인덱스라고 한다. 이 인덱스는 어휘 사전에 없는 모든 토큰에 대응되고 일반적으로 1이다. 실제로 `token_index = vocabulary.get(token, 1)`과 같이 인덱스를 추출한다. 정수 시퀀스를 단어로 디코딩할 때는 1이 "[UNK]" 같은 OOV 토큰으로 변환된다.

일반적으로 사용하는 특별한 토큰이 두 개 있다. 첫 번째는 인덱스 0으로 사용되는 마스킹(masking) 토큰이고, 두 번째는 인덱스 1으로 사용되는 OOV 토큰이다. OOV 토큰이 "인식할 수 없는 단어"라는 의미인 반면 마스킹 토큰은 "단어가 아니라 무시할 수 있는 토큰"이라는 뜻이다. 특히 시퀀스 데이터를 패딩하기 위해 사용된다. 배치 데이터는 동일해야 하기 때문에 배치에 있는 모든 시퀀스는 길이가 같아야 한다. 따라서 길이가 짧은 시퀀스를 긴 것에 맞추기 위해 패딩할 때 사용할 수 있다. IMDB 데이터셋의 정수 시퀀스 배치 역시 이런 식으로 패딩되어 있다.

### 11.2.4 TextVectorization 층 사용하기

텍스트 벡터화 과정은 다음과 같이 파이썬으로 쉽게 구현할 수 있다.

```
import string

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text
                       if char not in string.punctuation)
        
    def tokenize(self, text):
        return text.split()
    
    def make_vocabulary(self, dataset):
        self.vocabulary = { "": 0, "[UNK]": 1 }
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict(
            (v, k) for k, v in self.vocabulary.items()
        )
    
    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]
    
    def decode(self, int_sequence):
        return " ".join(
            self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence
        )

vectorizer = Vectorizer()
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]
vectorizer.make_vocabulary(dataset)
```

다음과 같이 사용한다.

```
>>> test_sentence = "I write, rewrite, and still rewrite again"
>>> encoded_sequence = vectorizer.encode(test_sentence)
>>> print(encoded_sequence)
[2, 3, 5, 7, 1, 5, 6]
>>> decoded_sentence = vectorizer.decode(encoded_sequence)
>>> print(decoded_sentence)
i write rewrite and [UNK] rewrite again
```

하지만 이런 방식은 성능이 높지 않기 때문에 실전에서는 케라스의 `TextVectorization` 층을 사용한다. 이 층은 `tf.data` 파이프라인이나 케라스 모델에 사용할 수 있다.

`TextVectorization` 층은 다음과 같이 사용한다.

```
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(
    output_mode="int",
)
```

`TextVectorization` 층은 기본적으로 텍스트 표준화를 위해 소문자로 바꾸고 구두점을 제거하며 토큰화를 위해 공백으로 나눈다. 하지만 사용자 정의 함수를 제공하여 표준화와 토큰화 절차를 조정할 수 있기 때문에 충분히 유연하다. 이때 사용자 정의 함수는 표준 문자열이 아닌 `tf.string` 텐서를 처리해야 한다. 예를 들어 이 층의 기본적인 동작은 다음 함수와 동일하다.

```
import re
import string
import tensorflow as tf

def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]", ""
    )

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

text_vectorization = TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)
```

텍스트 말뭉치(corpus)의 어휘 사전을 인덱싱하려면 문자열을 반환하는 `Dataset` 객체나 파이썬 문자열의 리스트로 이 층의 `adapt()` 메소드를 호출하면 된다.

```
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]
text_vectorization.adapt(dataset)
```

`get_vocabulary()` 메소드를 사용해 계산된 어휘 사전을 추출할 수 있다. 정수 시퀀스로 인코딩된 텍스트를 다시 단어로 변환할 때 유용하다. 어휘 사전의 처음 두 항목은 마스킹 토큰(인덱스 0), OOV 토큰(인덱스 1)이다. 어휘 사전의 항목은 빈도 순으로 정렬되어 있다. 따라서 실제 데이터셋의 경우 "the", "a" 같은 매우 흔한 단어가 먼저 나온다.

**코드 11-1. 어휘 사전 출력하기**
```
>>> text_vectorization.get_vocabulary()
['',
 '[UNK]',
 'erase',
 'write',
 'then',
 'rewrite',
 'poppy',
 'i',
 'blooms',
 'and',
 'again',
 'a']
```

예시 문장을 인코딩하고 디코딩해 보자.

```
>>> vocabulary = text_vectorization.get_vocabulary()
>>> test_sentence = "I write, rewrite, and still rewrite again"
>>> encoded_sentence = text_vectorization(test_sentence)
>>> print(encoded_sentence)
tf.Tensor([ 7  3  5  9  1  5 10], shape=(7,), dtype=int64)
>>> inverse_vocabulary = dict(enumerate(vocabulary))
>>> decoded_sentence = " ".join(inverse_vocabulary[int(i)] for i in encoded_sentence)
>>> print(decoded_sentence)
i write rewrite and [UNK] rewrite again
```

`TextVectorization` 층은 대부분 딕셔너리 룩업(lookup) 연산이기 때문에 GPU 또는 TPU에서 실행할 수 없고 CPU에서만 실행된다. 따라서 GPU에서 모델을 훈련할 시 CPU가 먼저 층을 실행한 후 그 출력이 GPU로 전송될 것이다. 이는 성능에 큰 영향을 미친다.

`TextVectorization` 층을 사용하는 방법은 두 가지가 있다. 먼저 다음과 같이 `tf.data` 파이프라인에 넣는 것이다.

```
# string_dataset은 문자열 텐서를 반환하는 데이터셋이다.
int_sequence_dataset = string_dataset.map(
    text_vectorization,
    # num_parallel_calls 매개변수를 사용하여 여러 개의 CPU 코어에서 map() 메소드를 병렬화한다.
    num_parallel_calls=4,
)
```

두 번째 방법은 다음과 같이 모델의 일부로 만드는 것이다.

```
# 문자열을 기대하는 심볼릭 입력을 만든다.
text_input = keras.Input(shape=(), dtype="string")
# 텍스트 벡터화 층을 적용한다.
vectorized_text = text_vectorization(text_input)
# 일반적인 함수형 API 모델처럼 그 위에 새로운 층을 추가할 수 있다.
embedded_input = keras.layers.Embedding(...)(vectorized_text)
output = ...
model = keras.Model(text_input, output)
```

벡터화 단계가 모델의 일부이면 모델의 나머지 부분과 동기적으로 수행된다. 훈련 단계마다 GPU에 놓인 모델의 나머지 부분이 CPU에 놓인 텍스트 벡터화 층 출력을 기다린다는 의미이다. 반면 `tf.data` 파이프라인에 층을 넣으면 CPU에서 데이터 처리를 비동기적으로 수행할 수 있다. 즉, GPU가 벡터화된 데이터 배치에서 모델을 실행하는 동안 CPU가 원시 문자열의 다음 배치를 벡터화할 수 있는 것이다.

따라서 모델을 GPU 또는 TPU에서 훈련하는 경우 첫 번째 방법을 사용하는 것이 최상의 성능을 얻기 위한 방법이다.



## 11.3 단어 그룹을 표현하는 두 가지 방법: 집합과 시퀀스

단어는 범주형 특성(미리 정의된 집합에 있는 값)이고 이를 처리하는 방법은 정해져 있다. 단어를 특성 공간의 차원으로 인코딩하거나 범주 벡터(단어 벡터)로 인코딩한다. 하지만 단어를 문장으로 구성하는 방식인 단어 순서를 인코딩하는 방법이 더 중요하다.

자연어에서 순서 문제는 흥미로운 문제이다. 시계열 데이터처럼 표준이 되는 순서가 없다. 언어가 다르면 비슷한 단어도 매우 다른 방식으로 나열된다. 순서가 확실히 중요하지만 의미와의 관계는 간단하지 않다.

단어의 순서 표현 방식은 여러 종류의 NLP 아키텍처를 발생시키는 핵심이다. 가장 쉬운 방법은 순서를 무시하고 텍스트를 순서가 없는 단어의 집합으로 처리하는 것이다. 이것이 **BoW 모델**이다. 시계열의 타임스텝처럼 한 번에 하나의 단어씩 등장하는 순서대로 처리해야 한다고 결정할 수 있다. 이런 경우엔 순환 신경망을 활용할 수 있다. 마지막으로 하이브리드 방식도 가능한데, 트랜스포머 아키텍처는 기술적으로 순서에 구애받지 않지만 처리하는 표현에 단어 위치 정보를 주입한다. 이를 통해 순서를 고려하면서 RNN과 달리 문장의 여러 부분을 동시에 볼 수 있다. 단어 순서를 고려하기 때문에 RNN과 트랜스포머 모두 **시퀀스 모델**이라고 부른다.

역사적으로 초기 머신 러닝 NLP 애플리케이션은 대부분 BoW 모델을 사용했다. 그러나 순환 신경망이 재발견되면서 2015년에서야 시퀀스 모델에 대해 관심이 생기기 시작했고, 오늘날에도 여전히 두 방식이 혼용되고 있다.

### 11.3.1 IMDB 영화 리뷰 데이터 준비하기

다시 IMDB 영화 리뷰 감성 분류 데이터셋을 사용할 것인데, 이전에 사용했던 것과 달리 원시 IMDB 텍스트 데이터부터 처리해 보자.

먼저 앤드류 마스(Andrew Maas)의 스탠포드 페이지에서 데이터셋을 내려받고 압축을 해제한다.

```
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!rm -f aclImdb_v1.tar.gz
```

다음과 같은 구조를 가진 aclImdb 디렉터리가 생긴다.

```
aclImdb/
...train/
......pos/
......neg/
...test/
......pos/
......neg/
```

예를 들어 train/pos/ 디렉터리에는 1만 2,500개의 텍스트 파일이 담겨 있다. 각 파일은 훈련 데이터로 사용할 긍정적인 영화 리뷰의 텍스트를 담고 있으며, 부정적인 리뷰는 neg/ 디렉터리에 담겨 있다. 모두 합쳐 훈련용으로 2만 5,000개의 텍스트 파일이 있고 테스트용으로 또 다른 2만 5,000개의 파일이 있다.

train/unsup 디렉터리도 있는데, 이 디렉터리는 필요하지 않으므로 삭제한다.

```
!rm -r aclImdb/train/unsup
```