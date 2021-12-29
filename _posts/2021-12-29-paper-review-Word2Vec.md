---
layout: post
title: "[Paper Review] [Word2Vec] Efficient Estimation of Word Representations in Vector Space"
date: 2021-12-08 21:45
image: word2vec.JPG
tags: paper-review NLP Word2Vec Word-Representation Word-Embeddings
categories: data-intelligence-lab paper-review
use_math: true
---


## Efficient Estimation of Word Representations in Vector Space

<br>

- **paper:** [https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)
- **keywords:** word representation, distributed representation of words, NNLM, RNNLM, hierarchical softmax, CBOW, Skip-gram, semantic, syntactic, computational complexity 


<br>

### **목차**

> Abstract <br><br>
1 Introduction <br>
1.1 Goals of the Paper <br>
1.2 Previous Work <br><br>
2 Model Architectures <br>
2.1 Feedforward Neural Net Language Model (NNLM) <br>
2.2 Recurrent Neural Net Language Model (RNNLM) <br>
2.3 Parallel Training of Neural Networks <br><br>
3 New Log-linear Models <br>
3.1 Continuous Bag-of-Words Model <br>
3.2 Continuous Skip-gram Model <br><br>
4 Results <br>
4.1 Task Description <br>
4.2 Maximization of Accuracy <br>
4.3 Comparison of Model Architectures <br>
4.4 Large Scale Parallel Training of Models <br>
4.5 Microsoft Research Sentence Completion Challenge <br><br>
5 Examples of the Learned Relationships <br>
6 Conclusion <br>
7 Follow-Up Work <br><br>
References

<br><br>

### Abstract

- **매우 큰 데이터셋**으로부터 continuous vector representations of words를 계산하는 새로운 모델 2개를 제안한다.
- representations 퀄리티는 word similarity task에서 측정하고, 결과는 neural networks를 기반으로 하는 이전 best 모델들(=NNLM, RNNLM)과 비교한다.
- 우리가 제안한 모델이 훨씬 낮은 계산비용으로 크게 정확도 향상이 됨을 확인한다. (i.e. 16개 단어로부터 높은 퀄리티의 word vectors를 학습시키는데 하루보다 덜 걸린다.)
- 게다가, syntactic(구문적), semantic(의미적) 단어 유사성을 측정하기 위해 직접 구성한 test set에 대해서는 가장 좋은 성능을 보인다.

<br><br>

### Introduction

**[limits of existing word representations]**
- Dictionary lookup: words represented as indices in a vocabulary
 - good: simplicity, robustness, 적은 데이터로 훈련시킨 복잡한 모델보다 많은 데이터로 훈련시킨 간단한 모델의 성능이 더 좋다.
 - limits for many tasks:
   - 자동음성인식 - 보통 높은 퀄리티의 데이터의 크기가 성능을 좌우한다. 주로 백만개.
   - 기계번역 - 현재 말뭉치들은 몇십억개 이하의 단어를 포함하고 있다.

- 머신러닝 기술의 발전으로 최근에는 더 큰 데이터셋으로 더 복잡한 모델을 학습하는 것이 가능해졌다. 그러니 이제 복잡한 모델이 간단한 모델보다 성능이 더 좋다.
- 아마 가장 성공적인 개념은 단어의 distributed representaion(분산표현)을 사용하는 것이다. (-> 신경망기반 언어모델)


<br><br>

### Model Architectures

#### 모델 비교를 위한 계산 복잡도 정의
$O = E \times T \times Q$
- $E$: 훈련 epoch 수
- $T$: 훈련데이터셋의 단어 개수
- $Q$: 각 모델 구조에 의해 정의됨

<br>

**[Feedforward Neural Net Language Model (NNLM)]**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147644502-46208c01-238e-4b17-9211-8dd7af29ad7b.png" width="70%" height="70%" />
</div>

- 구성: input, projection, hidden, output layer
- 계산복잡도(softmax): $Q = N \times D + N \times D \times H + H \times V$
  - dominating term: $H \times V$
- 계산복잡도(hierarchical softmax): $Q = N \times D + N \times D \times H + H \times \log_2 (V)$
  - dominating term: $N \times D \times H$

> The dominant term is the term the one that gets biggest 

<br>

- 단점:
  - 몇 개의 단어를 볼지 미리 정해야하고, 그 수가 고정된다.
  - 이전 단어들만 고려한다.
  - 속도가 느리다

<br>

**[Recurrent Neural Net Language Model (RNNLM)]**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147646186-5c0be8af-507d-441a-af39-31a135ef18bc.png" width="70%" height="70%" />
</div>

- 구성: input, hidden, output layer
- 계산복잡도(softmax): $Q = H \times H + H \times V$
 - dominating term: $H \times V$
- 계산복잡도(hierarchical softmax): $Q = H \times H + H \times \log_2 (H)$
 - dominating term: $H \times H$

- hidden layer값들이 다시 input으로 들어온다. (short-term memory 역할)
- 학습할 단어를 순차적으로 입력하므로 input으로 사용할 단어의 개수를 정해줄 필요가 없다.

<br><br>

### Word2Vec Models

<br>

- 기존 모델들의 계산복잡도의 대부분은 비선형적인 hidden layer로부터 야기되므로 더 효율적인 모델을 만들기위해 hidden layer를 제거하였다.
- 대신 projection layer 사용

<br>

**[Continuous Bag-of-Words Model (CBOW)]**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147648185-3b1e48e0-b568-4c35-80e1-eeb76bd74891.png" width="40%" height="40%" />
</div>

- 구성: input, projection, output layer
- 계산복잡도: $Q = N \times D + D \times \log_2 (V)$
- 주변단어로 중심단어를 예측

<br>

**[Countinuous Skip-gram Model (Skip-gram)]**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147648503-8f80fcea-6ba2-409a-aaec-83b537eb2bb6.png" width="40%" height="40%" />
</div>

- 구성: input, projection, output layer
- 계산복잡도: $Q = C \times (D + \log_2 (V))$
- 중심단어로 주변단어를 예측

<br><br>

### Results & Conclusions

<br>

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147654980-dd7cda6f-59d4-48ab-9002-1845a334205d.png" width="70%" height="70%" />
</div>

- 같은 방향의 벡터를 가지는 두 쌍의 단어는 비슷한 관계를 가졌음을 추정가능
- 이 특성을 이용하여 워드 벡터간 연산 가능

<br>

- 구문론적, 의미론적 관계를 가지는 단어쌍으로 구성한 테스트셋으로 word representation 벡터의 퀄리티를 측정하였다.
- 기존의 신경망 모델들(feedforward, recurrent)에 비해 더 퀄리티 있는 워드 벡터를 훈련시키는 것이 가능해졌다.
- 계산복잡도를 크게 줄임으로써 훨씬 큰 데이터셋으로부터 보다 정확하게 고차원 워드 벡터를 계산해낼 수 있게 되었다.

<br><br>

### More to study

**[Hierarchical Softmax (계층적 소프트맥스)]**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147655231-98cf0366-9e7a-4f74-9445-7b61437cebb9.png" width="70%" height="70%" />
</div>

- 계산복잡도를 줄이기 위해 softmax 대신 사용한다.
- 각 단어를 leaf node로 가지는 binary tree를 만들어 놓고, 해당하는 단어의 확률을 계산할 때 root에서부터 해당 leaf로 가는 path에 따라 확률을 곱해나가며 단어가 등장할 최종확률을 계산한다.
- 본 논문에서는 Huffman tree를 사용한다.
  - 단어의 빈도수 순으로 정렬 후 binary tree를 만들기 때문에 빈도수가 더 큰 단어가 트리의 위쪽에 있게 된다.

**단어 등장 확률을 구하는 공식:**

<div style="text-align: center">
 <img src="https://user-images.githubusercontent.com/42146731/147659754-6b15aa8d-b57d-43cd-8b8b-065df2384959.png" width="70%" height="70%" />
</div>
 
- $L(w)$: $w$라는 단어의 leaf 노드에 도달하기 까지 path의 길이
- $n(w, i)$: root 노드에서부터 w라는 leaf노드까지의 path에서 만나는 i번째 노드
  - $n(w, 1)$은 root노드, $n(w,L(w))$은 $w$에 해당하는 leaf노드
- $ch(node)$: node의 고정된 임의의 한 자식. 왼쪽 or 오른쪽(보통 왼쪽노드를 사용한다.)
- $[x]$: x가 참이면 1, 거짓이면 -1을 반환하는 함수
- $v_n(w,j)$: n(w,j)는 $w$에 해당하는 leaf노드까지의 path에 있는 internal node들이고, $v_n(w,j)$는 이 internal node들에 대한 벡터를 뜻한다. 기존의 $W'$대신 사용.
- $h$: hidden layer의 벡터.
  - $h$대신 $v_{w_i}$로 표현하기도 한다. 이는 i번째 단어에 대한 벡터를 뜻한다.
  - input으로 들어가는 word의 vector이다. Skip-Gram의 경우에는 중심 단어의 vector, CBOW의 경우 그 주변의 context 단어들의 평균 vector.
- $\sigma$: sigmoid function. sigmoid함수를 적용해서 확률로 만든다.

<br><br>

### [References]

- [word2vec 관련 이론 정리](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
- [[Text mining] 4. Word2vec (3) Hierarchical Softmax](https://analysisbugs.tistory.com/184)
- [Q&A - Hierarchical Softmax in word2vec](https://www.youtube.com/watch?v=pzyIWCelt_E)
- [Hierarchical Softmax](https://seunghan96.github.io/ne/03.Hierarchical_Softmax/)
- [딥러닝을 이용한 자연어 처리(Natural Language Processing With Deep Learning) – CS224n 강의 노트 1-2](http://solarisailab.com/archives/959)

