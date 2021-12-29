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

- **paper:** https://arxiv.org/pdf/1301.3781.pdf
- **keywords:** word representation, distributed representation of words, NNLM, RNNLM, hierarchical softmax, CBOW, Skip-gram, semantic, syntactic, computational complexity 


<br>

### 목차

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

[limits of existing word representations]
- Dictionary lookup: words represented as indices in a vocabulary
 - good: simplicity, robustness, 적은 데이터로 훈련시킨 복잡한 모델보다 많은 데이터로 훈련시킨 간단한 모델의 성능이 더 좋다.
 - limits for many tasks:
   - 자동음성인식: 보통 높은 퀄리티의 데이터의 크기가 성능을 좌우한다. 주로 백만개.
   - 기계번역: 현재 말뭉치들은 몇십억개 이하의 단어를 포함하고 있다.

- 머신러닝 기술의 발전으로 최근에는 더 큰 데이터셋으로 더 복잡한 모델을 학습하는 것이 가능해졌다. 그러니 이제 복잡한 모델이 간단한 모델보다 성능이 더 좋다.
- 아마 가장 성공적인 개념은 단어의 distributed representaion(분산표현)을 사용하는 것이다. (-> 신경망기반 언어모델)


<br><br>

### Model Architectures

- training complexity는 다음 식에 비례한다.

$O = E x T x Q$

### [References]



