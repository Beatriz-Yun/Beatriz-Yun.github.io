---
layout: post
title: "[논문 리뷰] FastText: Enriching Word Vectors with Subword Information"
date: 2021-12-29 21:00
image: word2vec.JPG
tags: paper-review NLP Word2Vec Word-Representation Word-Embeddings
categories: data-intelligence-lab paper-review

use_math: true
---


## Efficient Estimation of Word Representations in Vector Space

<br>

- paper: https://arxiv.org/pdf/1607.04606.pdf
- keywords: word representation, word embedding, Skip-gram, negative sampling

<br>

### 목차

> **1 Introduction**<br>
**2 Related work**<br><br>
**3 Model**<br>
3.1 General model<br>
3.2 Subword model<br><br>
**4 Experimental setup**<br>
4.1 Baseline<br>
4.2 Optimization<br>
4.3 Implementation details<br>
4.4 Datasets<br><br>
**5 Results**<br>
5.1 Human similarity judgement<br>
5.2 Word analogy tasks<br>
5.3 Comparison with morphological representations<br>
5.4 Effect on the size of the training data<br>
5.5 Effect of the size of n-grams<br>
5.6 Language modeling<br><br>
**6 Qualitative analysis**<br>
6.1 Nearest neighbors<br>
6.2 Character n-grams and morphemes<br>
6.3 Word similarity for OOV words<br><br>
**7 Conclusion**<br>


<br><br>

### Introduction

<br>

기존 word representation의 한계점:
  1. rare words에 대한 정확한 임베딩이 어렵다.<br>
      - Word2Vec은 분포가설(distributional hypothesis)을 따르기 때문에 형태적으로 유사한 단어여도 ...에 나타나지 않으면 ...로 쳐주지 않는다.
  3. 단어의 형태학적 특성을 고려하지 못한다.

FastText는 skip-gram모델을 기반으로 subword information을 담을 수 있는 character n-gram을 활용한 모델이다.

<br>

### General Model

#### [skip-gram]
- 중심단어로부터 주변단어를 예측하는 방향으로 학습
- 주어진 window의 크기로 sliding하면서 중신단어 별 각 주변단어의 확률값을 업데이트함.

<br>

**skip-gram의 objective function**
- 중심단어로부터 주변단어를 예측할 확률값의 log-likelihood를 최대화
- context word가 등장할 확률을 정하는 방법으로 softmax함수를 사용
   - softmax함수는 context word 하나에 대해서는 잘 예측할 수 있으나, 주위 다른 단어들을 잘 예측하지 못한다.


**skip-gram의 한계점:** corpus의 차원 V에 비해 실질적으로 업데이트 계산에 사용되는 부분은 일부이기 때문에 corpus의 크기가 커질수록 비효율적인 연산이 증가한다.

<br>

**skip-gram with negative sampling**
=> 다중분류문제를 이진분류문제로 치환!
- positive example: target word의 실제 context word
- negative example: 실제 context word가 아닌 일정한 확률값으로 뽑힌 word

<br>

### Subword Model

<br>

### Conclusion




<br>


### FastText의 목적함수(objective function)


