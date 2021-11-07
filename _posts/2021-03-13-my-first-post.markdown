---
layout: post
title: "블로그 바꿔가는 과정"
date: 2021-03-13 22:22:58 +0200
image: sehui1.jpg
tags:
categories: blog
---

#### 2021-03-12
- 프로필 사진 바꾸기
- sns 링크 바꾸기
- disqus 설정 (웹사이트 추가 후, short name 수정)
- newsletter부분 삭제

  (_layouts/post.html 43번째 줄에서 % include newsletter.html % 지웠음.)


#### 2021-03-13
- Google Analytics 설정 (추적 ID 수정)
- 카테고리 페이지에서 카테고리 누르면 404오류 정정

  (category 폴터 추가 후, 카테고리명.md파일 생성하고 layout: category-page)
- 포스트 비공개 설정 (published: false)
