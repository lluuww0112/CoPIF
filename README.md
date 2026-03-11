# 26/02/12

생성일: 2026년 2월 12일 오후 2:22
태그: 데이터 생성, 아이디어 도출

# CoPIF : Coordinate Prompted Image Free

## 1. 선행 연구

---

> CapDec
> 
> - 사전학습된 CLIP을 그대로 사용
> - CLIP의 global vector를 이용해 captioning 모델을 image-free training을 진행
> - 이 때 image-text간의 실세계 gap을 좁히기 위해 가우시안 노이즈를 추가
> 
> IFseg
> 
> - 사전학습된 VLM의 정렬 능력을 전제
> - VLM의 임베딩 벡터를 이용해 feature map을 모사
> - segmentation VL model에 모사된 feature map을 입력, 최종 결과는 각 feature map 패치에 대응되는 logit을 이용해 분류
> - 예를 들어 고양이로 분류된 패치 지역은 고양이 segmentation 결과임
> 
> CLOSE
> 
> - image-text 간의 gap을 줄이기 위한 다양한 방법들을 실험
> - 실험 결과 text에 가우시안 노이즈를 추가하여 VLM에 입력하면 vision task 수행 능력이 향상됨을 입증
> 

## 2. 기존 Image Free의 문제점

---

> CapDec
> 
> - global vector만을 이용하여 downstream 모델을 학습
> - global vector는 맥락만을 담고 있을 뿐 위치 정보를 반영하기에는 어려움
> - 따라서 해당 논문에서도 captioning만을 수행함
> 
> IFseg
> 
> - segmentation에서는 객체의 위치가 중요할 뿐 이미지 상에 존재하는 객체가 어느 맥락에 위치하고 있는지는 중요하지 않음
> - 따라서 IFseg에서는 feature map을 만들 때 맥락이 담긴 vector를  사용하는 것이 아닌 임베딩 테이블의 벡터를 가져와 feature map모사
> - 예를 들어 흰 고양이와, 검은 고양이, 또는 흰 고양이 옆에 있는 검은 고양이 등등과 같은 정보를 feature map에 담아내지는 못함
> - 이로 인해 visual grounding 관련 task는 수행하지 못함
> 

## 3. CoPIF의 아이디어

---

> CLIP을 이용하면 맥락이 담긴 feature map을 모사할 수 있지 않을까?
> 
> - IFseg에서 가장 중요한 전제는 “사전학습된 VLM은 이미지와 텍스트 간의 정렬능력을 가지고 있다”임
> - 그렇다면 사전학습된 VLM말고 CLIP 또한 feature map 생성에 사용할 수 있는 것 아닌가?
> - 또 CLIP은 단순 단어 임베딩 뿐만 아니라 문장 임베딩 또한 가능
> - 결과적으로 IFseg에서 feature map을 구성했던 것과 비슷하게 CLIP으로 feature map을 구성한다면 REC와 같은 visul grounding또한 image free training이 가능해질 것이라 판단
> 
> 노이즈 주입
> 
> - 여러 연구에서 두 모달리티 간의 gap을 보간하기 위해 다양한 방법이 제안되어 왔지만 그 중 가장 효과적이라고 많이 언급되고 있는 것은 당연코 노이즈 주입임
> - 선행 연구 CLOSE에서도 단순 노이즈 주입, linear adapter, gap mean shift 등을 사용했지만 가장 범용적으로 높은 성능을 보이는 방법은 가우시안 노이즈 주입이며 노이즈 스케일을 하이퍼 파라미터로 취급, 각 task에 대해서 최적의 값을 제안하기도 했음
> 

## 4. Feature Map은 어떻게 만들어야 할까?
---

1. 기존 오픈 데이터셋에서 참조 표현(referring expression)과 bbox annotation을 추출
2. 실제 입력 크기를 넘지 않는 범위에서 랜덤 사이즈의 그리드 맵을 생성한 뒤, 기존 bbox 위치에 맞춰 해당 영역에 참조 표현의 임베딩 벡터를 배치
3. 참조 표현이 배치되지 않은 나머지 위치는 랜덤 벡터로 채움
4. 생성된 그리드 맵을 KNN 기반 보간법으로 최종 입력 feature map 크기에 맞춤
5. 완성된 feature map으로 image-free 학습 수행


## 5. Original Image Feature Map과 Image-Free Feature Map 사이의 차원 불일치 문제
---
텍스트 인코더를 이용해 feature map을 모사하는 것은 합리적이지만 가장 큰 문제는 text embedding 차원과 이미지 인코더가 만들어 내는 feature map의 채널(emb_dim)이 불일치한다는 문제가 존재   
이를 해결하기 위해 DenseCLIP에서 제안한 방식을 사용. 해당 방식은 다음과 같음   

> CLIP의 이미지 인코더의 상위 계층의 각 패치는 텍스트 표현(의미)와 이미 정렬되어 있음을 전제.   
> 하지만 기존 CLIP은 분류 문제를 위해 마지막 CLS 토큰에 대해서 self-attention을 수행했기 떄문에 다른 패치들은 텍스트 의미적 정렬을 잃어버리게 됨
> 따라서 DenseCLIP에서는 마지막 층의 self-attention을 제거. 다음 과정을 수행   


1. n-1번 째 출력(n번 층 입력)에 query, key 가중치는 적용하지 않고 오직  value 가중치만 1x1 conv연산으로 적용
2. value projection만 적용된 패치들에 visual projection layer를 1x1로 적용   

결과적으로 별도의 학습 없이 패치 레벨로 텍스트와 의미적 정렬을 유지시킴