---
layout: default
permalink: index.html
title: Personal Homepage of foo boo
description: "Adaboost, GBM"
---

# 1. adaboost
## main idea
### random 예측보다 약간 나은 성능을 보이는 weak model은 임의의 정확한 strong model로 향상 될 수 있다.

첫 번째 모델이 먼저 문제를 풀고 앞서 해결하지 못한 어려운 케이스에 대해 다음 단계에서 집중적으로 풀어나가는 방식이다.

예를 들면 문제집 한권을 첫 번째 사람에게 풀게하고, 앞선 사람의 정답율이 낮은 문제를 다음 사람에게 제공하여 문제를 풀게한다. 그 다음 사람은 앞선 사람들이 약했던 부분에 대해 최대한 집중적으로 풀도록 시킨다. 이러한 과정을 반복하여 결합하게 되면 결과가 좋아진다는 것이다. 이와 같이 앞선 모델의 영역 별 정답률이 그 다음 사람의 공부해야 할 데이터의 비중에 영향을 미치게 되는 시퀀셜 프로세스이다.

Base learner(기저 모델)로는 선형에 가까운 linear regression,logistic regression,stump tree(split을 한번만 하는)등과 같은 계산 복잡도가 낮은 weak model을 사용한다.

![](http://hosun17.github.io/images/1.bmp)

stump tree를 사용한다고 가정하면
초기에는 개별 데이터 포인트가 선택될 확률이 항상 1/N로 uniform 하다. 하지만 모델의 결과에 따라 다음 단계에서는 각 데이터 포인트가 선택될 확률이 아래와 같이 다르게 조정된다. (현재 모델에 의해서 정분류된 데이터는 선택될 확률이 감소, 오분류된 데이터 포인트는 선택될 확률이 증가)

![](http://hosun17.github.io/images/8.PNG)

이러한 가중치 조정을 통해서 샘플링된 데이터를 가지고 두번째 stump tree를 만들고, 모델에 의한 결과(정분류 또는 오분류에)에 따라 각 데이터 포인트의 가중치가 재조정한다.

![](http://hosun17.github.io/images/9.PNG)

모델에 의한 결과에 따라 데이터 포인트의 가중치를 재조정하고 학습하는 작업을 충분히 반복한다.

![](http://hosun17.github.io/images/10.PNG)

위와 같이 충분히 반복된 계산을 통해 구해진 α들을 통해서 영역들을 적당히 결합하게 되면, 단일 모형은 단순한 stump tree일지라도, 최종적인 분류 경계면은 아래와 같이 복잡한 모형을 만들 수 있게된다.

![](http://hosun17.github.io/images/11.PNG)


## Pseudocode of AdaBoost

![](http://hosun17.github.io/images/2.bmp)

#### 1. ensemble size T를 몇 개로 할 것 인가는 Hyperparameter로 사용자가 결정한다.
#### 2. X는 input variable , y는 binary classification의 target으로 +1,-1로 표현한다.
#### 3. S에 대해 Uniform Distribution D1(i)을 Define한다.
D1(i)는 i번째 instance가 1번 Dataset에서 선택 될 확률.(Instance가 10개라면 모두 0.1로 초기화)
#### 4. Sequential process
t=1 에서 T까지 distribution Dt에 대해서 모델 ht를 학습시킨다. 학습된 모델의 ϵt(오분류율 : 모형의 예측값 ht(x)과 실제 레이블인 y이 다른 비율)를 계산하여 0.5보다 크면 제외한다. 이는 binary classifier에서 performance가 0.5보다는 커야하는 것을 의미한다.

αt(모델들을 결합할 때 사용 할 개별 모형의 가중치)는 아래와 같이 구할 수 있으며,

![](http://hosun17.github.io/images/3.PNG)

εt이 0.5에 가까울수록 αt는 0에 가까워지고, εt이 0에 가까워질수록 αt는 ∞로 커진다. 다시 말하면 모델이 랜덤에 가까우면 최종 결과물을 만들어낼 때 신뢰도는 0에 가깝고, 모델이 정확하면 정확할 수록 최종 예측할 때 그 모델에 부여하는 가중치를 크게 한다는 의미이다.

i번째 instance가 t+1 시점에서 학습용 데이터에 선택될 확률은 t 시점의 확률에 비례하며, 아래와 같이 구할 수 있다.
(Zt는 모든 데이터 포인트들이 선택될 확률의 합이 1이 되도록 해주는 normalize factor임.)

![](http://hosun17.github.io/images/2.PNG)

정분류일 경우 다음 단계에서 선택될 확률이 작아지며,
![](http://hosun17.github.io/images/12.PNG)

오분류일 경우 다음 단계에서 선택될 확률은 커진다.
![](http://hosun17.github.io/images/13.PNG)

정확도가 높은 모형에서 오분류된 데이터 포인트가 더 중요하게 판단되고 αt에 따라 다음 단계의 Dataset에 선택될 확률은 정확도가 낮은 모델에 비해 상대적으로 더 커지게 된다.

#### 5. 계산된 αt들을 통해서 영역들을 결합한다.

# 2. GBM (Gradient boosting machine)
#### main idea
현재의 모델이 residual 만큼 잘못 맞주고 있으니, 두번째 예측 모델을 만들어서 잔차 만큼만 예측하도록 학습 시키자. 1번과 2번 모형을 결과물을 합치면 정확하게 정답을 맞출 수 있을 것이다. GBM은 x들 혹은 Instance를 건드는 것이 아니라. 그대로 두고 지속적으로 y 값을 건들어요 객체들이 추정해야 하는 정답들을 바꾸는 것이다. 

![](http://hosun17.github.io/images/foo.png)

잔차에 대해 한번 만에 완벽하게 빵하고 모두 맞추지는 못하므로 앞선 모델이 맞추지 못한 만큼을 계속적으로 다음 모델이 학습시키도록 설계하는 것이다.

어떻게 이러한 아이디어가 Gradient와 관련이 있을까?
Squared loss function을 f(x)에 대해 미분하게 되면

![](http://hosun17.github.io/images/foo.png)

Gradient는 아래와 같이 계산되고,

![](http://hosun17.github.io/images/foo.png)

Gradient Descent Algorithm과 마찬가지로 잔차(실제 값에서 함수의 추정 값을 뺀)는 Gradient의 반대 방향으로 움직여야 한다.

![](http://hosun17.github.io/images/foo.png)

![](http://hosun17.github.io/images/foo.png)

Tree1 : Regression을 tree로 한다는 것은 Split point를 기준으로 해당하는 영역의 y값의 평균을 추정하는 것이다. 계단형의 함수가 나온다. 
Tree2 : y값은 실제 값과 앞선 Tree에 의해서 추정된 값과의 차이에 의해서 결정됨. 여전히 잔차가 남아있으므로 그 잔차들을 기준으로 Tree를 추정한다… 이것을 계속적으로 반복하여 추정한다. 언젠가는 맞추겠지…


해당하는 그래디언트 구한다. 거기에 대해서 로스펑션을 구한다. 해당하는 실제값과 추정값의 차이인 잔차를 구한다.
그 잔차를 다음 단계의 y 값으로 치환한다. 
최종적인 결과물은 앞서 구했던 결과물 들을 싹 다 더해준다. 
Y바는 정답인데 +-1로 정의된 정답이고 F : 현재 모형에서의 예측 값 
정분류가 된다면 로스는 0에 가까워 지고, 오분류가 되면 + 이므로 로스가 커진다.

데이터 셋에 노이즈가 있는 상태에서 잔차를 다음단계의 Y값으로 넣는다는 것은 노이즈를 외우겠다는 것인데 실질적으로 오리지널 GBM으로 학습시키면 노이즈에 굉장히 민감하면서 과적합이 되는 모형이 됨.

가우시안 노이즈를 준 데이터를 오리지널 GBM으로 학습시킨 결과 practical하지 않은 모델이 학습되었다. 
과적합을 방지하기 위해 학습을 덜하는 방법…
### 1. subsampling

원래는 오리지널 GBM은 모든 데이터에 대한 loss를 구해서 다음 단계에서 학습을 시켜야 하는데 80, 70% 이렇게 subsampling을 하여 학습을 시킨다. 예를들어 10000개의 데이터를subsampling rate을 0.8로 두겠다라고 한다면 8000개의 데이터만 다음 단계에서 학습하는 방식이다.  

### 2. Shrinkage factor 

Original GBM의 Output= F1(x) + f2(x)+ … ft(x)이나 과적합이 너무 많아, Shrinkage factor를 주는 방법 (1보다 작은 값)
예를 들어 f1(x) +0.9f2(x)+0.9제곱f3(x)… 일부러 뒤쪽의 모형에 대해서는 최종 예측을 할 때 결합 가중치를 낮추어 놓는 방법.

### 3. 

neuralnet에서 일정 횟 수의 iteration이 반복되면 더 이상 학습시키지 않는 것처럼, 앙상블 population size를 좀 적게 두어 끝까지 학습을 시켜 잔차를 끝까지 추정하는 것이 아니라 일정 수준이 되면 중단하는 방법.


### GBM(Gradient boosting machine)의 장점 

GBM의 가장 큰 장점 중 하나는 변수의 상대적 중요도(통계적인 유의성은 아님)를 산출할 수 있다는 것이다.


예를 들어 


![](http://hosun17.github.io/images/foo.png)

Influence j(T)  








