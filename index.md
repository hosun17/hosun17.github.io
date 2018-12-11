---
# Business Analytics : Adaboost, GBM
---
이 post는 고려대학교 산업경영공학과 DSBA연구실 강필성 교수님의 Business-Analytics강의를 바탕으로 작성되었습니다

# 1. adaboost
### main idea
### random 예측보다 약간 나은 성능을 보이는 weak model은 임의의 정확한 strong model로 향상 될 수 있다.

첫 번째 모델이 먼저 문제를 풀고 앞서 해결하지 못한 어려운 케이스에 대해 다음 단계에서 집중적으로 풀어나가는 방식이다.

예를 들면 문제집 한권을 첫 번째 사람에게 풀게하고, 앞선 사람의 정답율이 낮은 문제를 다음 사람에게 제공하여 문제를 풀게한다. 그 다음 사람은 앞선 사람들이 약했던 부분에 대해 최대한 집중적으로 풀도록 시킨다. 이러한 과정을 반복하여 결합하게 되면 결과가 좋아진다는 것이다. 이와 같이 앞선 모델의 영역 별 정답률이 그 다음 사람의 공부해야 할 데이터의 비중에 영향을 미치게 되는 시퀀셜 프로세스이다.

기저 모델(Base learner)로는 선형에 가까운 linear regression,logistic regression,stump tree(split을 한번만 하는)등과 같은 계산 복잡도가 낮은 weak model을 사용한다.

![](http://hosun17.github.io/images/1.bmp)

stump tree를 사용한다고 가정하면
초기에는 개별 데이터 포인트가 선택될 확률이 항상 1/N로 uniform 하다. 하지만 모델의 결과에 따라 다음 단계에서는 각 데이터 포인트가 선택될 확률이 아래와 같이 다르게 조정된다. (현재 모델에 의해서 정분류된 데이터는 선택될 확률이 감소, 오분류된 데이터 포인트는 선택될 확률이 증가)

(https://www.cse.buffalo.edu/~jcorso/t/CSE455/files/lecture_boosting.pdf)
![](http://hosun17.github.io/images/8.PNG)

이러한 가중치 조정을 통해서 샘플링된 데이터를 가지고 두번째 stump tree를 만들고, 모델에 의한 결과(정분류 또는 오분류에)에 따라 각 데이터 포인트의 가중치가 재조정한다.

(https://www.cse.buffalo.edu/~jcorso/t/CSE455/files/lecture_boosting.pdf)
![](http://hosun17.github.io/images/9.PNG)

모델에 의한 결과에 따라 데이터 포인트의 가중치를 재조정하고 학습하는 작업을 충분히 반복한다.

(https://www.cse.buffalo.edu/~jcorso/t/CSE455/files/lecture_boosting.pdf)
![](http://hosun17.github.io/images/10.PNG)

위와 같이 충분히 반복된 계산을 통해 구해진 α들을 통해서 영역들을 적당히 결합하게 되면, 단일 모형은 단순한 stump tree일지라도, 최종적인 분류 경계면은 아래와 같이 복잡한 모형을 만들 수 있게된다.

(https://www.cse.buffalo.edu/~jcorso/t/CSE455/files/lecture_boosting.pdf)
![](http://hosun17.github.io/images/11.PNG)

### Pseudocode of AdaBoost

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

아래와 같이 부스팅에서는 각 단계에서 데이터 포인트가 선택될 확률이 조정된 결과 특정 데이터 포인트의 선택이 집중되는 것을 볼 수 있다.

![](http://hosun17.github.io/images/3.bmp)

아래는 Class A와 B에 대해 Adaboost를 통해 실제 학습된 결과이다.

![](http://hosun17.github.io/images/24.PNG)

(https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py)
```
# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()

```


# 2. GBM (Gradient boosting machine)
### Motivation

### 추가의 회귀 모형으로 잔차를 예측하려고 한다면 어떻게 될까?

![](http://hosun17.github.io/images/4.bmp)

현재의 모델이 residual 만큼 잘못 맞주고 있으니, 그 다음 예측 모델을 만들어서 잔차 만큼만 예측하도록 학습 시키자. 각 모형을 결과물을 결합하면 정확하게 정답을 맞출 수 있을 것이다.

![](http://hosun17.github.io/images/14.PNG)

GBM에서는 Instance는 그대로 두고 지속적으로 앞선 모델의 잔차를 y 값 즉, 객체들이 추정해야 하는 정답들을 바꾸어 다음 모델을 학습시킨다. 잔차에 대해 한번 만에 완벽하게 맞추지는 못하므로 앞선 모델이 맞추지 못한 만큼을 계속적으로 다음 모델이 학습시키도록 설계한다.

### 이러한 아이디어가 어떻게 Gradient와 관련이 있을까?

![](http://hosun17.github.io/images/15.PNG)

![](http://hosun17.github.io/images/16.PNG)

Squared loss function을 f(x)에 대해 미분하게 되면 Gradient는 아래와 같이 계산되고, Gradient Descent Algorithm과 마찬가지로 잔차(실제 값에서 함수의 추정 값을 뺀)는 loss function의 negative gradient로 표현되며, loss function의 최소값을 찾기 위해서는 Gradient의 반대 방향으로 이동하여야 하기 때문이다.

### 회귀 모형의 예

![](http://hosun17.github.io/images/17.PNG)

Tree1 : Regression을 tree로 한다는 것은 Split point를 기준으로 해당 영역의 y값의 평균을 추정하는 것이며, 위와 같이 계단형의 함수가 나온다.
Tree2 : y값은 실제 값과 앞선 Tree에 의해서 추정된 값과의 차이에 의해서 결정되며, 그 값 들을 기준으로 Tree를 추정한다.
이러한 프로세스을 지속적으로 반복하여 추정한다.

### Pseudocode of Gradient

Friedman (2001), Natekin and Knoll (2013)
![](http://hosun17.github.io/images/18.PNG)

GBM의 pseudocode는 Adaboost에 비해 간단하며 아래와 같다.
#### 1. 해당하는 Gradient를 구하고, 거기에 대해 loss function을 계산한다.
#### 2. 그 다음 해당하는 실제값과 추정값의 차이인 잔차를 구하여 다음 단계의 y 값으로 치환한다.
#### 3. 최종적인 결과물은 앞서 구했던 결과물들을 모두 더해준다.

Regression과 Classification에서 사용되는 Loss function은 아래와 같다.

![](http://hosun17.github.io/images/20.PNG)


![](http://hosun17.github.io/images/21.PNG)

### Dataset에 노이즈가 있는 상태에서 잔차를 다음 단계의 Y값으로 넣는다는 것은 노이즈를 모두 학습하겠다는 것인데 노이즈에 굉장히 민감하면서 과적합이 되는 문제는 어떻게 해결해야 할까?

아래 예에서 가우시안 노이즈를 준 데이터를 오리지널 GBM을 사용한 결과 practical하지 않은 모델이 학습되는 것을 보여준다.

![](http://hosun17.github.io/images/19.PNG)

### 1. subsampling
과적합을 방지하기 위해 학습을 덜하는 방법이다. 오리지널 GBM은 모든 데이터에 대한 잔차를 구해서 다음 단계에서 학습을 시켜야 하는데 80% 혹은 70% 정도만 subsampling을 하여 학습을 시킨다. 예를 들어 10000개의 데이터를 subsampling rate을 0.8로 두겠다라고 한다면 8000개의 데이터 만을 다음 단계에서 학습시키겠다는 의미이다.

### 2. Shrinkage factor
Original GBM의 Output= f1(x) + f2(x)+ … ft(x)이나 과적합이 너무 많아, Shrinkage factor(1보다 작은 값)를 주는 방법이다.
예를 들어 f1(x) +0.9*f2(x)+0.9^2*f3(x)… 와 같이 일부러 다음 모형에 대해 최종 예측을 할 때 결합 가중치를 낮추어 준다.

### 3. Early Stopping
neuralnet에서 일정 횟 수의 iteration이 반복되면 더 이상 학습시키지 않는 것처럼 앙상블 population size를 좀 적게 두는 방법이다. 끝까지 학습을 시켜 잔차를 추정하는 것이 아니라 일정 수준이 되면 중단한다.


### GBM(Gradient boosting machine)의 장점 

GBM의 가장 큰 장점 중 하나는 통계적인 유의성은 아니지만 변수의 상대적 중요도를 산출할 수 있다는 점이다.
먼저 각 Tree의 모든 Split에서 변수의 influence를 구하고, 모든 Tree에서 계산된 변수 별 Influence의 평균을 계산함으로써 변수의 중요도를 산출한다.

![](http://hosun17.github.io/images/23.PNG)

예를 들어 Tree 모형에서 분기(Split)에 사용된 변수와 Information Gain이 주어진 상황이라면 변수의 중요도는 아래와 같이 계산될 수 있다.

![](http://hosun17.github.io/images/22.PNG)

아래는 GBM을 통한 학습한 결과와 추출된 변수의 중요도이다.

(https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)
![](http://hosun17.github.io/images/25.PNG)

```
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# #############################################################################
# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# #############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

```





