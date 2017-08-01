### 실용주의 머신러닝 스터디 5

1) openai gym에서 튜토리얼 따라 강화학습 예제 돌려보기
- https://gym.openai.com/
- [openai_gym.py](openai_gym.py)

2) GAN 으로 MNIST 이미지를 생성하는 예제 따라해보기
- https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb

3) Facebook Prophet API로 비트코인 예측하기
- https://facebookincubator.github.io/prophet/docs/quick_start.html#python-api
  
- [prophet_quickstart.py](prophet_quickstart.py)
- [prophet_forecast_growth.py](prophet_forecast_growth.py)
- `m.plot()` 으로 그래프 출력 안될때 
   ```python
   from matplotlib import pyplot as plt
   plt.show()
   ```
   
-----
### GAN 관련 
GAN - LOSS FUNCTION(MIN-MAX)
DCGAN 
    - Discriminator(구분자) + Generator(생성자)
    - Reconstruction loss
    - z 벡터 산술연산이 가능 

CYCLEGAN - 스타일 변경 
DISCONGAN 

GAN 의 정확도 측정 
- 정확도 판별은 알수가 없다. 
- 가시적인 성과가 나타날 수가 없다. 
- GAN 의 단점 

최근에는 GAN 보다는 DCGAN 을 주로 쓴다. 

관련 자료
    - https://tensorflow.blog/2016/11/24/gan-pixelcnn/
    - http://www.khshim.com/archives/20
    
    
### facebook prophet

- 시계열 예측/분석
- 3가지 컴포넌트 : growth, seasonality(방학, 휴가 등), holiday(주기성 X, 전체추이에 큰 영향을 주는 이벤트)
- logistic growth : capacity(상한선)


model fitting 
- stan 
    - map : 기본값, 확률분포에서 가장 높은 값을 선택 
    - mcmc : 모형의 변동성을 더 자세히 볼 수 있음 
    
----
prediction 을 위험하다.
ETF
RNN 보다는 sequence decision making 



 
 
   
