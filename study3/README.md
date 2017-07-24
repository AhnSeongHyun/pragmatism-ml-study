### 실용주의 머신러닝 스터디 3

### [중급] Tensorflow 유닛테스트 하나 작성해보기
- https://stackoverflow.com/…/34204…/run-tensorflow-unit-tests

### [초급] Tensorflow Linear Regression 튜토리얼 따라하기
- https://cpuu.postype.com/post/161789

### [상급] LSTM 혹은 어느 알고리즘이든 비트코인 시계열 예측 구현하기
- https://www.facebook.com/groups/634363263421303/664340533756909/ 

- 비트코인 RAW 데이터 : bitcoin_ticker.csv
- 학습 및 테스트용 데이터 : korbit_btckrw.csv(bitcoin_ticker.csv 에서 korbit, btckrw 만 필터링 된 데이터)

- 참고 : [모두의 딥러닝:RNN](https://docs.google.com/presentation/d/1UpZVnOvouIbXd0MAFBltSra5rRpsiJ-UyBUKGCrfYoo/edit)

- INPUT : last, diff_24h, diff_per_24h, bid, ask, low, high, volume
- LABEL : last


```
X => [ 0.43727382  0.59409594  0.31512924  0.28729875  0.85172798  0.60770975 0.54615385  0.30185717]
Y => [ 0.43727382]
```

```
trainX : 43898  trainY : 43898
testX : 18814  testY : 18814
```

- Activation Func:
    - tf.nn.sigmoid : 0.12304099649190903
    - tf.nn.relu    : 0.05581469088792801
    - tf.nn.relu6   : 0.04537376016378403
    - tf.nn.tanh    : 0.13985252380371094
    - tf.nn.softsign : 0.06960246711969376
    - tf.nn.softplus : 0.040385302156209946
    
