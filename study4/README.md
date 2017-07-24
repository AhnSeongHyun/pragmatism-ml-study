### 실용주의 머신러닝 스터디 4

### [중급] Regression 모델의 데이터를 tensorboard에서 출력 + 개선점 도출 1
- 모델은 본인이 원하는 아무 모델이나 좋습니다.

### [중급] Classification 모델의 데이터를 tensorboard에서 출력 + 개선점 도출 2
- 모델은 본인이 원하는 아무 모델이나 좋습니다.

### [초급] Tensorboard로 3가지 모델의 학습 로그 출력 (Tensorboard 튜토리얼 수준) 
---

### tensorboard 기본 사용법 

```python 

# 출력 대상 지정 
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

# 출력 디렉토리 지정 
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("linear_reg_tensorboard/%s_%s_%s" % (activation_func.__name__, str(learning_curve), str(epoch)), sess.graph)

for step in range(epoch):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        summary = sess.run(merged, feed_dict={X: trainX, Y: trainY})
        writer.add_summary(summary, step)
```

![linear-tensorboard](./images/linear-tensorboard.png)
### MNIST tensorboard 를 이용한 튜닝

- Activation Function | Learning Rate
    - mnist_<class 'tensorflow.python.training.adam.AdamOptimizer'>_0.01_100
    - mnist_<class 'tensorflow.python.training.adam.AdamOptimizer'>_0.03_100
    - mnist_<class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'>_0.001_100
    - mnist_<class 'tensorflow.python.training.gradient_descent.GradientDescentOptimizer'>_0.01_100
    
![cnn-tensorboard](./cnn/linear-tensorboard.png)
   