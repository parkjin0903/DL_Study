import sys 
import os
sys.path.append("C:/Users/qkrwl/Downloads/deep-learning-from-scratch-master") # 부모 디렉토리의 파일을 가져올 수 있도록 설정
from python.dataset.mnist import load_mnist # dataset 폴더의 mnist파일에서 load_mnist 함수 import

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)