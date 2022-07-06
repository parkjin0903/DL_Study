'''<merging ndarrays>'''

import numpy as np

# np.concatenate(vstack hstack 보다 일반화 됨. 고차원 텐서에도 편해서 이걸 선호)

a = np.random.randint(0, 10, (3, ))
b = np.random.randint(0, 10, (4, ))

concat = np.concatenate([a, b]) # 오른쪽으로 붙음
concat0 = np.concatenate([a, b], axis=0) 

dataset_tmp = list()
for iter in range(100):
  data_sample =np.random.uniform(0, 5, (1, 4))
  dataset_tmp.append(data_sample)

concat = np.concatenate(dataset_tmp, axis=0)

# np.dstack 차원 생성 (단, (100, 200, 3)을 세개 쌓으면 (100, 200, 9)가 나와서 잘 안씀)

R = np.random.randint(0, 10, (100, 200))
G = np.random.randint(0, 10, size=R.shape)
B = np.random.randint(0, 10, size=R.shape)

image = np.dstack([R, G, B]) # image.shape = (100, 200, 3)

# np.stack (dstack과 차이점에 주목 -> (100, 200)을 세개 쌓으면 (3, 100, 200)로 일관되게 나옴)
# 고차원에서 주로 사용

a = np.random.randint(0, 10, (100, 200, 300))
b = np.random.randint(0, 10, (100, 200, 300))
c = np.random.randint(0, 10, (100, 200, 300))

print(np.stack([a, b, c], axis=1).shape) # (100, 3, 200, 300)

'''<repeating ndarray>'''

# numpy.repeat(a, repeats, axis=None) : 원소별 반복

x = 3

rep = np.repeat(x, 2) # [3, 3]

x = np. array([1, 2, 3])

rep = np.repeat(x, 3) # [1 1 1 2 2 2 3 3 3]

x = np.arange(4).reshape((2, 2)) 

rep = np.repeat(x, repeats=3, axis=0) # axis=0 방향으로 [0 1] 3번 반복 후 [2 3] 3번 반복  
rep = np.repeat(x, repeats=[2, 1], axis=0) #axis=0 방향으로 [0 1] 2번 반복 후 [2 3] 1번 반복

# numpy.tile(A, reps) : 전체적으로 반복

a = np.arange(4)
tile = np.tile(a, reps=3)
tile1 = np.tile(a, reps=[1, 2]) # tile은 횟수가 아닌 차원별 반복 수 [1, 2] (list 말고도 ndarray 넣어도 됨)

a = np.arange(6).reshape((2, 3))
reps = np.array([3, 5])
tile = np.tile(a, reps=reps*a.shape)

# Application of repetition

x = np.arange(4*3).reshape((4, 3))

x = x.sum(axis=0, keepdims=True)

x = x.repeat(repeats=3, axis=0)

x = x.T

# y = x.sum(0, keepdims=True).repeat(3, 0).T

'''<making coordinates>'''

x = np.arange(-2, 3)
y = np.arange(-2, 3)

X = x.reshape((1, -1)).repeat(y.shape[0], axis=0)
Y = y.reshape((-1, 1)).repeat(x.shape[0], axis=1)

Z = np.square(X) + np.square(Y)

# numpy.meshgrid(*xi) :3차원 표현할 때 주로 사용하자. x 좌표 y 좌표 설정해주면 알아서 x * y 형식으로 돌려줌

x = np.arange(-2, 3)
y = np.arange(-2, 3)

X, Y = np.meshgrid(x, y)
Z = np.square(X) + np.square(Y)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
Z = np.square(X) + np.square(Y)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(X, Y, Z)
ax.tick_params(labelsize=20)


'''<Tricks for Fully-connected Operations for scalars>'''

# for loop 쓰기 전에 생각해보자!

# 안 좋은 예

x = np.arange(5)
y = np.arange(2, 6)

for x_ in x:
  for y_ in y:
    print(x_ + y_, end=' ')
  print()

# 조금 안 좋은 예

x = np.arange(5)
y = np.arange(2, 6)

for x_ in x:
  print(x_ + y)

# 괜찮은 예 but 고차원 불가

x = np.arange(5)
y = np.arange(2, 6)

X, Y = np.meshgrid(x, y)
Z = X + Y

X, Y, Z = X.T, Y.T, Z.T

# 좋은 예 (Broadcasting 이용)

x = np.arange(5)
y = np.arange(2, 6)

X = x.reshape((-1, 1))
Y = y.reshape((1, -1))
Z = X + Y


'''<Tricks for Fully-connected Operations for vectors>'''

X = np.random.uniform(-5, 5, (4, 2))
Y = np.random.uniform(-5, 5, (3, 2))

# 안 좋은 예

for x in X:
  for y in Y:
    add = x + y

# 좋은 예 (using broadcasting)

X = np.expand_dims(X, axis=1) # (4, 1, 2)
Y = np.expand_dims(Y, axis=0) # (1, 3, 2)
Z = X + Y # (4, 3, 2)

# 응용 Euclidean Distance

X = np.expand_dims(X, axis=1) # (4, 1, 2)
Y = np.expand_dims(Y, axis=0) # (1, 3, 2)

Z = np.sqrt(np.sum(np.square(X - Y), axis=-1)) # axis=-1 기준
