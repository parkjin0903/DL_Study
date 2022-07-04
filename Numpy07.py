'''<removing dummy dimension>'''
import numpy as np

a = np.ones(shape=(1, 3, 10))
b = a.reshape((10, ))
c = a.reshape((-1, ))
d = a.flatten()
e = a.reshape(*a.shape[1:])
f = a[0, ...]
g = b[..., 0]

a = np.arange(9).reshape((1, -1))
b = np.arange(9).reshape((-1, 1))
c = a[0, :]
d = b[:, 0]

a = np.ones(shape=(1, 1, 4, 1, 3, 1))
b = np.squeeze(a)
d = a.squeeze()

'''<changing dimension>'''
a = np.random.normal(size=(3, 4, 5, 6))
b = np.swapaxes(a, 0, 1) #0과 1 부분 바뀜 (두 개의 차원 바꿀 때)
c = np.swapaxes(a, 0, -1) #마지막 차원 -1 이 주로 쓰임

a = np.random.normal(size=(3, 4, 5, 6))
b = np.moveaxis(a, source=0, destination=-1) # 잘 안씀 기준 점에서 이동하는 점으로 표현

a = np.random.normal(size=(3, 4))
b = np.transpose(a) # 많이 씀
c = a.T

a = np.random.normal(size=(3, 4, 5))
b = np.transpose(a, axes=(2, 0, 1)) # 2차원이 0차원으로 이동 0차원이 1차원으로 이동 1차원이 2차원으로 이동

'''<Merging ndarray>'''

# np.hstack and np.vstack

a = np.random.randint(0, 10, (4, ))
b = np.random.randint(0, 10, (4, ))

vstack = np.vstack([a, b]) #[a, b] 대신 (a, b)로 대체 가능           
hstack = np.hstack([a, b]) # 주의점 : reshape를 통해 (3, 4) 과 (3, 1)처럼 조정해줘야 함

'''---------------------------------------------------------------------------'''

dataset = np.empty((0, 4)) # 스택 쌓을 대상 만들음

for iter in range(5): # 추천하지는 않음 stack은 for로 하면 메모리를 자꾸 잡아먹음
  data_sample = np.random.uniform(0, 5, (1, 4))
  dataset = np.vstack((dataset, data_sample))
  print(f"iter/shape: {iter}/{dataset.shape}")                  

# 극복 tip 방식

a = np.random.randint(0, 10, (1, 4))
b = np.random.randint(0, 10, (1, 4))
c = np.random.randint(0, 10, (1, 4))

arr_list = [a, b, c]
vstack = np.vstack(arr_list)

# 극복 구체적 예시 (vstack 사용 -> 좋은 방법)

dataset_tmp = list()
for iter in range(100):
  data_sample = np.random.uniform(0, 5, (1, 4))
  dataset_tmp.append(data_sample)

dataset = np.vstack(dataset_tmp)