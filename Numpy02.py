'''making ndarrays'''

# random.randn(d0, d1, ... , dn) : 평균 0, 표준편차 1 sampling

# random.normal(loc=0.0, scale=1.0, size=None) : loc = 평균, scale = 표준편차, size = 데이터 수

import numpy as np

normal1 = np.random.normal(loc=-2, scale=1, size=(200, ))

normal2 = np.random.normal(loc=[-2, 0, 3],
                          scale=[1, 2, 5],
                          size=(200, 3))

normal3 = np.random.normal(loc=-2, scale=1, size=(3, 3))

# random.rand(d0, d1, ... , dn) : [0, 1) 사이에서 뽑음

uniform = np.random.rand(1000)
uniform1 = np.random.rand(2, 3, 4)

# random.uniform(low=0.0, high=1.0, size=None)

uniform2 = np.random.uniform(low=-10, high=10, size=(10000, ))

# random.randint(low, high=None, size=None, dtype=int)

randint = np.random.randint(low=0, high=7, size=(20, )) #0은 inclusive 7은 exclusive
print(randint) # [3 5 2 4 0 0 0 3 1 1 1 4 2 3 1 0 4 3 2 6]

'''------------------------------------------------------------------------------------'''

'''meta-data of ndarrays'''

scalar_np = np.array(3.14)
vector_np = np.array([1, 2, 3])
matrix_np = np.array([[1, 2], [3, 4]])
tensor_np = np.array([[[1, 2, 3],
                       [4, 5, 6]]
                      
                      [[11, 12, 13]
                       [14, 15, 16]]])

print(scalar_np.ndim) # 0
print(vector_np.ndim) # 1
print(matrix_np.ndim) # 2
print(tensor_np.ndim) # 3

print("shape / dimension")
print("{} / {}".format(scalar_np.shape, len(scalar_np.shape))) # () / 0
print("{} / {}".format(vector_np.shape, len(vector_np.shape))) # (3.) / 1
print("{} / {}".format(matrix_np.shape, len(matrix_np.shape))) # (2, 2) / 2
print("{} / {}".format(tensor_np.shape, len(tensor_np.shape))) # (2, 2, 3) / 3

a = np.array([1, 2, 3]) # (3,)
b = np.array([[1, 2, 3]]) # (1, 3)
c = np.array([[1], [2], [3]]) # (3, 1)

M = np.ones(shape=(2, 3, 4, 5, 6)) 
print("size of M:", M.size) # size = 720

#int uint float

int8_np = np.array([1.5, 2.5, 3.5], dtype=np.int8) 
uint8_np = np.array([1.5, 2.5, 3.5], dtype=np.uint8)

print(int8_np) # [1 2 3]
print(uint8_np) # [1 2 3]


