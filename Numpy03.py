'''<Changing ndarrays>'''

# numpy.reshape(a, newshape, order='C')

import numpy as np

a = np.arange(6)
b = np.reshape(a, (2, 3))

c = np.arange(24)
d = np.reshape(c, (2, 3, 4))

# ndarray.reshape(shape, order='C')

e = a.reshape((2, 3))

# -1 in np.reshape

a = np.arange(12)
b = a.reshape((2, -1))

# numpy.resize(a, new_shape) : reshape와 달리 빈 부분을 채움 (reshape는 오류남) -> reshape을 선호 (resize는 버그가 만들어짐)

a = np.arange(9)
b = np.resize(a, (2, 3, 3))

# ndarray.resize(new_shape, refcheck=True)

a = np.arange(9)
b = a.resize((2, 2))

# ndarray.flatten(order='C')

M = np.arange(9)
N = M.reshape((3, 3))
O = N.flatten() # Copy (원소 수정 가능)

# ndarray.ravel([order])

M = np.arange(9)
N = M.reshape((3, 3))
O = N.ravel() # View (원소 수정 하려면 .copy() 필요)

'''<Memory Optimization>'''

# view(같은 메모리라서 원본에 영향을 미침) / copy(다른 메모리) 

a = np.arange(5)
b = a.view()

b[0] = 100

print(a) # [100 1 2 3 4]
print(b) # [100 1 2 3 4]

a = np.arange(5)
b = a.copy()

b[0] = 100

print(a) # [0 1 2 3 4]
print(b) # [100 1 2 3 4]

'''--------------------------------------------------------------------------------------------'''

a = np.arange(5)
b = a.copy()
c = a.view() # View
d = a[0:3] # View

print(b.base is a) # False
print(c.base is a) # True
print(d.base is a) # True

'''--------------------------------------------------------------------------------------------'''

a = np.arange(5)

a = np.arange(4)
b = np.reshape(a, (2, 2)) # View
# b = np.reshape(a, (2, 2)).copy() 로 해결

b[0, 0] = 100

print(b.base is a, '\n') # True
print(a) # [100 1 2 3]
print(b) # [[100 1]
          # [ 2  3]]

a = np.arange(5)
b = np.resize(a, (2, 2)) # Copy

b[0, 0] = 100

print(b.base is a, '\n') # False
print(a) # [0 1 2 3 4]
print(b) # [[100 1]
          # [ 2  3]]

'''--------------------------------------------------------------------------------------------'''

M = np.array([1, 2, 3], np.int8)
print(M.dtype) # int8

N = M.astype(np.uint32)
O = M.astype(np.float32)
print(N.dtype) # uint32
print(O.dtype) # float32
