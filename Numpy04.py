'''<Element-wise Operations and Broadcasting>'''

import numpy as np

a = np.random.randint(-5, 5, (5, ))
b = np.random.randint(-5, 5, (5, ))

print(a + b) # list와 달리 연산이 가능

'''<Indexing and Slicing ndarrays>'''

a = np.arange(10)
indices = np.array([0, 3, 6, -1])
print(f"ndarray: \n{a}\n")

print(a[[0, 3, 6, -1]]) # [0 3 6 9]

print(a[indices]) # [0 3 6 9]

print(a[a%3==0]) # [0 3 6 9]

a = np.arange(1, 11)
print(a) # [ 1 2 3 4 5 6 7 8 9 10]

a[:5] = 0
print(a) # [ 0 0 0 0 0 6 7 8 9 10] (리스트는 for data_idx in range(5): a[data_idx] = 0)

a[::2] = 200

a[5:-1:3] = 300

'''--------------------------------------------------------------------------------------------'''

a = np.arange(9).reshape((3, 3))

print(a[0, 0], a[0, 1], a[0, 2]) # print(a[0][0], a[0][1], a[0][2]) 와 같음
print(a[1, 0], a[1, 1], a[1, 2])
print(a[2, 0], a[2, 1], a[2, 2])

print(a[1:, 0]) # column
print(a[1:3, 1:3])
print(a[2:, :-2])

'''--------------------------------------------------------------------------------------------'''

image = np.arange(9).reshape((3, 3))

horizontal_flip = image[:, ::-1] # (:) 모든 원소 (::-1) -1 간격
vertical_flip = image[::-1, :]
rotation_flip = image[::-1, ::-1]

'''--------------------------------------------------------------------------------------------'''

images = np.random.normal(size=(32, 100, 200))

image0 = images[0, :, :]
print(image0.shape) # (100, 200)

image0 = images[0, ...] # ... 는 모든 차원 부름
print(image0.shape) # (100, 200)

image = np.random.normal(size=(3, 500, 300))

image_r = image[0]
image_g = image[1]
image_b = image[2]

'''--------------------------------------------------------------------------------------------'''
a = np.random.randint(0, 20, (10, ))

indices = np.random.randint(0, 10, size=(2, 3, 4))
print(indices)
print(a[indices]) # indices 의 size=(2, 3, 4) 꼴로 만듬. 구성 숫자는 indices 각 위치의 숫자가 index가 됨

'''--------------------------------------------------------------------------------------------'''

# indexing with bool ndarrays -> filter

a = np.random.randint(0, 20, (10, ))
print(a)

b_indices = a % 2 == 0
print(b_indices) # false true 로 이루어진 list

print(a[b_indices]) # 짝수만 필터


