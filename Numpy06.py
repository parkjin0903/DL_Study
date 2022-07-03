import numpy as np

'''<Rounding and Sorting>'''

# numpy.around(a, decimals=0, out=None)
# numpy.round_(a, decimals=0, out=None)
# ndarray.round(decimals=0, out=None)

# numpy.ceil(x) : 내림 / 올림
# numpy.floor(x)

# numpy.trunc : 그냥 날려버림

x = np.random.uniform(-5, 5, (5, ))

np_around = np.around(x, decimals =2) # 최종적으로 표기할 소수점 자리수 이므로 3번째 자리에서 반올림 (decimals 없으면 첫째 자리에서 반올림)
np_round_ = np.round_(x, decimals=2)
x_round = x.round(decimals=2)

trunc_where = np.where(x >= 0, np.floor(x), np.ceil(x))
trunc = np.trunc(x)

# trunc 소수점 trick 
trunc = 0.1*np.trunc(10*x)

# trunc frac trick

int_part = np.trunc(x)
frac_part = x - int_part

# numpy.sort(a, axis=-1, kind=None, order=None)
# numpy.argsort(a, axis=-1, kind=None, order=None)

sort = np.sort(x) # 오름차순
argsort = np.argsort(x) # index return
sort_rev = np.sort(x)[::-1] # 내림차순

pred = np.random.uniform(0, 100, (5, ))
pred /= pred.sum()

top3_pred = np.sort(pred)[::-1][:3] # 가장 높은 3개 slicing
top3_indices = np.argsort(pred)[::-1][:3]

'''--------------------------------------------------------'''

x = np.random.randint(0, 100, (4, 5))

sort = np.sort(x, axis=0)[::-1, :]
argsort = np.argsort(x, axis=0)[::-1, :]

# 대문자 contant로 짜는 암묵적 룰

PI = np.pi
E = np.e

# numpy.rad2deg(x)
# numpy.deg2rad(x)


degree = np.array([30, 45, 60, 90, 180, 360])
rad = np.deg2rad(degree)
degree = np.rad2deg(rad)

x = np.deg2rad(np.linspace(0, 360, 11))

# numpy.sinh() numpy.cosh() numpy.tanh()

sin, cos = np.sin(x), np.cos(x)
tan = np.tan(x)

# numpy.square(x)
# numpy.sqrt(x)
# numpy.cbrt(x) : x**(1/3)
# numpy.reciprocal(x) : x**(-1)

y2 = np.reciprocal(np.square(a))
z2 = np.reciprocal(np.sqrt(a))

# numpy.power(x1, x2) : x1 을 x2제곱(x1은 ndarray도 가능)
# 주의점 : x**-1 할 때 x 값을 int로 하면 값이 안나오고 정수로 변환하느랴 0 이 나올 수 있으므로 input 값을 float로 두고 하자

# numpy.log(x) : 0 포함 시 정의역 범위 문제로 오류

log = np.log(a)
exp = np.exp(log)

a = np.random.uniform(1, 5, (4, ))
b = np.random.uniform(1, 5, (4, ))

print((np.log(a) + np.log(b)).round(3)) # print(np.log(a*b).round(3))

log2 = np.log(a)/np.log(2)

# binary entropy

p = np.random.uniform(0, 1, (4, )) # 0과 1을 꼭 제외하자
be_e = -(p*np.log(p) + (1-p)*np.log(1-p))
be_2 = -(p*np.log(p)/np.log(2) + (1-p)*np.log(1-p)/np.log(2))

# numpy.dot(a, b)

u = np.random.randint(0, 5, (4, ))
v = np.random.randint(0, 5, (4, ))
b = np.random.randint(0, 5, ())

affine = np.dot(u, v) + b
activation = 1/(1 + np.exp(-affine))

'''---------------------------------'''

M = np.random.randint(0, 5, (3, 4))
U = np.random.randint(0, 5, (4, ))

mat_vec_mul = np.empty((3, ))
for row_idx, row in enumerate(M):
  mat_vec_mul[row_idx] = np.dot(row, u) # np_matmul = np.matmul(M, u)

# numpy.matmul(x1, x2)

X = np.random.uniform(0, 5, (3, 4))
W = np.random.uniform(0, 5, (4, 5))
b = np.random.uniform(0, 5, (5, ))

affine = np.matmul(X, W) + b
activation = 1/(1 + np.exp(-affine))

'''<Dimensionality Manipulation>'''

a = np.arange(9)
b = a.reshape((1, 9))
c = a.reshape((9, 1, 1))

# TIP: reshape 만을 이용하면 유지 보수에 불리해짐 -> unpack을 이용하자 (단, 이 방법 이용시 가운데에 차원 추가는 어려움)

a = np.random.normal(size = (100, 200))
b = a.reshape((1, *a.shape))
c = a.reshape((*a.shape, 1))

# using slicing 1 -> newaxis를 활용하자(가운데 차원 추가 용이)

a = np.arange(9)

row_vec1 = a[np.newaxis, :] # reshape((1, -1))
row_vec2 = a[None, :]

col_vec1 = a[:, np.newaxis]
col_vec2 = a[:, None]

# using slicing 2 

a = np.random.normal(size = (100, 200))

b = a[np.newaxix, ...] # (1, 100, 200)
c = a[..., np.newaxis]

# using expand_dims API : 명시적으로 보임

a = np.arange(9)
b = np.expand_dims(a, axis=0)
c = np.expand_dims(a, axis=1)
d = np.expand_dims(a, axis=(0, 1))