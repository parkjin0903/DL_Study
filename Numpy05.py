# numpy.cumsum(a, axis=None, dtype=None, out=None) : 누적 합 (axis 설정 x 시 벡터로 출력됨)
# ndarray.cumsum(axis=None, dtype=None, out=None )

a = np.arange(5)

cumsum = np.cumsum(a)

a = np.arange(3*4).reshape((3, 4))

cumsum = np.cumsum(a, axis = 0)

# numpy.prod(a, axis=None , dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
# numpy.cumpord(a, axis=None, dtype=None, out=None)

prod = np.prod(a)
cumprod = np.cumprod(a)

# numpy.diff(a, n=1, axis=-1, prepend=<no value>, append=<no value>)

a= np.random.randint(0, 10, (5, )) # [2 5 4 8 1]
diff = np.diff(a) # [ 3 -1 4 -7]

# numpy.mean(a, axis=None , dtype=None, out=None, keepdims=<no value>, *, where=<no value>) : 평균값
# ndarray.mean(axis=None , dtype=None, out=None, keepdims=False, *, where=True)

# numpy.median(a, axis=None , out=None, overwrite_input=False, keepdims=False) : 중앙값

# numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)
# ndarray.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)
# numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)
# ndarray.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

scores = np.random.normal(loc=10, scale=5, size=(100, ))

var = scores.var()
std = scores.std()

# numpy.amax(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
# ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

# numpy.amin(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
# ndarray.min(axis=None, out=None, keepdims=False, inital=<no value>, where=True)

# numpy.argmin(a, axis=None, out=None)
# ndarray.argmin(axos=None, out=None)

import numpy as np

means = [50, 60, 70]
stds = [3, 5, 10]
n_student, n_class = 100, 3

scores = np.random.normal(loc=means,
                          scale=stds,
                          size=(n_student, n_class))
scores = scores.astype(np.float32)

scores_max = np.max(scores, axis=0) # max
scores_max_idx = np.argmax(scores, axis=0) # index of max

# maximum and minimum 

u = np.random.randint(0, 10, (10, ))
v = np.random.randint(0, 10, (10, ))

maximum = np.maximum(u, v)
minimum = np.minimum(u, v)

u = np.random.randint(0, 10, (10, ))
v = np.random.randint(0, 10, (10, ))

maximum = np.zeros_like(u)
maximum[u >= v] = u[u >= v]
maximum[u < v] = v[u < v]

'''----------------------------------------'''

up_vals = np.full_like(u, fill_value=100)
down_vals = np.full_like(u, fill_value=-100)

print(np.where(u > v, up_vals, down_vals)) # u > v : up_vals / u < v : down_vals
print(np.where(u > v, u, v))