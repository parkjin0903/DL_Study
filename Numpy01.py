import numpy as np

# numpy.zeros(shape, dtype=float, order='C, *, like=None)
# numpy.ones(shape, dtype=float, order='C, *, like=None)

M = np.zeros(shape=(2, 3))

print(M.shape) # (2, 3)
print(M) 

# numpy.full(shape, fill_value, dtype=None, order='C, *, like=None)
# numpy.empty(shape, dtype=float, order='C', *, like=None)

M = np.full(shape=(2, 3), fill_value=3.14) # 3.14 * np.ones(shape=(2, 3))

'''-------------------------------------------------------------------'''

M = np.full(shape=(2, 3), fill_value=3.14)

zeros_like = np.zeros_like(M)
ones_like = np.ones_like(M)
full_like = np.full_like(M, fill_value=100)
empty_like = np.empty_like(M)

'''-------------------------------------------------------------------'''

# numpy.arange([start,]stop, [step,]dtype=None, *, like=None)

print(np.arange(2, 10, 2))
print(np.arange(10.5)) #10.5 보다 크지 않은 최대 정수
print(np.arange(1.5, 10.5))

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

print(np.linspace(0, 1, 5)) # array([0.  , 0.25, 0.5 , 0.75, 1. ])

a = np.linspace([1, 10, 100], [2, 20, 200], 5) #for문 없이도 가능하다는 점












