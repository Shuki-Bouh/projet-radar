import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

ee = np.zeros(12//2, dtype='complex')
print(arr[2::4])
ee[::2] =arr[::4] + arr[2::4] * 1j
ee[1::2] = arr[1::4] + arr[3::4] * 1j

print(ee)