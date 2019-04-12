import numpy as np
import tensorflow as tf
import keras

a = [1,3,2]
b =[3,5,8]

c = np.stack([a,b])

d = [[1,4,8],
     [2,3,6],
     [2, 5, 6],
     [2, 7, 6]]
e = [[2,7,5],
     [4,5,3],
     [2, 9, 6],
     [2, 12, 6]]
f = np.stack([d,e])
print(f.shape)
print(f)
print(f[:,2,:])