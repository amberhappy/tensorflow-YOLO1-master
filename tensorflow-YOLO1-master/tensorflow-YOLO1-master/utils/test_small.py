import numpy as np

# self.offset=np.transpose(np.reshape(np.array([np.arange(self.cell_size)]*self.cell_size*self.box_per_cell),
#                                            (self.box_per_cell,self.cell_size,self.cell_size)),(1,2,0))
print(np.arange(7))
b = np.array([np.arange(7)]*7*2)
print(b.shape)
c = np.reshape(b,(2,7,7))
print("c",c)
d = np.transpose(c,(1,2,0))
print("d",d)
print(d.shape)

bob1 = 7*7*20
bob2 = bob1 + 7*7*2
print(bob2)