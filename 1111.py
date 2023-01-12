import numpy as np

array=np.eye(10)[5]

array_list1=list(array)
array_list2=array.tolist()

print(type(array_list1[0]))
print(type(array_list2[0]))



