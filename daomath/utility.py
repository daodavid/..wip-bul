import numpy as np

def reduce_array(arr,by_step):
    i=0
    result = []
    leng = arr.shape
    p  = leng[0]
    while i < p:
        result.append(arr[i])
        i = i+by_step

    return np.array(result)




