import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        ma=np.max(z);
        arr=[]
        sum1=0;
        for x in z:
            val=np.exp(x-ma)
            arr.append(val)
            sum1=sum1+val
            
        for  i in range(len(arr)):
            arr[i]=(arr[i]/sum1)
        return np.round(np.array(arr), 4)
