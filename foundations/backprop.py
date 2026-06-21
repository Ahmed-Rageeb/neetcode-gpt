import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        y_pred=np.matmul(w,x)+b
        y_hat=1/(1+np.exp(-y_pred))
        dl_dw=(y_hat-y_true)*y_hat*(1.0-y_hat)
        dw=np.round(dl_dw*x,5)
        db=np.round(dl_dw,5)
        return(dw,db)
