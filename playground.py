import numpy as np

def scale_array(x, new_max):
    max_element = np.abs(x).max()
    scale_by = new_max / max_element
    return x * scale_by

var = np.array([-4, -5, 1, 2, 3])

var2 = scale_array(var, 1)

print(var2)
