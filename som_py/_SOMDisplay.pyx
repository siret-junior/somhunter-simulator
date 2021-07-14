""" Cython wrapping for SOM display """

# cimport the Cython declarations for numpy
cimport numpy as np
from cython cimport floating
import numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "SOMDisplay.h":
    void som_display (float * points, 
                    float * scores, 
                    size_t n, 
                    size_t dim, 
                    size_t swidth, 
                    size_t sheight, 
                    size_t * output, 
                    size_t seed)

# create the wrapper code, with numpy type annotations
def create_som_display(np.ndarray[float, ndim=2, mode="c"] points not None,
                     np.ndarray[float, ndim=1, mode="c"] scores not None,
                     size_t swidth, size_t sheight, size_t seed = 42):
    output = np.zeros(swidth * sheight, dtype=np.uint64)
    som_display(<float*> np.PyArray_DATA(points),
                <float*> np.PyArray_DATA(scores),
                points.shape[0],
                points.shape[1],
                swidth,
                sheight,
                <size_t*> np.PyArray_DATA(output),
                seed)
    return output