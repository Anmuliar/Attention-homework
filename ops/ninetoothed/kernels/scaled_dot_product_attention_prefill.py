import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()


def arrangement(
    q, k, v, scale, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
    #TODO: 实现kernel
    assert False, "This function is not implemented yet."
    


def application(q, k, v, scale, o):
    #TODO: 实现kernel
    assert False, "This function is not implemented yet."
    

# You can modify the following lines
# shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})
# q, k, v, o = (Tensor(4, shape_options=shape_options) for _ in range(4))
# tensors = (q, k, v, Tensor(0), o)

# kernel = ninetoothed.make(arrangement, application, tensors)
