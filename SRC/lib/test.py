from F1 import *
from F2 import *
from F3 import *
from F4 import *
from F5 import *
from F6 import *
from F7 import *
from F8 import *
from F9 import *
from F10 import *
from F11 import *
from F12 import *

def nfunc(x,f):
    d = {0:f1, 1:f2, 2:f3, 3:f4, 4:f5, 5:f6, 6:f7, 7:f6, 8:f7,
         9:f8, 10:f9, 11:f10, 12:f11, 13:f11, 14:f12, 15:f11, 16:f12,
         17:f11, 18:f12, 19:f12}
    return d[f](x)

Dims = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

for f in range(20):
    if Dims[f]>1:
        x = np.ones( (Dims[f]) )
    else:
        x=1
    val = nfunc( x, f)
    print f, val

