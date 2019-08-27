Wrapping NComp functions using Cython
=====================================
1. Add function signature from NComp to `$GEOCATCOMP/src/geocat/comp/_ncomp/ncomp.pxd` under the 'cdef extern from "ncomp/wrapper.h"' section. *Make sure to include "nogil" at the end of the function signature.*

2. Create a new function in `$GEOCATCOMP/src/geocat/comp/_ncomp/ncomp.pyx`, prepended with an underscore (`_linint2` for example). The Cython function signature acts as the "numpy interface" to NComp. All arguments should be explicitly typed as either np.ndarray or an appropriate C type (int, double, etc).

3. Create ncomp\_array structs: `cdef ncomp.ncomp_array*` for each input np.ndarray, using the `np_to_ncomp_array` function to allocate and populate the ncomp\_array structs.

4. Create output np.ndarray(s) using np.zeros (essentially equivalent to `calloc`ing), create ncomp\_array pointer(s) using `np_to_ncomp_array`.

5. Call C function from "ncomp" namespace, "ncomp.linint2" for example, capturing return value (standardized return codes and error handling still to be determined). Ensure function call is inside "`with nogil:`" block.

6. Return previously created output np.ndarray(s) -- consider returning a tuple if multiple return values are expected.

Wrapping Cython functions in Python
===================================
1. Create a new function in `$GEOCATCOMP/src/geocat/comp/__init__.py` (`linint2` for example).

2. Include `meta=True` as a keyword argument for any function that could potentially retain metadata; retaining metadata will be the default behavior.

3. Check for .chunks attribute on the primary input xarray.DataArray, set chunk sizes to be equal to the shape of the array if not already chunked.

4. Call Dask's `map_blocks` function. The first argument is the actual function name from Cython (`_ncomp._linint2` in this case), followed by positional parameters to the desired Cython function, then followed by keyword arguments for `map_blocks`. The `chunks` keyword argument is particularly important as the total number of chunks on the input array and output array must match in order for Dask to properly align data in the output array.

5. Re-attach metadata as needed.

6. Return output.
