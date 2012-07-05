# Optimized inner loop of load_svmlight_file.
#
# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: Simple BSD.

from libc.string cimport strchr
cimport numpy as np
import numpy as np
import scipy.sparse as sp

from ..utils.arraybuilder import ArrayBuilder


# csr_matrix.indices and .indptr's dtypes are undocumented. We derive them
# empirically.
_temp_csr = sp.csr_matrix(0)
_INDICES_DTYPE = _temp_csr.indices.dtype
_INDPTR_DTYPE = _temp_csr.indptr.dtype
del _temp_csr


cdef bytes COMMA = u','.encode('ascii')
cdef bytes COLON = u':'.encode('ascii')


def _load_svmlight_file(f, dtype, bint multilabel, bint zero_based,
                        long long offset_min, long long offset_max):
    cdef bytes line
    cdef char *hash_ptr, *line_cstr
    cdef Py_ssize_t hash_idx

    data = ArrayBuilder(dtype=dtype)
    indptr = ArrayBuilder(dtype=_INDPTR_DTYPE)
    indices = ArrayBuilder(dtype=_INDICES_DTYPE)
    if multilabel:
        labels = []
    else:
        labels = ArrayBuilder(dtype=np.double)

    if offset_min > 0:
        f.seek(offset_min)
        # drop the previous line that might be truncated and is to be fetched
        # by another call
        f.readline()

    for line in f:
        # skip comments
        line_cstr = line
        hash_ptr = strchr(line_cstr, '#')
        if hash_ptr == NULL:
            hash_idx = -1           # index of '\n' in line
        else:
            hash_idx = hash_ptr - <char *>line
        line = line[:hash_idx]

        line_parts = line.split()
        if len(line_parts) == 0:
            continue

        target, features = line_parts[0], line_parts[1:]
        if multilabel:
            target = [float(y) for y in target.split(COMMA)]
            target.sort()
            labels.append(tuple(target))
        else:
            labels.append(float(target))
        indptr.append(len(data))

        for i in xrange(1, len(line_parts)):
            idx, value = line_parts[i].split(COLON, 1)
            idx = int(idx)
            if idx < 0 or not zero_based and idx == 0:
                raise ValueError(
                        "invalid index %d in SVMlight/LibSVM data file" % idx)
            indices.append(idx)
            data.append(dtype(value))
        if offset_max != -1 and f.tell() >= offset_max:
            # Stop here and let another call deal with the following.
            break

    indptr.append(len(data))

    indptr = indptr.get()
    data = data.get()
    indices = indices.get()

    if not multilabel:
        labels = labels.get()

    return data, indices, indptr, labels
