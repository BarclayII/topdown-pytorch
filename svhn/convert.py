import h5py
import numpy as np
import sys
import tqdm
import itertools
import numbers

def getnum(obj, f):
    if isinstance(obj, numbers.Number):
        return obj
    return f[obj].value

in_ = sys.argv[1]
out = sys.argv[2]

f = h5py.File(in_)
g = open(out, 'w')
assert len(f['digitStruct/name']) == len(f['digitStruct/bbox'])

for i in tqdm.trange(len(f['digitStruct/name'])):
    name_ref = f['digitStruct/name'][i][0]
    bbox_ref = f['digitStruct/bbox'][i][0]
    name = ''.join(chr(v) for v in f[name_ref].value)
    left_refs, top_refs, width_refs, height_refs, label_refs = [
            f[bbox_ref][k] for k in ['left', 'top', 'width', 'height', 'label']]
    ndigits = len(height_refs)
    bbox_data = [
            [
                np.asscalar(getnum(left_refs[j, 0], f)),
                np.asscalar(getnum(top_refs[j, 0], f)),
                np.asscalar(getnum(width_refs[j, 0], f)),
                np.asscalar(getnum(height_refs[j, 0], f)),
                np.asscalar(getnum(label_refs[j, 0], f)),
                ]
            for j in range(ndigits)
            ]
    print(name, ndigits, *itertools.chain(*bbox_data), file=g)

f.close()
g.close()
