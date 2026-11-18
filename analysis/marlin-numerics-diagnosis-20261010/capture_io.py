"""Read this capture's restricted, uncompressed torch.save tensor archives.

No PyTorch/GPU dependency; unsupported globals, compression, or endian fail.
Only used with the hash-verified local experiment artifacts.
"""
import collections
import io
import pickle
import struct
import zipfile
import numpy as np


def load(path):
    with zipfile.ZipFile(path) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = metadata.rsplit('/', 1)[0]
        assert archive.read(prefix+'/byteorder') == b'little'

        def rebuild(storage, offset, size, stride, requires_grad, hooks, *extra):
            array, kind = storage
            view = np.lib.stride_tricks.as_strided(array[offset:], shape=size,
                         strides=tuple(s*array.itemsize for s in stride), writeable=False)
            if kind == 'BFloat16Storage':
                view = (view.astype(np.uint32) << 16).view(np.float32)
            return view

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
                    return rebuild
                if (module, name) == ('collections', 'OrderedDict'):
                    return collections.OrderedDict
                if module == 'torch' and name in ['IntStorage', 'FloatStorage', 'BFloat16Storage']:
                    return name
                raise ValueError('Unsupported archive global: '+module+'.'+name)

            def persistent_load(self, identifier):
                category, kind, key, device, count = identifier
                assert category == 'storage' and device == 'cpu'
                member = archive.getinfo(prefix+'/data/'+key)
                assert member.compress_type == zipfile.ZIP_STORED
                with path.open('rb') as f:
                    f.seek(member.header_offset)
                    header = f.read(30)
                    name_len, extra_len = struct.unpack_from('<HH', header, 26)
                offset = member.header_offset+30+name_len+extra_len
                dtype = {'IntStorage':'<i4', 'FloatStorage':'<f4', 'BFloat16Storage':'<u2'}[kind]
                assert member.file_size == count*np.dtype(dtype).itemsize
                return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,)), kind

        return Reader(io.BytesIO(archive.read(metadata))).load()
