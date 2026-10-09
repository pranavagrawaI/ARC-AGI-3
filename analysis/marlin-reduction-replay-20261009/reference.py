"""Independent inverse of W4 Marlin's 16x64 repack; no kernel output as oracle."""
import math
import numpy as np


def unpack_marlin(packed, k, n):
    if k % 16 or n % 64 or np.asarray(packed).size != k*n//8:
        raise ValueError('Unsupported packed shape')
    words = np.asarray(packed, dtype=np.int32).view(np.uint32).reshape(k//16, n//64, 128)
    nibble = ((words[..., None] >> (np.arange(8, dtype=np.uint32)*4)) & 15).reshape(k//16, n//64, 1024)
    mapping = np.empty((16,64), dtype=np.int64)
    order = [0,2,4,6,1,3,5,7]
    for lane in range(32):
        for warp in range(4):
            for pos, val in enumerate(order):
                row = 2*(lane % 4) + [0,1,8,9][val % 4]
                col = 16*warp + lane//4 + (8 if val >= 4 else 0)
                mapping[row,col] = (lane*4+warp)*8 + pos
    tiles = nibble[..., mapping.ravel()].reshape(k//16,n//64,16,64)
    return tiles.transpose(0,2,1,3).reshape(k,n).astype(np.int16)-8


def unpermute_scales(scales, k):
    scales = np.asarray(scales, dtype=np.float32)
    groups, n = scales.shape
    perm = [i+8*j for i in range(8) for j in range(8)] if groups > 1 else [2*i+j for i in range(4) for j in [0,1,8,9,16,17,24,25]]
    if k % groups or scales.size % len(perm):
        raise ValueError('Unsupported scale grouping')
    return scales.reshape(-1,len(perm))[:,np.argsort(perm)].reshape(groups,n)


def dequantize(packed, scales, k, n):
    q = unpack_marlin(packed,k,n).astype(np.float32)
    s = unpermute_scales(scales,k)
    return q * np.repeat(s, k//s.shape[0], axis=0)


def errors(output, reference):
    actual = np.asarray(output,dtype=np.float64)
    expected = np.asarray(reference,dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError('Output/reference shape mismatch')
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        return dict(finite=False,max_abs=None,rms=None,reference_rms=None,nrms=None,nmax=None)
    delta = actual-expected
    rms_ref = float(np.sqrt(np.mean(expected**2)))
    return {'finite': bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
            'max_abs': float(np.max(np.abs(delta))), 'rms': float(np.sqrt(np.mean(delta**2))),
            'reference_rms': rms_ref, 'nrms': float(np.sqrt(np.mean(delta**2)))/max(rms_ref,1e-12),
            'nmax': float(np.max(np.abs(delta)))/max(rms_ref,1e-12)}


def numerical_gate(base, treatment, drift):
    return (all(x['finite'] for x in [base,treatment,drift])
            and max(base['nrms'],treatment['nrms']) <= .03
            and max(base['nmax'],treatment['nmax']) <= .25
            and treatment['nrms'] <= 1.10*base['nrms']+.0001
            and treatment['nmax'] <= 1.10*base['nmax']+.001
            and drift['nrms'] <= .005 and drift['nmax'] <= .05)


def paired_summary(pairs):
    arr = np.asarray(pairs,dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError('Expected positive finite paired baseline/treatment milliseconds')
    logs = np.log(arr[:,0]/arr[:,1])
    rng = np.random.default_rng(20261009)
    bootstrap = logs[rng.integers(0,len(logs),size=(10000,len(logs)))].mean(axis=1)
    lo, hi = np.exp(np.quantile(bootstrap,[.025,.975]))
    return {'pairs': len(arr), 'geometric_speedup': math.exp(float(logs.mean())),
            'bootstrap_95': [float(lo),float(hi)], 'faster_pairs': int((logs>0).sum()),
            'median_speedup': float(np.median(arr[:,0]/arr[:,1]))}


def qualify_outputs(baseline, treatment, poisoned, fp32):
    base=errors(baseline,fp32);candidate=errors(treatment,fp32)
    drift=errors(treatment,baseline);poison=errors(poisoned,fp32)
    agreement=errors(poisoned,treatment)
    scale=max(base['reference_rms'] or 0,1e-12)
    if drift['finite']:drift.update(nrms=drift['rms']/scale,nmax=drift['max_abs']/scale)
    passed=(numerical_gate(base,candidate,drift) and poison['finite']
            and poison['nrms']<=1.10*base['nrms']+.0001 and poison['nmax']<=1.10*base['nmax']+.001
            and agreement['finite'] and agreement['nrms']<=.001 and agreement['nmax']<=.01)
    return {'baseline':base,'treatment':candidate,'drift':drift,'poisoned_scratch':poison,'poison_agreement':agreement,'passed':passed}
