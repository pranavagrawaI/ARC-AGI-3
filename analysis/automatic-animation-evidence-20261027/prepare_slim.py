"""Slim transport for a frozen notebook: large cell bodies ride in a private dataset, the notebook keeps hash-bound stubs.

Why: SaveKernel cancels (HTTP 499) any body that takes more than ~10 s to upload; the 10-30/10-31 uplinks ran 13-50 KB/s
and the 928 KB notebook needs ~100 KB/s. Dataset uploads use resumable blob sessions with no such window.

  prepare_slim.py SRC_NOTEBOOK OUT_DIR KERNEL_ID TITLE DATASET_ID [CELL ...]

Writes OUT_DIR/kernel/{notebook,kernel-metadata.json}, OUT_DIR/dataset/{bodies,dataset-metadata.json}, OUT_DIR/SLIM.json.
Each stub hands the original cell text to the live IPython shell (run_cell + raise_error), so IPython's own input
transformers and magics run exactly as they would on the inline cell. Pre-transforming is wrong: IPython rewrites
whitespace-only lines and the trailing newline of a %%writefile body (176 bytes on the peak patch cell).
reconstruct() inverts the transform; SLIM.json records that it reproduces SRC byte-for-byte in the cell sources.
"""
import hashlib,json,sys
from pathlib import Path

HELPER='''def _frozen_cell(name, sha):
    import hashlib, pathlib
    slug = %r
    paths = [pathlib.Path('/kaggle/input/datasets') / slug / name, pathlib.Path('/kaggle/input') / slug.split('/')[-1] / name]
    hits = [p for p in paths if p.is_file()]
    assert hits, ('frozen cell body not mounted', name, [str(p) for p in paths])
    data = hits[0].read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, ('frozen cell body hash mismatch', name)
    return data
'''
MARK='# Slim transport stub: this cell\'s frozen body is in the attached dataset, bound by SHA-256 (prepare_slim.py).\n'

def sha(b):return hashlib.sha256(b).hexdigest()
def source(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']

def stub(index,body,dataset,first):
 helper=HELPER%dataset if first else ''
 name=f'cell{index:02d}.txt';data=body.encode()
 # Nested run_cell catches exceptions; raise_error re-raises so the notebook stops exactly where the inline cell would.
 code=f"_frozen_result = get_ipython().run_cell(_frozen_cell({name!r}, {sha(data)!r}).decode('utf-8'), store_history=False)\n_frozen_result.raise_error()\n"
 return MARK+helper+code,name,data,{'kind':'run_cell'}

def build(src,out,kernel_id,title,dataset,cells):
 nb=json.loads(Path(src).read_text());out=Path(out);(out/'kernel').mkdir(parents=True);(out/'dataset').mkdir(parents=True)
 records=[]
 for n,i in enumerate(sorted(cells)):
  cell=nb['cells'][i];assert cell['cell_type']=='code',i
  body=source(cell);text,name,data,info=stub(i,body,dataset,n==0)
  (out/'dataset'/name).write_bytes(data);cell['source']=text;cell['outputs']=[];cell['execution_count']=None
  records.append({'cell':i,'file':name,'sha256':sha(data),'bytes':len(data),**info})
 code_file=kernel_id.split('/')[-1]+'.ipynb';(out/'kernel'/code_file).write_text(json.dumps(nb,indent=1))
 meta=json.loads((Path(src).parent/'kernel-metadata.json').read_text())
 meta.update({'id':kernel_id,'title':title,'code_file':code_file});meta['dataset_sources']=meta['dataset_sources']+[dataset]
 (out/'kernel'/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 (out/'dataset'/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':dataset.split('/')[-1],'licenses':[{'name':'other'}]},indent=2)+'\n')
 rebuilt=reconstruct(out/'kernel'/code_file,out/'dataset',records)
 original=[(c['cell_type'],source(c)) for c in json.loads(Path(src).read_text())['cells']]
 slim={'source_notebook':str(src),'source_sha256':sha(Path(src).read_bytes()),'kernel_id':kernel_id,'dataset':dataset,'cells':records,
       'slim_notebook_sha256':sha((out/'kernel'/code_file).read_bytes()),'slim_notebook_bytes':(out/'kernel'/code_file).stat().st_size,
       'metadata_sha256':sha((out/'kernel'/'kernel-metadata.json').read_bytes()),'reconstruction_equal':rebuilt==original}
 (out/'SLIM.json').write_text(json.dumps(slim,indent=2)+'\n')
 if not slim['reconstruction_equal']:raise SystemExit('reconstruction differs from source')
 return slim

def reconstruct(slim_notebook,dataset_dir,records):
 """Cell sources the kernel actually executes, with each stub replaced by the original cell text it stands for."""
 nb=json.loads(Path(slim_notebook).read_text());by={r['cell']:r for r in records}
 cells=[]
 for i,c in enumerate(nb['cells']):
  if i in by:
   r=by[i];data=(Path(dataset_dir)/r['file']).read_bytes();assert sha(data)==r['sha256']
   assert source(c).startswith(MARK)
   cells.append((c['cell_type'],data.decode()))
  else:cells.append((c['cell_type'],source(c)))
 return cells

if __name__=='__main__':
 src,out,kid,title,ds,*cells=sys.argv[1:]
 print(json.dumps(build(src,out,kid,title,ds,[int(c) for c in cells]),indent=1))
