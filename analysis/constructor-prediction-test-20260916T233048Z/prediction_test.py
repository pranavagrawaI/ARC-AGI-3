"""Offline commands only. No auth loader or paid endpoint implementation."""
import argparse,json
from pathlib import Path
from pipeline import construct,freeze_artifact,predict_once,score_bound
from fixtures import Server,IDENTITY,replies

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
 for mode in ('fixture-construct','freeze-artifact','predict','score'):
  q=sub.add_parser(mode);q.add_argument('--prefix',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
  if mode=='freeze-artifact':q.add_argument('--artifact',type=Path,required=True)
  if mode in ('predict','score'):
   q.add_argument('--frozen',type=Path,required=True);q.add_argument('--query',type=Path,required=True)
  if mode=='score':
   q.add_argument('--outcome',type=Path,required=True);q.add_argument('--binding-sha256',required=True)
 a=p.parse_args()
 if a.mode=='fixture-construct':
  with Server(replies(IDENTITY)) as server:r=construct(a.prefix,a.output,server.url)
  print(json.dumps({'fixture_only':True,'status':r.status,'frozen':str(a.output/'FROZEN.json')}))
 elif a.mode=='freeze-artifact':
  freeze_artifact(a.artifact,a.prefix,a.output);print(a.output/'FROZEN.json')
 elif a.mode=='predict':
  sha=predict_once(a.frozen,a.prefix,a.query,a.output)
  print(json.dumps({'binding_sha256':sha,'prediction':str(a.output/'PREDICTION.json'),'outcome_not_read':True}))
 else:print(json.dumps(score_bound(a.output,a.frozen,a.prefix,a.query,a.outcome,a.binding_sha256),indent=2))
if __name__=='__main__':main()
