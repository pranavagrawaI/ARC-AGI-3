"""Execute extracted pinned real counting methods; not a processor runtime test."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]/'source-review'

def function(path,name,namespace):
    tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.returns=None
    for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
    node.decorator_list=[]
    code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(code,str(path),'exec'),namespace)
    return namespace[name]

class SourceSemantics(unittest.TestCase):
    def test_control_full_offsets_survive_feature_release(self):
        source=ROOT/'sglang/srt/managers/schedule_batch.py'
        count=function(source,'compute_mm_token_counts',{})
        class Item:
            offsets=[(10,409),(500,899)]
            def is_image(self):return True
            def is_audio(self):return False
            def is_video(self):return False
        request=SimpleNamespace(mm_items=[Item()])
        release=function(source,'release_features',{})
        for cached in (0,400,800):
            request.cached_tokens=cached
            request.mm_items[0].feature=object()
            release(request)
            self.assertIsNone(request.mm_items[0].feature)
            self.assertEqual(count(request),(800,0,0))
    def test_candidate_complete_placeholder_sum_independent_of_cache(self):
        source=ROOT/'vllm/entrypoints/openai/chat_completion/serving.py'
        count=function(source,'_get_mm_token_counts',{'cast':lambda typ,val:val,'MultiModalPlaceholders':dict})
        placeholders=[SimpleNamespace(length=400),SimpleNamespace(length=400)]
        for cached in (0,400,800):
            value=count({'mm_placeholders':{'image':placeholders},'num_cached_tokens':cached})
            self.assertEqual(value,{'image':800})
