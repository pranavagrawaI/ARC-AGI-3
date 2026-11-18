"""Behavioral CPU regressions: JSON semantic equality does not bind a prompt."""
import copy,json,sys,unittest
from pathlib import Path
from tokenizers import Tokenizer
from reference import digest,check_prepared
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'trial-5-serving-screen/prompt-audit'))
import audit

class WireReference(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  assets=BASE/'trial-3-matched-preparation/processor-assets/control'
  cls.tokenizer=Tokenizer.from_file(str(assets/'tokenizer.json'))
  cls.template=audit.processor_fit.template(assets)
  cls.model=audit.source_models((BASE/'trial-5-serving-screen/prompt-audit/source/protocol.py').read_text())
  cls.catalog=json.loads((BASE/'trial-9-readiness-local/catalog.json').read_text())
  subset=json.loads((BASE/'trial9-cutoff-local-20261023/evidence/partial-response-screen.json').read_text())['rows']
  cls.case=subset[0];cls.record=cls.catalog['records'][cls.case['key']]
  directory=BASE/'trial9-cutoff-local-20261023/evidence/partial-members/members/serving-wire-capacity9/output/control'/cls.case['rid']
  cls.actual_raw=(directory/'wire.http').read_bytes().split(b'\r\n\r\n',1)[1]
  cls.actual=json.loads(cls.actual_raw);cls.source=audit.corpus.payload_for(cls.record)
 def render(self,body):
  text=self.template.render(messages=audit.processor_fit.messages_for_template(body['messages']),tools=audit.normalize_tools(body['tools'],self.model),add_generation_prompt=True,**body['chat_template_kwargs'])
  return text,self.tokenizer.encode(text,add_special_tokens=False).ids
 def test_only_parameters_order_changes_render_ids_despite_semantic_json_equality(self):
  original,ids=self.render(self.actual)
  changed=copy.deepcopy(self.actual)
  changed['tools'][0]['function']['parameters']=self.source['tools'][0]['function']['parameters']
  restored,restored_ids=self.render(changed)
  self.assertEqual(self.actual,changed)
  self.assertEqual(audit.corpus.wire_hash(self.actual),audit.corpus.wire_hash(changed))
  self.assertNotEqual(original,restored);self.assertNotEqual(ids,restored_ids)
  self.assertEqual(digest(restored.encode()),self.record['render_sha'])
  self.assertEqual(len(restored_ids)+399*len(self.record['images']),self.record['local_prompt'])
 def test_exact_body_hash_rejects_semantically_equal_order_change(self):
  record={'prepared_body_sha256':digest(self.actual_raw)}
  self.assertEqual(check_prepared(self.actual_raw,record),self.actual)
  changed=json.dumps(self.source,ensure_ascii=False,separators=(',',':')).encode()
  self.assertEqual(json.loads(changed),self.actual)
  self.assertNotEqual(digest(changed),record['prepared_body_sha256'])
  with self.assertRaisesRegex(ValueError,'bytes/order mismatch'):check_prepared(changed,record)
 def test_derived_reference_keeps_raw_body_not_recanonicalized_hash(self):
  record={'prepared_body_sha256':digest(self.actual_raw)}
  changed=self.actual_raw+b'\n'
  self.assertEqual(json.loads(changed),self.actual)
  with self.assertRaisesRegex(ValueError,'bytes/order mismatch'):check_prepared(changed,record)

if __name__=='__main__':unittest.main()
