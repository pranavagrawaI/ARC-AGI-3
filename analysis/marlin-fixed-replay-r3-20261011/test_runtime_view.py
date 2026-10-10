import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from runtime_views import prepare_draft_view,EXPERT_TARGET,DENSE_IGNORE


class RuntimeViewTests(unittest.TestCase):
    def test_fresh_view_recreates_config_identity_and_links_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'target';source=root/'source';work=root/'work'
            target.mkdir();source.mkdir();target.joinpath('tokenizer.json').write_text('{}')
            config={'architectures':['Qwen3ForCausalLM'],'quantization_config':{'quant_method':'compressed-tensors','config_groups':{'mtp_routed_experts':{'targets':['RoutedExperts'],'weights':{'num_bits':4,'group_size':32,'symmetric':True}}}}}
            source.joinpath('config.json').write_text(json.dumps(config))
            source.joinpath('model.safetensors.index.json').write_text(json.dumps({'weight_map':{'mtp.layers.0.mlp.experts.0.gate_proj.weight':'shard.safetensors'}}))
            source.joinpath('shard.safetensors').write_bytes(b'fixture-weights')
            expected=json.loads(json.dumps(config));q=expected['quantization_config'];q['config_groups']['mtp_routed_experts']['targets']=[EXPERT_TARGET];q['ignore']=[DENSE_IGNORE]
            identity=json.dumps({'source':str(source.resolve()),'target':str(target.resolve()),'config':expected},sort_keys=True)
            name='draft-view-'+hashlib.sha256(identity.encode()).hexdigest()[:16]
            self.assertFalse((work/name/'config.json').exists())
            actual=prepare_draft_view(target,source,work)
            self.assertEqual(actual,work/name)
            self.assertEqual(json.loads((actual/'config.json').read_text()),expected)
            self.assertTrue((actual/'shard.safetensors').is_symlink())
            self.assertEqual((actual/'shard.safetensors').resolve(),source/'shard.safetensors')
            self.assertEqual(prepare_draft_view(target,source,work),actual)


if __name__=='__main__':unittest.main()
