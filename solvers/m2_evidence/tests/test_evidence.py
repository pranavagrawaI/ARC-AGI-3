"""Evidence-record contracts through actual sandbox processes (stdlib only)."""
from test_basics import SandboxCase, sandbox

ADD = "record_evidence(claim='c{n}',prediction='p',test='t',observed='o',interpretation='i'{extra})"
KW = "claim='c',prediction='p',test='t',observed='o',"


class EvidenceRecord(SandboxCase):
    def test_append_only_revision_and_listing(self):
        store = {}
        self.assertFalse(self.run_code(ADD.format(n=1, extra='') + '\n' + ADD.format(n=2, extra=", scope='game'"), saved_data=store)['error'])
        r = self.run_code(ADD.format(n=3, extra=', revises=1') + "\nprint([e['id'] for e in evidence()], [e['id'] for e in evidence(include_revised=True)], [e['id'] for e in evidence(scope='game')])", saved_data=store)
        self.assertFalse(r['error'], r)
        self.assertEqual(r['stdout'].strip(), '[2, 3] [1, 2, 3] [2]')
        self.assertEqual([e['claim'] for e in store[sandbox.EVIDENCE_KEY]], ['c1', 'c2', 'c3'])

    def test_invalid_entries_change_nothing(self):
        store = {}
        self.run_code(ADD.format(n=1, extra=''), saved_data=store)
        for bad in ["record_evidence("+KW+"interpretation='')", "record_evidence("+KW+"interpretation='i', scope='run')",
                    "record_evidence("+KW+"interpretation='i', revises=9)", "record_evidence("+KW+"interpretation='x'*401)",
                    "record_evidence("+KW+"interpretation='i', revises=True)"]:
            with self.subTest(bad=bad):
                self.assertIn('ValueError', self.run_code(bad, saved_data=store)['error'])
        # the smoke-run failure: scope passed positionally landed in interpretation
        r = self.run_code("record_evidence('c','p','t','o','level')", saved_data=store)
        self.assertIn('keyword arguments only', r['error'])
        self.assertEqual(len(store[sandbox.EVIDENCE_KEY]), 1)

    def test_survives_timeout_and_is_outside_saved_data(self):
        store = {}
        r = self.run_code(ADD.format(n=1, extra='') + '\nwhile True: pass', saved_data=store, timeout=1)
        self.assertTrue(r['timed_out'])
        r = self.run_code("print(list_data()); print(len(evidence()))\nsave_data('__evidence__', 1)", saved_data=store)
        self.assertIn('reserved', r['error'])
        self.assertEqual(r['stdout'].split('\n')[:2], ['{}', '1'])
        r = self.run_code('for i in range(64): save_data(str(i), i)', saved_data=store)
        self.assertFalse(r['error'], r)

    def test_digest_prefers_current_and_bounds_size(self):
        store = {sandbox.EVIDENCE_KEY: []}
        for i in range(1, 41):
            sandbox._evidence_operation(store, {'operation': 'evidence_add', 'entry': {
                'claim': 'c' * 300, 'prediction': 'p', 'test': 't', 'observed': 'o', 'interpretation': 'i'}},
                {'level': 2, 'step': i})
        sandbox._evidence_operation(store, {'operation': 'evidence_add', 'entry': {
            'claim': 'old level', 'prediction': 'p', 'test': 't', 'observed': 'o', 'interpretation': 'i'}}, {'level': 1, 'step': 50})
        digest = sandbox.render_evidence_digest(store, 2)
        self.assertLess(len(digest), 4800)
        self.assertIn('- E40 (level 2)', digest)
        self.assertNotIn('old level', digest)
        self.assertIn('other current entries', digest)
        self.assertIn('old level', sandbox.render_evidence_digest(store, 2, include_previous_level=True) or
                      sandbox.render_evidence_digest({sandbox.EVIDENCE_KEY: store[sandbox.EVIDENCE_KEY][-1:]}, 2, include_previous_level=True))
        self.assertEqual(sandbox.render_evidence_digest({}, 1), '')
