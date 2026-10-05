"""Behavioral bounds and honest fallback at Builder phase transitions."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from qualify import HERE
from hybrid.builder import ALLOCATION, Allocation, HybridBuilderStrategy, PhaseRuntime
from hybrid.accounting import final_usage
from tycho.agent.builder import REPORT_FILE


class LifecycleRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        ws = SimpleNamespace(dir=self.root, write_file=self.write)
        self.builder = SimpleNamespace(_executor=SimpleNamespace(ws=ws), _report_before=None,
            _world_model_before=None, _tool_surface=SimpleNamespace(_trace_path=self.root/'tools.jsonl'))
        self.events = []
        self.runtime = SimpleNamespace(event_manager=SimpleNamespace(add=self.events.append))

    def write(self, name, content):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)

    def tearDown(self):
        self.directory.cleanup()

    def test_phase_expiry_stops_calls_and_bounds_native_timeout(self):
        calls = []
        async def generate(**kwargs):
            calls.append(kwargs); return 'response', 'event'
        self.runtime.generate = generate
        async def run():
            loop = asyncio.get_running_loop()
            bounded = PhaseRuntime(self.runtime, self.builder, 'construction', loop.time()+10, construction=True)
            for _ in range(12): await bounded.generate(tools=[])
            with self.assertRaises(TimeoutError): await bounded.generate(tools=[])
            self.assertEqual(len(calls), 12)
            self.assertTrue(all(0 < c['timeout'] <= 10 for c in calls))
            self.assertTrue(all(c['tool_choice']=='required' for c in calls[4:]))
        asyncio.run(run())

    def test_cancel_propagates_and_slow_generation_is_interrupted(self):
        async def generate(**kwargs): await asyncio.sleep(10)
        self.runtime.generate = generate
        async def run():
            bounded = PhaseRuntime(self.runtime, self.builder, 'report', asyncio.get_running_loop().time()+.02)
            with self.assertRaises(TimeoutError): await bounded.generate()
            bounded = PhaseRuntime(self.runtime, self.builder, 'report', asyncio.get_running_loop().time()+10)
            task = asyncio.create_task(bounded.generate())
            await asyncio.sleep(0); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            stats, receipt = final_usage(self.root, {})
            self.assertEqual(stats, {})
            self.assertEqual(receipt['interrupted_builder_calls_unknown_usage'], 2)
        asyncio.run(run())

    def test_exhausted_finish_writes_honest_notes_and_report(self):
        async def run():
            token = ALLOCATION.set(Allocation(asyncio.get_running_loop().time()-1, 0))
            try:
                strategy = HybridBuilderStrategy()
                checkpoint = SimpleNamespace(required=True, resolved=False)
                config = SimpleNamespace(max_tokens=100, temperature=.7)
                forced = await strategy._enforce_world_model_checkpoint(self.runtime, self.builder,
                    None, {}, [], config)
                result = await strategy._finish_report(self.runtime, self.builder, [], config, forced,
                    checkpoint, objective_review=False)
                self.assertTrue(forced)
                self.assertIn('confidence: none', result[0])
                self.assertFalse(result[-1])
                self.assertEqual((self.root/REPORT_FILE).read_text(), result[0])
                self.assertTrue((self.root/'notes/world_model.md').exists())
                events = [json.loads(line) for line in (self.root/'builder-lifecycle.jsonl').read_text().splitlines()]
                self.assertEqual([r['phase'] for r in events], ['checkpoint_timeout', 'report_timeout'])
                self.assertEqual(final_usage(self.root, {})[1]['interrupted_builder_calls_unknown_usage'], 0)
            finally: ALLOCATION.reset(token)
        asyncio.run(run())


if __name__=='__main__': unittest.main()
