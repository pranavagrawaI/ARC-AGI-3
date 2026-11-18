"""Allocate owned NOOA prefill IDs before typed events/rendering; never edit strings."""

import ast
import inspect
import textwrap


class CallIDs:
    def __init__(self):
        self.local = set()
        self.provider = set()
        self.next_local = 0

    def allocate(self):
        while True:
            value = f"local_prefill_{self.next_local:08x}"
            self.next_local += 1
            if value not in self.local | self.provider:
                self.local.add(value)
                return value

    def accept_provider(self, ids):
        if len(ids) != len(set(ids)) or set(ids) & (self.local | self.provider):
            raise ValueError("duplicate provider tool-call ID; no renaming allowed")
        self.provider.update(ids)


def install(ids):
    """One process-local, shared-arm hook at the pinned framework's ID birth site.

    Compile the inspected trusted function with ONLY its prefill_id assignment
    changed. No framework file is written, no model text is transformed, and no
    other uuid generation is affected. The rest of the AST is identical.
    """
    from nooa.strategies.codeact import CodeActStrategy

    original = CodeActStrategy._execute_prefill_step
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "prefill_id"
        ):
            expected = (
                ast.parse('prefill_id = f"prefill_{uuid4().hex[:8]}"').body[0].value
            )
            if ast.dump(node.value) != ast.dump(expected):
                raise RuntimeError("NOOA prefill birth-site drift")
            node.value = ast.Call(
                func=ast.Name(id="_owned_prefill_id", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
            changed += 1
    if changed != 1:
        raise RuntimeError("NOOA prefill birth-site count drift")
    namespace = dict(original.__globals__, _owned_prefill_id=ids.allocate)
    exec(
        compile(
            ast.fix_missing_locations(tree), "<trusted-prefill-ID-adapter>", "exec"
        ),
        namespace,
    )
    CodeActStrategy._execute_prefill_step = namespace["_execute_prefill_step"]
