from dataclasses import dataclass

from radjax_student.learning.hooks import (
    HookContext,
    HookPolicy,
    HookResult,
    dispatch_hooks,
)


@dataclass(frozen=True)
class Hook:
    hook_id: str
    priority: int
    fail: bool = False

    def on_event(self, context):
        return HookResult(status="fail" if self.fail else "pass")


def test_hook_order_and_failure_policies_are_deterministic():
    context = HookContext("r", "loop_start", 1, 0)
    hooks = (Hook("b", 1), Hook("a", 1), Hook("bad", 0, True))
    fail = dispatch_hooks(hooks, HookPolicy(), context)
    assert fail.blockers and fail.receipts[0].hook_id == "bad"
    disabled = dispatch_hooks(hooks, HookPolicy("disable_hook"), context)
    assert disabled.disabled_hook_ids == ("bad",)
    continued = dispatch_hooks(hooks, HookPolicy("warn_and_continue"), context)
    assert [r.hook_id for r in continued.receipts] == ["bad", "a", "b"]
