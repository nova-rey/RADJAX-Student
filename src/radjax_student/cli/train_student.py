"""Deprecated tiny smoke shim; this is not the RADJAX product train command."""

from __future__ import annotations

from radjax_student.training import run_tiny_train_step


def main() -> int:
    result = run_tiny_train_step()
    print(f"initial_loss={result.initial_loss}")
    print(f"final_loss={result.final_loss}")
    print(f"parameters_changed={str(result.parameters_changed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
