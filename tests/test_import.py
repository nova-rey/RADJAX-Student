def test_package_imports() -> None:
    import radjax_student

    assert radjax_student.__all__ == []
    assert not hasattr(radjax_student, "TinyDebugStudentBackend")
