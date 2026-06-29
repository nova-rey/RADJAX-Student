def test_package_imports() -> None:
    import radjax_student

    assert radjax_student.TinyDebugStudentBackend().architecture_id == "tiny_debug"
