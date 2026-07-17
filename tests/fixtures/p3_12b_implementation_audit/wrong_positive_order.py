def _positive(*args):
    return args


def adversary_first(b):
    return b


def adversary_second(b):
    return b


_FUNCTIONS = (adversary_first, adversary_second)
positives = (_positive("second", "fixture", {}), _positive("first", "fixture", {}))
