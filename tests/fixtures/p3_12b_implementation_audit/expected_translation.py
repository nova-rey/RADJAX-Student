def _positive(*args):
    return args


def adversary_first(b):
    return b


def adversary_second(b):
    return b


def _matches_expected():
    return True


_FUNCTIONS = (adversary_first, adversary_second)
positives = (_positive("first", "fixture", {}), _positive("second", "fixture", {}))
