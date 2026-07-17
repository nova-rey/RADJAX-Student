def _positive(*args):
    return args


def adversary_first(b):
    return b


_FUNCTIONS = (adversary_first,)
positives = (_positive("first", "fixture", {}), _positive("second", "fixture", {}))
