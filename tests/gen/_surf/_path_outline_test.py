from gen._surf._path_outline import PathOutline


def test_repr_round_trips():
    p = PathOutline([(0, 1, 'X'), (1, 1 + 1j, 'Z')])
    r = repr(p)
    assert r.startswith('PathOutline(')
    assert eval(r, {'PathOutline': PathOutline, 'frozenset': frozenset}) == p
