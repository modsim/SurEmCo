import pytest

import numpy as np

from .tracker import Tracker


def generate_emitter(x=1.0, y=1.0, start=0, stop=10, dx=0.0, dy=0.0, dark=None):
    if dark is None:
        dark = []
    if not isinstance(dark, list):
        dark = [dark]

    result = []

    for step, n in enumerate(range(start, stop)):
        if n not in dark:
            result.append((x + step * dx, y + step * dy, n))

    return result


def take_n(iter, n=1):
    buffer = tuple()
    for item in iter:
        buffer = buffer + (item,)
        if len(buffer) == n:
            yield buffer
            buffer = tuple()


@pytest.mark.parametrize('data_and_expected', [
    (generate_emitter(),
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (generate_emitter(x=1.0, y=1.0) + generate_emitter(x=5.0, y=5.0),
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    (generate_emitter(x=1.0, y=1.0, dark=5),
     {
         ('memory', 0): [1, 1, 1, 1, 1, 2, 2, 2, 2],
         ('memory', 1): [1, 1, 1, 1, 1, 1, 1, 1, 1],
         ('memory', 2): [1, 1, 1, 1, 1, 1, 1, 1, 1]
     }),
    (generate_emitter(x=1.0, y=1.0, dark=[5, 6]),
     {
         ('memory', 0): [1, 1, 1, 1, 1, 2, 2, 2],
         ('memory', 1): [1, 1, 1, 1, 1, 2, 2, 2],
         ('memory', 2): [1, 1, 1, 1, 1, 1, 1, 1]
     }),
    (generate_emitter(x=1.0, y=1.0, dark=5) + generate_emitter(x=5.0, y=5.0, dark=5),
     {
         ('memory', 0): [1, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4],
         ('memory', 1): [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         ('memory', 2): [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     }),
    (generate_emitter(x=1.0, y=1.0, dark=[5, 6]) + generate_emitter(x=5.0, y=5.0, dark=[5, 6]),
     {
         ('memory', 0): [1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 4],
         ('memory', 1): [1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 4],
         ('memory', 2): [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
     }),
    #(generate_emitter(x=1.0, y=1.0, dx=0.1, dy=0.1),
    # {
    #     ('maximum_displacement', 0.0): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     ('maximum_displacement', 1.0): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # }),
    # (generate_emitter(x=1.0, y=1.0, dx=0.1, dy=0.1) + generate_emitter(x=5.0, y=5.0, dx=0.1, dy=0.1),
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
])
@pytest.mark.parametrize('maximum_displacement', [1.0, 0.0])
@pytest.mark.parametrize('memory', [0, 1, 2])
@pytest.mark.parametrize('mode', [Tracker.Mode.STATIC, Tracker.Mode.MOVING])
@pytest.mark.parametrize('strategy', [Tracker.Strategy.BRUTE_FORCE, Tracker.Strategy.KD_TREE])
@pytest.mark.parametrize('precision', [1.0, 5.0])
def test_tracker(data_and_expected, maximum_displacement, memory, mode, strategy, precision):
    tracker = Tracker()

    data, expected = data_and_expected

    data = np.array(data)

    if isinstance(expected, dict):
        kwargs = dict(maximum_displacement=maximum_displacement, memory=memory, mode=mode, strategy=strategy)
        expected_keys = {key: dict(take_n(key, 2)) for key in expected.keys()}

        def match_dicts(test, possibilites):
            hits = 0
            for test_key, test_value in test.items():
                if test_key in possibilites:
                    if possibilites[test_key] == test_value:
                        hits += 1
            return hits

        matches = {key: match_dicts(dict_key, kwargs) for key, dict_key in expected_keys.items()}

        best_key, best_hits = list(sorted(matches.items(), key=lambda match: match[1], reverse=True))[0]

        assert best_hits == len(best_key)//2, "No expected value found."

        expected = expected[best_key]

    expected = np.array(expected)


    transfer = tracker.empty_track_input_type(len(data))
    transfer['x'] = data[:, 0]
    transfer['y'] = data[:, 1]
    transfer['frame'] = data[:, 2]
    transfer['index'] = range(len(data))
    transfer['precision'] = precision

    tracker.track(transfer,
                  maximum_displacement=maximum_displacement,
                  memory=memory,
                  mode=mode,
                  strategy=strategy)

    del tracker

    assert len(transfer['label']) == len(expected)
    assert (transfer['label'] == expected).all()