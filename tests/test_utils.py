from tfrecord.utils import chunks


def test_chunks():
    ret = chunks(list(range(11)), 2)
    ret_list = list(ret)
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]
    assert ret_list == expected

    ret = chunks(list(range(5)), 100)
    ret_list = list(ret)
    expected = [[0, 1, 2, 3, 4]]
    assert ret_list == expected
