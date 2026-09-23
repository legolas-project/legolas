import struct
from io import BytesIO

import pytest
from pylbo.utilities.datfiles.istream_reader import (
    DTYPES,
    read_boolean_from_istream,
    read_complex_from_istream,
    read_float_from_istream,
    read_int_from_istream,
    read_mixed_from_istream,
    read_string_from_istream,
)

ENCODE_AS = "utf-8"
KINDS = {
    str: DTYPES["str"],
    int: DTYPES["int"],
    float: DTYPES["float"],
    complex: DTYPES["complex"],
    bool: DTYPES["bool"],
}


def _get_istream_str(stream):
    return BytesIO(stream.encode(ENCODE_AS))


def _get_istream(istream):
    if not isinstance(istream, list):
        istream = [istream]
    kind = KINDS[type(istream[0])]
    return BytesIO(struct.pack(f"{len(istream)}{kind}", *istream))


def test_read_string_from_istream():
    word = "test"
    istream = _get_istream_str(word)
    assert read_string_from_istream(istream, len(word)) == word


def test_read_string_from_istream_with_offset():
    word = "this is a test"
    istream = _get_istream_str(word)
    assert read_string_from_istream(istream, len(word) - 10, offset=10) == word[10:]


def test_read_string_from_istream_with_amount():
    words = ["test1", "test2", "test3"]
    istream = _get_istream_str("".join(words))
    assert read_string_from_istream(istream, len(words[0]), amount=len(words)) == words


def test_read_int_from_istream():
    istream = _get_istream(10)
    assert read_int_from_istream(istream) == 10


def test_read_int_from_istream_with_offset():
    istream = _get_istream([1, 2, 3, 4])
    assert read_int_from_istream(istream, offset=8) == 3


def test_read_int_from_istream_with_amount():
    istream = _get_istream([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert read_int_from_istream(istream, amount=5) == (0, 1, 2, 3, 4)


def test_read_bool_from_istream():
    istream = _get_istream(True)
    assert read_boolean_from_istream(istream) is True
    istream = _get_istream(False)
    assert read_boolean_from_istream(istream) is False


def test_read_bool_from_istream_with_offset():
    istream = _get_istream([1, 0, 1])
    assert read_boolean_from_istream(istream, offset=4) is False
    istream = _get_istream([1, 0, 1])
    assert read_boolean_from_istream(istream, offset=8) is True


def test_read_float_from_istream():
    istream = _get_istream(10.0)
    assert read_float_from_istream(istream) == 10.0


def test_read_float_from_istream_with_offset():
    istream = _get_istream([1.5, 2.5, 3.3, 4.1])
    assert read_float_from_istream(istream, offset=8) == 2.5


def test_read_float_from_istream_with_amount():
    istream = _get_istream([1.2, 4.5, 1.5, 2.5, 3.3, 4.1])
    assert read_float_from_istream(istream, amount=3) == (1.2, 4.5, 1.5)


def test_read_complex_from_istream():
    istream = _get_istream([10.0, 1.0])
    assert read_complex_from_istream(istream) == 10.0 + 1.0j


def test_read_complex_from_istream_with_offset():
    istream = _get_istream([1.5, 1.0, 2.5, 2.0, 3.3, 1.5, 4.1, 0.3])
    assert read_complex_from_istream(istream, offset=32) == 3.3 + 1.5j


def test_read_complex_from_istream_with_amount():
    istream = _get_istream([1.2, 1.0, 4.5, 2.0, 1.5, 1.0, 2.5, 2.0, 3.3, 1.5, 4.1, 0.3])
    assert read_complex_from_istream(istream, amount=3) == (
        1.2 + 1.0j,
        4.5 + 2.0j,
        1.5 + 1.0j,
    )


def test_read_mixed_from_istream():
    values = [1, 5, 10.5, 1.0, 2.0, 4.6]
    istream = BytesIO(struct.pack("iidddd", *values))
    # note that if we write 3 integers, we need to read FOUR integers
    # since byteread starts is done in multiples of 8 bytes
    # writing 2 integers means start reading doubles at byte 8,
    # whereas writing 3 integers means start reading doubles at byte 16 and NOT 12
    assert read_int_from_istream(istream) == 1
    assert read_int_from_istream(istream) == 5
    assert read_complex_from_istream(istream) == 10.5 + 1.0j
    assert read_float_from_istream(istream) == 2.0

    istream = BytesIO(struct.pack("iidddd", *values))
    assert read_mixed_from_istream(istream, fmt="iidddd") == tuple(values)


def test_read_mixed_from_istream_invalid():
    values = [1, 2, 3]
    istream = BytesIO(struct.pack("iii", *values))
    with pytest.raises(ValueError):
        read_mixed_from_istream(istream, fmt="iij")
