"""A rank's piece of a field: columns padded to one shape on all ranks (ADR 0063)."""

import numpy as np

from gfem._engine.fields import rank_piece


def test_one_process_returns_the_columns_and_never_asks_the_width():
    columns = np.arange(6.0).reshape(3, 2)

    def width():
        raise AssertionError("the width is a collective under ranks only")

    assert rank_piece(columns, width, None) is columns


def test_under_ranks_zero_rows_pad_the_columns_to_the_width():
    columns = np.array([[1 + 2j, 3], [4, 5j]])
    piece = rank_piece(columns, lambda: 4, object())
    want = np.array([[1 + 2j, 3], [4, 5j], [0, 0], [0, 0]])
    assert piece.dtype == complex
    np.testing.assert_array_equal(piece, want)


def test_a_real_piece_keeps_the_dtype_asked():
    piece = rank_piece(np.ones((1, 2)), lambda: 3, object(), dtype=float)
    assert piece.dtype == float
    np.testing.assert_array_equal(piece, [[1, 1], [0, 0], [0, 0]])
