from __future__ import annotations


def test_from_generator_exported_from_package_root():
    from zpyflow import from_generator

    assert from_generator(x for x in [1, 2, 3]).to_list() == [1, 2, 3]
