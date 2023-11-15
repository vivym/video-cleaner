def to_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise TypeError(f"Expected int or tuple[int, int], got {type(x)}")
