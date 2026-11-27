"""ShuffledStream: streaming shuffle by file order + document buffer, without datasets' shuffle."""

import pytest

pytest.importorskip("datasets")

from brickformer.data import ByteTokenizer  # noqa: E402
from brickformer.data.datasets import PretrainDataset, ShuffledStream, load_stream  # noqa: E402


@pytest.fixture
def files(tmp_path):
    """Four text files of 50 numbered lines each: line ``f-i`` is document i of file f."""
    out = []
    for f in range(4):
        p = tmp_path / f"part{f}.txt"
        p.write_text("".join(f"{f}-{i}\n" for i in range(50)))
        out.append(str(p))
    return out


def _docs(stream):
    return [ex["text"] for ex in stream]


def test_is_a_permutation_reproducible_and_reshuffled_per_epoch(files):
    s = ShuffledStream("text", data_files=files, buffer_size=16, seed=1)
    first = _docs(s)
    assert sorted(first) == sorted(f"{f}-{i}" for f in range(4) for i in range(50))  # nothing lost or doubled
    assert first != sorted(first, key=lambda d: tuple(map(int, d.split("-"))))       # actually shuffled
    assert _docs(ShuffledStream("text", data_files=files, buffer_size=16, seed=1)) == first
    s.set_epoch(1)
    second = _docs(s)
    assert sorted(second) == sorted(first) and second != first


def test_file_order_is_shuffled(files):
    orders = set()
    for seed in range(8):
        docs = _docs(ShuffledStream("text", data_files=files, buffer_size=1, seed=seed))  # buffer 1 = in order
        orders.add(tuple(dict.fromkeys(d.split("-")[0] for d in docs)))
    assert len(orders) > 1


def test_does_not_use_datasets_shuffle(files, monkeypatch):
    import datasets

    def refuse(*a, **k):
        raise AssertionError("IterableDataset.shuffle holds GBs of RAM on many-file parquet streams")

    monkeypatch.setattr(datasets.IterableDataset, "shuffle", refuse)
    assert len(_docs(load_stream("text", data_files=files, shuffle_buffer=8))) == 200
    ds = PretrainDataset("text", ByteTokenizer(), seq_len=16, data_files=files, streaming=True, shuffle_buffer=8)
    assert all(len(ex["input_ids"]) == 16 for _, ex in zip(range(20), ds))


def test_without_buffer_streams_in_order(files):
    assert _docs(load_stream("text", data_files=files))[:3] == ["0-0", "0-1", "0-2"]
    with pytest.raises(ValueError):
        ShuffledStream("text", data_files=files, buffer_size=0)
