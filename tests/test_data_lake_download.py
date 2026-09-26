"""Tests for fetching data lake files - biomni.utils.check_and_download_s3_files.

The data lake counts a file that exists as downloaded, so what matters is that
a file only ever exists complete: a download cut short must leave nothing at
the target, or every later query reads the truncated dataset and none fetches
it again.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
import requests
from biomni import utils

BUCKET = "https://bucket.example"
NAME = "dataset.tsv"
PAYLOAD = [b"gene\tscore\n", b"APOE\t9.1\n", b"TREM2\t7.4\n"]


class FakeResponse:
    """The slice of ``requests.Response`` the downloader uses."""

    def __init__(self, chunks, *, content_length=None, fail_at=None, error=None, status=200):
        self.chunks = chunks
        size = sum(len(c) for c in chunks) if content_length is None else content_length
        self.headers = {"content-length": str(size)}
        self.fail_at = fail_at
        self.error = error
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Client Error")

    def iter_content(self, chunk_size):
        for index, chunk in enumerate(self.chunks):
            if index == self.fail_at:
                raise self.error
            yield chunk


def serve(monkeypatch, response_for):
    """Route the downloader's requests to ``response_for(url)``, recording each url."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        assert kwargs.get("timeout"), "a download without a timeout can hang the chat turn forever"
        return response_for(url)

    monkeypatch.setattr(utils.requests, "get", fake_get)
    return calls


def fetch(lake):
    return utils.check_and_download_s3_files(BUCKET, str(lake), [NAME], folder="data_lake")


def test_a_completed_download_is_renamed_into_place(tmp_path, monkeypatch):
    calls = serve(monkeypatch, lambda url: FakeResponse(PAYLOAD))

    assert fetch(tmp_path) == {NAME: True}
    assert calls == [f"{BUCKET}/data_lake/{NAME}"]
    assert (tmp_path / NAME).read_bytes() == b"".join(PAYLOAD)
    assert os.listdir(tmp_path) == [NAME]


def test_a_connection_closed_early_leaves_nothing_behind(tmp_path, monkeypatch):
    serve(monkeypatch, lambda url: FakeResponse(PAYLOAD[:1], content_length=1000))

    assert fetch(tmp_path) == {NAME: False}
    assert os.listdir(tmp_path) == []


def test_an_error_mid_download_leaves_nothing_behind(tmp_path, monkeypatch):
    serve(monkeypatch, lambda url: FakeResponse(PAYLOAD, fail_at=2, error=requests.ConnectionError("reset")))

    assert fetch(tmp_path) == {NAME: False}
    assert os.listdir(tmp_path) == []


def test_an_http_error_leaves_nothing_behind(tmp_path, monkeypatch):
    serve(monkeypatch, lambda url: FakeResponse([b"<Error>NoSuchKey</Error>"], status=404))

    assert fetch(tmp_path) == {NAME: False}
    assert os.listdir(tmp_path) == []


def test_an_interrupted_download_leaves_nothing_behind(tmp_path, monkeypatch):
    # KeyboardInterrupt is not an Exception: the old in-place download left its
    # partial file at the target, where it passed for the finished dataset.
    serve(monkeypatch, lambda url: FakeResponse(PAYLOAD, fail_at=1, error=KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        fetch(tmp_path)
    assert os.listdir(tmp_path) == []


def test_the_file_does_not_exist_until_every_byte_has_arrived(tmp_path, monkeypatch):
    # A process killed at any point - a pod evicted, the OOM killer - runs no
    # cleanup at all, so the target must simply not exist before the end.
    target = tmp_path / NAME
    seen = []

    class WatchedResponse(FakeResponse):
        def iter_content(self, chunk_size):
            for chunk in super().iter_content(chunk_size):
                seen.append(target.exists())
                yield chunk

    serve(monkeypatch, lambda url: WatchedResponse(PAYLOAD))

    assert fetch(tmp_path) == {NAME: True}
    assert seen == [False] * len(PAYLOAD)
    assert target.read_bytes() == b"".join(PAYLOAD)


def test_what_a_killed_download_leaves_does_not_stop_the_next_fetch(tmp_path, monkeypatch):
    # What a SIGKILL leaves: a partial file, which the next fetch must ignore.
    (tmp_path / f".{NAME}.abc123.part").write_bytes(PAYLOAD[0])
    calls = serve(monkeypatch, lambda url: FakeResponse(PAYLOAD))

    assert fetch(tmp_path) == {NAME: True}
    assert len(calls) == 1
    assert (tmp_path / NAME).read_bytes() == b"".join(PAYLOAD)


def test_abandoned_partial_downloads_are_swept_and_live_ones_kept(tmp_path, monkeypatch):
    abandoned = tmp_path / f".{NAME}.old.part"
    live = tmp_path / f".{NAME}.new.part"
    other_file = tmp_path / ".other.tsv.old.part"
    for partial in (abandoned, live, other_file):
        partial.write_bytes(b"x")
    long_ago = time.time() - utils._STALE_PARTIAL_S - 60
    os.utime(abandoned, (long_ago, long_ago))
    os.utime(other_file, (long_ago, long_ago))
    serve(monkeypatch, lambda url: FakeResponse(PAYLOAD))

    assert fetch(tmp_path) == {NAME: True}
    assert not abandoned.exists()
    assert live.exists()  # another process may still be writing it
    assert other_file.exists()  # swept when that file is next fetched


def test_a_file_already_present_is_not_fetched(tmp_path, monkeypatch):
    (tmp_path / NAME).write_bytes(b"local copy")
    calls = serve(monkeypatch, lambda url: FakeResponse(PAYLOAD))

    assert fetch(tmp_path) == {NAME: True}
    assert calls == []
    assert (tmp_path / NAME).read_bytes() == b"local copy"


def test_two_chats_needing_the_same_file_download_it_once(tmp_path, monkeypatch):
    release = threading.Event()
    first_started = threading.Event()

    class SlowResponse(FakeResponse):
        def iter_content(self, chunk_size):
            first_started.set()
            assert release.wait(5)
            yield from super().iter_content(chunk_size)

    calls = serve(monkeypatch, lambda url: SlowResponse(PAYLOAD))
    results = {}
    seen = {}

    def chat(key):
        results[key] = fetch(tmp_path)
        # What the chat's code step would read the moment the fetch returns.
        seen[key] = (tmp_path / NAME).read_bytes() if (tmp_path / NAME).exists() else None

    first = threading.Thread(target=chat, args=("first",))
    first.start()
    assert first_started.wait(5)
    second = threading.Thread(target=chat, args=("second",))
    second.start()
    time.sleep(0.2)  # the second is now waiting for the first's download
    release.set()
    first.join(5)
    second.join(5)

    assert results == {"first": {NAME: True}, "second": {NAME: True}}
    assert seen == {"first": b"".join(PAYLOAD), "second": b"".join(PAYLOAD)}
    assert len(calls) == 1
