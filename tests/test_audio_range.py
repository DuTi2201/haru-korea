import unittest
from types import SimpleNamespace

from starlette.datastructures import Headers

from app.api.routers.audio import _audio_response

DATA = bytes(range(100))


def req(range_header=None):
    return SimpleNamespace(headers=Headers({"range": range_header} if range_header else {}))


class RangeTests(unittest.TestCase):
    def test_full_body_advertises_ranges(self):
        r = _audio_response(req(), DATA, "audio/aac")
        self.assertEqual((r.status_code, r.body), (200, DATA))
        self.assertEqual(r.headers["accept-ranges"], "bytes")

    def test_probe_first_two_bytes(self):  # what Safari sends first
        r = _audio_response(req("bytes=0-1"), DATA, "audio/aac")
        self.assertEqual((r.status_code, r.body), (206, DATA[0:2]))
        self.assertEqual(r.headers["content-range"], "bytes 0-1/100")

    def test_open_ended_and_suffix_and_clamped(self):
        self.assertEqual(_audio_response(req("bytes=90-"), DATA, "a/b").body, DATA[90:])
        self.assertEqual(_audio_response(req("bytes=-10"), DATA, "a/b").body, DATA[-10:])
        r = _audio_response(req("bytes=50-9999"), DATA, "a/b")
        self.assertEqual((r.status_code, r.body, r.headers["content-range"]), (206, DATA[50:], "bytes 50-99/100"))

    def test_unsatisfiable(self):
        r = _audio_response(req("bytes=100-"), DATA, "a/b")
        self.assertEqual(r.status_code, 416)
        self.assertEqual(r.headers["content-range"], "bytes */100")

    def test_garbage_range_falls_back_to_full(self):
        for h in ("bytes=a-b", "items=0-5", "bytes=-"):
            self.assertEqual(_audio_response(req(h), DATA, "a/b").status_code, 200, h)


if __name__ == "__main__":
    unittest.main()
