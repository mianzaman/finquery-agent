"""Tests for the in-memory rate limiter, with the clock passed in."""

from app.rate_limit import RateLimiter


def test_requests_inside_the_limit_pass_and_the_next_is_told_how_long_to_wait() -> None:
    limiter = RateLimiter(limit=3, window=60)

    assert [limiter.check("a", now) for now in (0.0, 10.0, 20.0)] == [None, None, None]
    assert limiter.check("a", 25.0) == 35  # the first request leaves the window at 60 s
    assert limiter.check("a", 59.5) == 1  # never 0: a client told to wait waits at least 1 s


def test_the_window_slides() -> None:
    limiter = RateLimiter(limit=2, window=60)
    assert [limiter.check("a", now) for now in (0.0, 30.0)] == [None, None]

    assert limiter.check("a", 59.0) == 1
    assert limiter.check("a", 60.0) is None  # the request from 0 s has left
    assert limiter.check("a", 61.0) == 29  # the one from 30 s has not


def test_a_refused_request_is_not_counted() -> None:
    limiter = RateLimiter(limit=1, window=60)
    assert limiter.check("a", 0.0) is None

    assert [limiter.check("a", now) for now in (1.0, 2.0, 59.0)] == [59, 58, 1]
    assert limiter.check("a", 60.0) is None  # hammering while limited did not extend the wait


def test_clients_are_counted_separately_and_idle_ones_are_forgotten() -> None:
    limiter = RateLimiter(limit=1, window=60)

    assert limiter.check("a", 0.0) is None
    assert limiter.check("b", 1.0) is None
    assert limiter.check("a", 2.0) == 58
    assert set(limiter._hits) == {"a", "b"}
    assert limiter.check("c", 100.0) is None
    assert set(limiter._hits) == {"c"}


def test_idle_clients_are_swept_once_per_window_not_on_every_request() -> None:
    limiter = RateLimiter(limit=1, window=60)
    for index in range(5):
        assert limiter.check(f"client{index}", float(index)) is None

    assert limiter.check("z", 62.0) is None  # the sweep at 62 s drops clients 0 to 2
    assert set(limiter._hits) == {"client3", "client4", "z"}
    assert limiter.check("y", 70.0) is None  # no sweep yet: the last one was 8 s ago
    assert set(limiter._hits) == {"client3", "client4", "z", "y"}


def test_beyond_the_key_limit_new_clients_share_one_bucket() -> None:
    limiter = RateLimiter(limit=2, window=60, max_keys=3)
    for key in ("a", "b", "c"):
        assert limiter.check(key, 0.0) is None

    assert [limiter.check(key, 1.0) for key in ("d", "e", "f")] == [None, None, 60]
    assert len(limiter._hits) == 4  # a, b, c and the shared bucket: it cannot grow further
    assert limiter.check("a", 2.0) is None  # known clients keep their own count
    assert limiter.check("g", 61.0) is None  # the old clients have gone, so there is room
    assert "g" in limiter._hits
