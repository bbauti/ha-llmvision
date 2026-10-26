"""Tests for the shared Timeline keeping its in-memory events in sync with the DB."""

import asyncio
import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.llmvision.timeline import (
    TIMELINE_KEY,
    Event,
    Timeline,
    get_timeline,
)


def _make_hass(tmp_path) -> Mock:
    hass = Mock()
    hass.data = {}
    hass.config.path = lambda *parts: str(tmp_path.joinpath(*parts))
    # The constructor schedules DB setup, reload and cleanup; tests run them explicitly
    hass.async_create_task = lambda coro: coro.close()

    async def run_in_executor(func, *args):
        return func(*args)

    hass.async_add_executor_job = run_in_executor
    return hass


def _make_entry(entry_id: str = "settings") -> Mock:
    entry = Mock()
    entry.entry_id = entry_id
    entry.data = {"retention_time": 0}
    entry.options = {}
    return entry


def _event(uid: str, title: str = "Person at the door") -> Event:
    start = datetime.datetime.now(datetime.timezone.utc)
    return Event(
        uid=uid,
        title=title,
        start=start,
        end=start + datetime.timedelta(seconds=10),
        description="desc",
        key_frame="",
        camera_name="camera.front",
        category="person",
        label="person",
    )


async def _make_timeline(tmp_path, hass=None) -> Timeline:
    hass = hass or _make_hass(tmp_path)
    media = tmp_path / "snapshots"
    with patch("custom_components.llmvision.timeline.os.makedirs"):
        timeline = Timeline(hass, _make_entry())
    timeline._media_path = str(media)
    media.mkdir(parents=True, exist_ok=True)
    (tmp_path / "llmvision").mkdir(exist_ok=True)
    await timeline._initialize_db()
    timeline._migrating = False
    return timeline


class TestSharedTimeline:
    def test_get_timeline_reuses_instance(self, tmp_path):
        hass = _make_hass(tmp_path)
        entry = _make_entry()
        with patch("custom_components.llmvision.timeline.os.makedirs"):
            first = get_timeline(hass, entry)
            second = get_timeline(hass, entry)
        assert first is second
        assert hass.data[TIMELINE_KEY] is first

    def test_get_timeline_new_instance_for_other_entry(self, tmp_path):
        hass = _make_hass(tmp_path)
        with patch("custom_components.llmvision.timeline.os.makedirs"):
            first = get_timeline(hass, _make_entry("a"))
            second = get_timeline(hass, _make_entry("b"))
        assert first is not second


class TestInMemoryEvents:
    @pytest.mark.asyncio
    async def test_insert_does_not_reload_table(self, tmp_path):
        timeline = await _make_timeline(tmp_path)
        await timeline.load_events()

        with patch.object(timeline, "load_events", AsyncMock()) as load:
            await timeline._insert_event(_event("one"))
            events = await timeline.get_all_events()

        load.assert_not_awaited()
        assert [e.uid for e in events] == ["one"]

    @pytest.mark.asyncio
    async def test_insert_before_initial_load_is_read_from_db(self, tmp_path):
        timeline = await _make_timeline(tmp_path)

        await timeline._insert_event(_event("one"))
        events = await timeline.get_all_events()

        assert [e.uid for e in events] == ["one"]

    @pytest.mark.asyncio
    async def test_memory_matches_db_after_writes(self, tmp_path):
        timeline = await _make_timeline(tmp_path)
        await timeline.load_events()
        await timeline._insert_event(_event("one"))
        await timeline._insert_event(_event("two"))

        with patch.object(
            timeline, "_get_category_from_label", AsyncMock(return_value="vehicle")
        ):
            await timeline.update_event(
                uid="one",
                start=datetime.datetime.now(datetime.timezone.utc),
                end=datetime.datetime.now(datetime.timezone.utc),
                title="Car parked",
                description="desc",
                key_frame="",
                camera_name="camera.front",
                label="vehicle",
            )
        await timeline.delete_event("two")

        in_memory = {(e.uid, e.title, e.category) for e in timeline.events}
        await timeline.load_events()
        from_db = {(e.uid, e.title, e.category) for e in timeline.events}
        assert in_memory == from_db == {("one", "Car parked", "vehicle")}

    @pytest.mark.asyncio
    async def test_cleanup_keeps_linked_snapshot_before_initial_load(self, tmp_path):
        timeline = await _make_timeline(tmp_path)
        linked = tmp_path / "snapshots" / "linked.jpg"
        orphan = tmp_path / "snapshots" / "orphan.jpg"
        linked.write_bytes(b"x")
        orphan.write_bytes(b"x")
        event = _event("one")
        event.key_frame = str(linked)
        await timeline._insert_event(event)

        # Files older than the cleanup grace period
        with patch(
            "custom_components.llmvision.timeline.os.path.getmtime", return_value=0
        ):
            await timeline._cleanup()

        assert linked.exists()
        assert not orphan.exists()

    @pytest.mark.asyncio
    async def test_insert_during_initial_load_is_not_lost(self, tmp_path):
        timeline = await _make_timeline(tmp_path)

        # An initial load holding the lock, as if its SELECT ran before the insert
        await timeline._events_lock.acquire()
        insert = asyncio.ensure_future(timeline._insert_event(_event("one")))
        await asyncio.sleep(0.05)
        timeline.events = []
        timeline._loaded = True
        timeline._events_lock.release()
        await insert

        assert [e.uid for e in await timeline.get_all_events()] == ["one"]

    @pytest.mark.asyncio
    async def test_cleanup_skips_when_events_cannot_be_loaded(self, tmp_path):
        timeline = await _make_timeline(tmp_path)
        snapshot = tmp_path / "snapshots" / "maybe-linked.jpg"
        snapshot.write_bytes(b"x")

        with patch.object(timeline, "load_events", AsyncMock()), patch(
            "custom_components.llmvision.timeline.os.path.getmtime", return_value=0
        ):
            await timeline._cleanup()

        assert snapshot.exists()

    @pytest.mark.asyncio
    async def test_update_of_missing_event_adds_nothing(self, tmp_path):
        timeline = await _make_timeline(tmp_path)
        await timeline.load_events()

        with patch.object(
            timeline, "_get_category_from_label", AsyncMock(return_value="person")
        ):
            await timeline.update_event(
                uid="gone",
                start=datetime.datetime.now(datetime.timezone.utc),
                end=datetime.datetime.now(datetime.timezone.utc),
                title="Ghost",
                description="",
                key_frame="",
                camera_name="camera.front",
                label="person",
            )

        assert timeline.events == []
