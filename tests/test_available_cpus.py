"""CPU detection must follow the container's cgroup quota, not the host's
core count — torch.set_num_threads(48) inside a fractional-CPU container
thrashes and inflates per-thread memory."""

import pytest

from app.services import trocr_service
from app.services.trocr_service import _available_cpus


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """Point the cgroup constants at temp files; missing ones stay missing."""

    def _write(v2=None, v1_quota=None, v1_period=None):
        for name, attr, value in (
            ("cpu.max", "CGROUP_V2_CPU_MAX", v2),
            ("cfs_quota_us", "CGROUP_V1_CPU_QUOTA", v1_quota),
            ("cfs_period_us", "CGROUP_V1_CPU_PERIOD", v1_period),
        ):
            path = tmp_path / name
            if value is not None:
                path.write_text(value)
            monkeypatch.setattr(trocr_service, attr, str(path))

    return _write


def test_reads_cgroup_v2_quota(cgroup):
    cgroup(v2="200000 100000")  # 2 cores
    assert _available_cpus() == 2


def test_v2_fractional_quota_floors_to_one_core(cgroup):
    cgroup(v2="50000 100000")  # half a core
    assert _available_cpus() == 1


def test_v2_unlimited_falls_back_to_affinity(cgroup):
    cgroup(v2="max 100000")
    assert _available_cpus() >= 1


def test_reads_cgroup_v1_quota(cgroup):
    cgroup(v1_quota="400000", v1_period="100000")  # 4 cores
    assert _available_cpus() == 4


def test_v1_unlimited_quota_falls_back(cgroup):
    cgroup(v1_quota="-1", v1_period="100000")
    assert _available_cpus() >= 1


def test_no_cgroup_files_falls_back(cgroup):
    cgroup()
    assert _available_cpus() >= 1


def test_malformed_quota_does_not_raise(cgroup):
    cgroup(v2="garbage")
    assert _available_cpus() >= 1


def test_zero_period_does_not_raise(cgroup):
    cgroup(v2="100000 0")
    assert _available_cpus() >= 1
