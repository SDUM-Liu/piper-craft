"""Ownership probes must never create a second device writer."""
import pytest
from astra_vla import launcher


def test_current_workers_reused_by_read_only_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher,'RUNTIME',tmp_path)
    calls=[]
    def read(path, method, **kwargs):
        calls.append((path.name,method))
        return {'motion_control':{'pose_source':'controller_sdk'},'pipeline':'rgb_only'}
    monkeypatch.setattr(launcher,'call',read)
    for name in ('robot','rgb','cameras'):
        (tmp_path/(name+'.sock')).touch()
        assert launcher.reuse_worker(name)
    assert calls==[('robot.sock','status'),('rgb.sock','snapshot'),('cameras.sock','snapshot')]


@pytest.mark.parametrize('name',['robot','cameras'])
def test_older_worker_is_not_silently_replaced(tmp_path,monkeypatch,name):
    monkeypatch.setattr(launcher,'RUNTIME',tmp_path)
    (tmp_path/(name+'.sock')).touch()
    monkeypatch.setattr(launcher,'call',lambda *a,**kw:{})
    with pytest.raises(RuntimeError,match='older'):
        launcher.reuse_worker(name)


def test_timeout_does_not_authorize_a_duplicate_writer(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'RUNTIME',tmp_path)
    (tmp_path/'robot.sock').touch()
    def timeout(*a,**kw):raise TimeoutError('owner still exists')
    monkeypatch.setattr(launcher,'call',timeout)
    with pytest.raises(TimeoutError):
        launcher.reuse_worker('robot')


def test_missing_or_dead_socket_can_start_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'RUNTIME',tmp_path)
    assert not launcher.reuse_worker('robot')
    (tmp_path/'robot.sock').touch()
    def dead(*a,**kw):raise ConnectionRefusedError()
    monkeypatch.setattr(launcher,'call',dead)
    assert not launcher.reuse_worker('robot')
