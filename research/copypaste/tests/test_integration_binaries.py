"""Integration tests against the real binaries (auto-skipped if absent).

These exercise the full pipeline including objdump and rodata string
resolution, and lock the end-to-end anchor results:
  - redis-server : ZRANGE -> 6-member family
  - amc          : ReadStrptrMaybe -> 77-member family with string-literal params
"""
import os

import pytest

from neardup.disasm import disasm, demangle, rodata
from neardup.report import analyze_families

REDIS = '/home/aaslyan/cp-subjects/redis/src/redis-server'
AMC = '/home/aaslyan/openacr-mine/build/release/amc'


def _run(path):
    str_at = rodata(path)
    funcs = disasm(path)
    dm = demangle(list(funcs))
    return analyze_families(funcs, dm, str_at, minmembers=2)


@pytest.mark.skipif(not os.path.exists(REDIS), reason="redis-server fixture absent")
def test_real_redis_zrange_six_members():
    fams = _run(REDIS)
    zr = [f for f in fams if f['representative'] == 'zrangeCommand']
    assert len(zr) == 1
    fam = zr[0]
    assert fam['members_count'] == 6
    assert fam['kinds'] == ['immediate constant']


@pytest.mark.skipif(not os.path.exists(AMC), reason="amc binary absent")
def test_real_amc_readstrptr_family_with_string_params():
    fams = _run(AMC)
    rs = [f for f in fams
          if 'ReadStrptrMaybe' in f['representative'] and f['members_count'] >= 50]
    assert rs, "expected the large ReadStrptrMaybe family"
    fam = max(rs, key=lambda f: f['members_count'])
    assert fam['members_count'] == 77
    # full rodata resolution recovers the string-literal parameter
    assert 'string literal' in fam['kinds']
    assert 'callee' in fam['kinds']
