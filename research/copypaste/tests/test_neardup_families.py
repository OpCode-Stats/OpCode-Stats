"""Hermetic regression tests over committed objdump fixtures.

These require no binary on disk and no objdump run -- they parse the checked-in
`tests/fixtures/*.objdump` text, so they lock the detector's behavior on the two
anchor families:
  - Redis ZRANGE wrappers  -> one 6-member value-parameter family
  - OpenACR ReadStrptrMaybe -> one family, callback/policy parameter detected
"""
import json
import os

from conftest import FIXTURES

from neardup.disasm import parse_disasm, null_string_at
from neardup.normalize import norm_inst_role, norm_signature
from neardup.report import analyze_families, render_json


def _families(fixture):
    funcs = parse_disasm(open(os.path.join(FIXTURES, fixture)).read())
    dm = {n: n for n in funcs}
    return funcs, analyze_families(funcs, dm, null_string_at, minmembers=2)


# --------------------------------------------------------------------------
# Redis ZRANGE -- the headline handwritten copy-paste family
# --------------------------------------------------------------------------
ZRANGE_MEMBERS = {
    'zrangeCommand', 'zrevrangeCommand', 'zrangebyscoreCommand',
    'zrevrangebyscoreCommand', 'zrangebylexCommand', 'zrevrangebylexCommand',
}


def test_zrange_is_one_six_member_family():
    _, fams = _families('redis_zrange.objdump')
    assert len(fams) == 1, "ZRANGE wrappers must collapse to a single family"
    fam = fams[0]
    assert fam['members_count'] == 6
    names = {fam['representative']} | {m['name'] for m in fam['members']}
    assert names == ZRANGE_MEMBERS


def test_zrange_varies_only_by_immediate_constants():
    _, fams = _families('redis_zrange.objdump')
    fam = fams[0]
    # the two varying parameters (rangetype, direction) are compile-time constants
    assert fam['kinds'] == ['immediate constant']
    assert fam['suggested_abstraction'] == 'value params'
    assert 'template/constexpr' in fam['style']


def test_zrange_representative_value_is_zero_and_variants_are_small_ints():
    _, fams = _families('redis_zrange.objdump')
    fam = fams[0]
    assert fam['differences'], "expected per-line immediate diffs for ZRANGE"
    for d in fam['differences']:
        assert d['kind'] == 'immediate constant'
        assert d['rep_value'] == '0'  # zrangeCommand uses ZRANGE_AUTO/FORWARD == 0
        # variants are the other rangetype/direction enum values
        assert all(v.startswith('0x') for v in d['variants'])


# --------------------------------------------------------------------------
# OpenACR ReadStrptrMaybe -- generated-code family, policy/callee parameter
# --------------------------------------------------------------------------
def test_readstrptr_is_one_family():
    funcs, fams = _families('openacr_readstrptr.objdump')
    assert len(funcs) == 8
    assert len(fams) == 1
    assert fams[0]['members_count'] == 8


def test_readstrptr_detects_callback_policy_parameter():
    _, fams = _families('openacr_readstrptr.objdump')
    fam = fams[0]
    # per-family callees differ -> a callback/policy parameter is the abstraction
    assert 'callee' in fam['kinds']
    assert 'callback/policy param' in fam['suggested_abstraction']
    assert fam['style'] == 'runtime params/policy'


# --------------------------------------------------------------------------
# JSON report shape
# --------------------------------------------------------------------------
def test_json_report_is_serializable_and_shaped():
    funcs, fams = _families('redis_zrange.objdump')
    report = render_json('redis_zrange.objdump', fams, minmembers=2)
    blob = json.dumps(report)  # must be JSON-serializable
    back = json.loads(blob)
    assert back['family_count'] == 1
    fam = back['families'][0]
    for key in ('representative', 'members_count', 'members', 'differences',
                'kinds', 'suggested_abstraction', 'rep_src'):
        assert key in fam
    assert fam['members'][0]['name'] in ZRANGE_MEMBERS
    assert 'similarity' in fam['members'][0]


# --------------------------------------------------------------------------
# Compiler-idiom normalization -- the reason the 6 wrappers group despite
# xor/mov idiom differences
# --------------------------------------------------------------------------
def test_zero_idioms_normalize_together():
    assert norm_inst_role('xor', 'eax,eax') == ('set-imm', 'rax')
    assert norm_inst_role('sub', 'rax,rax') == ('set-imm', 'rax')
    assert norm_inst_role('mov', 'eax,0x0') == ('set-imm', 'rax')
    # all three are the same normalized role
    assert (norm_inst_role('xor', 'eax,eax')
            == norm_inst_role('mov', 'eax,0x0'))


def test_nonzero_immediate_is_set_imm():
    assert norm_inst_role('mov', 'esi,0x2') == ('set-imm', 'rsi')


def test_padding_normalizes_to_pad():
    assert norm_inst_role('nop', '') == ('pad',)


def test_signature_length_matches_instruction_count():
    funcs, _ = _families('redis_zrange.objdump')
    ins = funcs['zrangeCommand']
    assert len(norm_signature(ins)) == len(ins)
