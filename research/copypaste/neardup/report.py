"""Structured family model + text / JSON rendering.

`analyze_families` computes one structured model per near-duplicate family
(the single source of truth); `render_text` and `render_json` are pure
renderers over that model. The text renderer reproduces the historical
neardup_report.py output byte-for-byte.
"""
import sys

from .diff import diff_members
from .disasm import src_range
from .family import group_normalized


def _hint_and_style(kinds):
    """Suggested-abstraction hint + implementation style from the diff kinds."""
    parts = []
    if kinds & {'immediate constant'}:
        parts.append("value params")
    if 'string literal' in kinds:
        parts.append("string-literal params")
    if 'callee' in kinds:
        parts.append("a callback/policy param")
    if 'data ref' in kinds:
        parts.append("data-ref params")
    hint = " + ".join(parts) if parts else "no differences (byte/ICF-foldable)"
    style = ("template/constexpr (hot path, zero-cost)"
             if kinds <= {'immediate constant', 'string literal'}
             else "runtime params/policy")
    return hint, style


def analyze_families(funcs, dm, str_at, minmembers=2):
    """Compute structured models for every normalized-signature family.

    funcs   : {name: instrs}          (from disasm)
    dm      : {name: demangled}       (from demangle)
    str_at  : vaddr -> str|None       (from rodata / null_string_at)
    Returns a list of family dicts in canonical (rendered) order.
    """
    fams = group_normalized(funcs, minmembers)
    models = []
    for fi, fam in enumerate(fams, 1):
        rep = fam[0]
        rins = funcs[rep]
        f, l0, l1 = src_range(rins)
        # sl -> {repval: [variants set, kind]}, preserving repval insertion order
        perline = {}
        members = []
        for m in fam[1:]:
            mins = funcs[m]
            if len(mins) != len(rins):
                continue
            d = diff_members(rins, mins, rep, m, str_at)
            mf, ml0, ml1 = src_range(mins)
            sim = 100.0 * (len(rins) - len(d)) / len(rins)
            members.append({
                'name': m, 'demangled': dm.get(m, m),
                'src': {'file': mf, 'line_start': ml0, 'line_end': ml1},
                'similarity': sim,
            })
            for (sl, rd, md, kind) in d:
                bucket = perline.setdefault(sl, {})
                cell = bucket.setdefault(rd, [set(), ''])
                cell[0].add(md)
                cell[1] = kind
        kinds = {cell[1] for bucket in perline.values() for cell in bucket.values()}
        hint, style = _hint_and_style(kinds)
        # flatten differences in text-render order: sorted source line, then
        # repval insertion order within the line
        differences = []
        for sl in sorted(k for k in perline if k):
            for repval, (variants, kind) in perline[sl].items():
                differences.append({
                    'file': f, 'line': sl, 'kind': kind,
                    'rep_value': repval, 'variants': sorted(variants),
                })
        models.append({
            'index': fi,
            'representative': rep,
            'representative_demangled': dm.get(rep, rep),
            'members_count': len(fam),
            'instrs': len(rins),
            'rep_src': {'file': f, 'line_start': l0, 'line_end': l1},
            'members': members,
            'differences': differences,
            'kinds': sorted(kinds),
            'suggested_abstraction': hint,
            'style': style,
            '_perline': perline,  # internal: exact-shape state for text renderer
        })
    return models


def render_text(binpath, models, minmembers=2, out=None):
    """Render the historical text report (byte-identical to legacy output)."""
    out = out or sys.stdout
    p = lambda s='': print(s, file=out)
    p(f"# near-duplicate copy-paste report: {binpath}")
    p(f"# {len(models)} families (>= {minmembers} members)\n")
    for fam in models:
        f = fam['rep_src']['file']
        l0, l1 = fam['rep_src']['line_start'], fam['rep_src']['line_end']
        p(f"== Family {fam['index']}: {fam['members_count']} near-duplicate "
          f"functions, {fam['instrs']} instrs each ==")
        p(f"   representative: {fam['representative_demangled']}")
        p(f"                   [{f}:{l0}-{l1}]")
        for mem in fam['members']:
            ms = mem['src']
            p(f"   ~ {mem['demangled']}   [{ms['file']}:{ms['line_start']}-"
              f"{ms['line_end']}]  sim {mem['similarity']:.1f}%")
        perline = fam['_perline']
        if perline:
            p("   -- what differs (mapped to source line) --")
            for sl in sorted(k for k in perline if k):
                for repval, (variants, kind) in perline[sl].items():
                    sv = sorted(variants)
                    shown = ", ".join(sv[:4]) + (" ..." if len(sv) > 4 else "")
                    p(f"       {f}:{sl:<4} [{kind}]  {repval}  ->  {shown}")
        p(f"   => {fam['members_count']} funcs -> ONE, parameterized by: "
          f"{fam['suggested_abstraction']}.  Suggested: {fam['style']}.\n")


def render_json(binpath, models, minmembers=2):
    """Serializable report dict: families with members, source ranges,
    parameter diffs/kinds, and suggested abstraction."""
    families = []
    for fam in models:
        families.append({
            'index': fam['index'],
            'representative': fam['representative'],
            'representative_demangled': fam['representative_demangled'],
            'members_count': fam['members_count'],
            'instrs': fam['instrs'],
            'rep_src': fam['rep_src'],
            'members': fam['members'],
            'differences': fam['differences'],
            'kinds': fam['kinds'],
            'suggested_abstraction': fam['suggested_abstraction'],
            'style': fam['style'],
        })
    return {
        'binary': binpath,
        'min_members': minmembers,
        'family_count': len(families),
        'families': families,
    }
