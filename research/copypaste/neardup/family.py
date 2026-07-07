"""Clustering functions into near-duplicate families.

Three tiers of grouping key, increasing tolerance:
  - group_exact       : identical opcode (mnemonic) stream            [Tier 2a]
  - group_normalized  : identical compiler-idiom-normalized signature [Tier 2b]
  - group_fuzzy       : MinHash/LSH k-gram similarity >= threshold    [Tier 3 cand.]

Each returns a list of families (lists of function names) with >= min members,
sorted by (member count, total instructions) descending -- the canonical order
the text report renders.
"""
from collections import defaultdict

from .disasm import opcodes
from .normalize import norm_signature


def _sorted_families(groups, funcs, minmembers):
    fams = [v for v in groups.values() if len(v) >= minmembers]
    fams.sort(key=lambda v: (len(v), sum(len(funcs[n]) for n in v)), reverse=True)
    return fams


def group_normalized(funcs, minmembers=2):
    """Families keyed by compiler-idiom-normalized signature (default report tier)."""
    groups = defaultdict(list)
    for name, ins in funcs.items():
        groups[norm_signature(ins)].append(name)
    return _sorted_families(groups, funcs, minmembers)


def group_exact(funcs, minmembers=2):
    """Families keyed by exact opcode (mnemonic) stream."""
    groups = defaultdict(list)
    for name, ins in funcs.items():
        groups[opcodes(ins)].append(name)
    return _sorted_families(groups, funcs, minmembers)


def group_fuzzy(funcs, threshold=0.85, num_perm=64, k=5, minmembers=2):
    """Fuzzy families via MinHash+LSH over k-gram opcode shingles.

    Requires `datasketch`. Returns [] if it is unavailable.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        return []
    ops = {name: opcodes(ins) for name, ins in funcs.items()}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh = {}
    for name, o in ops.items():
        sh = {' '.join(o[i:i + k]) for i in range(len(o) - (k - 1))}
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode())
        mh[name] = m
        lsh.insert(name, m)
    par = {n: n for n in mh}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for n, m in mh.items():
        for c in lsh.query(m):
            if c != n and mh[c].jaccard(m) >= threshold:
                par[find(c)] = find(n)
    comp = defaultdict(list)
    for n in mh:
        comp[find(n)].append(n)
    return _sorted_families(comp, funcs, minmembers)
