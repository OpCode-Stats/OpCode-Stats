#!/usr/bin/env python3
"""Universal near-duplicate code analyzer.
Usage: neardup_analyze.py [--fuzzy] [--json OUT] <binary> [<binary> ...]
Measures, for ANY ELF binary (function analysis needs symbols):
  - near-duplicate prevalence (opcode-identical %, and --fuzzy near-dup %)
  - mergeable opportunity in bytes (from symbol sizes)
  - cross-binary document-frequency spectrum (when >1 binary given)
"""
import json, argparse
from collections import defaultdict

# Shared canonical parser -- no duplicate objdump regex parser in this repo.
from neardup.disasm import disasm, func_opcodes, func_sizes


def funcs_opcodes(path):
    """{func: opcode-tuple} via the shared neardup parser."""
    return func_opcodes(disasm(path))

def analyze_one(path, fuzzy=False):
    f=funcs_opcodes(path)
    if len(f)<2: return {'path':path,'functions':len(f),'note':'no/insufficient symbols (stripped?) — function analysis unavailable'}
    # exact opcode-identical groups
    g=defaultdict(list)
    for name,ops in f.items(): g[ops].append(name)
    dup=[v for v in g.values() if len(v)>1]
    dup_funcs=sum(len(v) for v in dup)
    # opportunity bytes (sum of family bytes minus one kept per group), needs nm sizes
    sizes=func_sizes(path); raw=path.split('/')[-1]
    opp=0
    for grp in dup:
        bs=[sizes.get(n,0) for n in grp]
        if any(bs): opp+=sum(bs)-max(bs)
    res={'path':path,'functions':len(f),
         'opcode_identical_funcs':dup_funcs,
         'opcode_identical_pct':round(100*dup_funcs/len(f),1),
         'exact_dup_groups':len(dup),
         'mergeable_bytes_est':opp}
    if fuzzy:
        try:
            from datasketch import MinHash, MinHashLSH
            lsh=MinHashLSH(threshold=0.85,num_perm=64);mh={}
            for name,ops in f.items():
                sh={' '.join(ops[i:i+5]) for i in range(len(ops)-4)}
                m=MinHash(num_perm=64)
                for s in sh: m.update(s.encode())
                mh[name]=m;lsh.insert(name,m)
            par={n:n for n in mh}
            def find(x):
                while par[x]!=x: par[x]=par[par[x]];x=par[x]
                return x
            for n,m in mh.items():
                for c in lsh.query(m):
                    if c!=n and mh[c].jaccard(m)>=0.85: par[find(c)]=find(n)
            comp=defaultdict(list)
            for n in mh: comp[find(n)].append(n)
            nd=sum(len(v) for v in comp.values() if len(v)>1)
            res['near_dup_pct']=round(100*nd/len(f),1)
        except ImportError: pass
    return res

def cross_binary(paths):
    sig2bins=defaultdict(set)
    for p in paths:
        for ops in set(funcs_opcodes(p).values()): sig2bins[hash(ops)].add(p)
    N=len(paths); df=defaultdict(int)
    for s,bs in sig2bins.items(): df[len(bs)]+=1
    return N, dict(sorted(df.items()))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('bins',nargs='+'); ap.add_argument('--fuzzy',action='store_true'); ap.add_argument('--json')
    a=ap.parse_args()
    report={'per_binary':[],'cross_binary':None}
    print(f"{'binary':22}{'funcs':>7}{'opc-id%':>9}{'near-dup%':>10}{'merge-KB':>10}")
    for p in a.bins:
        r=analyze_one(p,a.fuzzy); report['per_binary'].append(r)
        if 'note' in r: print(f"{p.split('/')[-1]:22}{r['functions']:>7}  {r['note']}")
        else: print(f"{p.split('/')[-1]:22}{r['functions']:>7}{r['opcode_identical_pct']:>8.1f}%"
                    f"{(str(r.get('near_dup_pct','-'))+'%' if a.fuzzy else '-'):>10}{r['mergeable_bytes_est']/1024:>9.1f}")
    if len(a.bins)>1:
        N,df=cross_binary(a.bins); report['cross_binary']={'n_binaries':N,'df_histogram':df}
        print(f"\ncross-binary DF (of {N} binaries): "+", ".join(f"DF{k}:{v}" for k,v in df.items()))
    if a.json: json.dump(report,open(a.json,'w'),indent=1); print(f"\nwrote {a.json}")
