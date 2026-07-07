#!/usr/bin/env python3
"""neardup_report -- point at a (-g) binary, get a copy-paste report:
which functions repeat, mapped to source ranges, with a 'what differs' diff.

For each near-duplicate family:
  - members (demangled) with source file:line-line range
  - pairwise similarity vs the representative
  - the exact differences (immediate constants / call targets / extra blocks),
    each mapped back to the source line -> "here it's X, there it's Y"

Thin CLI over the `neardup` package (all logic lives there).

Usage: neardup_report.py [--min N] [--json OUT] <binary-with-symbols-and-debug>
"""
import argparse
import json

from neardup.disasm import disasm, demangle, rodata
from neardup.report import analyze_families, render_text, render_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bin')
    ap.add_argument('--min', type=int, default=2, help='min members per family')
    ap.add_argument('--json', metavar='OUT',
                    help='write structured JSON report to OUT (- for stdout)')
    a = ap.parse_args()

    str_at = rodata(a.bin)
    funcs = disasm(a.bin)
    dm = demangle(list(funcs))
    models = analyze_families(funcs, dm, str_at, minmembers=a.min)

    if a.json:
        report = render_json(a.bin, models, minmembers=a.min)
        if a.json == '-':
            print(json.dumps(report, indent=1))
        else:
            with open(a.json, 'w') as fh:
                json.dump(report, fh, indent=1)
            print(f"wrote {a.json}  ({report['family_count']} families)")
    else:
        render_text(a.bin, models, minmembers=a.min)


if __name__ == '__main__':
    main()
