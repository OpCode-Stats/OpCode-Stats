"""Canonical disassembly parsing -- the ONE objdump parser in the codebase.

`run_objdump` (I/O) is split from `parse_disasm` (pure text -> functions) so
tests can feed committed objdump fixtures without needing the binary present.

Instruction tuple layout: (mnemonic, operand_str, srcfile, srcline).
Functions shorter than MINLEN instructions are dropped (wrapper noise floor).
"""
import re
import subprocess

FUNC = re.compile(r'^([0-9a-f]+) <(.+?)>:$')
SRCLINE = re.compile(r'^(/?\S+\.(?:cpp|cc|c|h|hpp)):(\d+)(?:\s|$)')
# --no-show-raw-insn => "  <addr>:\t<mnem> <operands>"
INS = re.compile(r'^\s*([0-9a-f]+):\s+([a-zA-Z][\w.]*)\s*(.*)$')
MINLEN = 12


def run_objdump(path):
    """Disassemble with intel syntax + source line interleave, no raw bytes."""
    return subprocess.run(
        ['objdump', '-d', '-l', '-M', 'intel', '--no-show-raw-insn', path],
        capture_output=True, text=True).stdout


def parse_disasm(text, minlen=MINLEN):
    """Pure parse of `objdump -dl --no-show-raw-insn` text.

    Returns {func_name: [(mnem, operand, srcfile, srcline), ...]} keeping only
    functions with >= minlen instructions.
    """
    funcs = {}
    cur = None
    ins = []
    sf = None
    sl = None
    for line in text.splitlines():
        m = FUNC.match(line.strip())
        if m:
            if cur and len(ins) >= minlen:
                funcs[cur] = ins
            cur = m.group(2)
            ins = []
            continue
        ms = SRCLINE.match(line.strip())
        if ms:
            sf, sl = ms.group(1), int(ms.group(2))
            continue
        mi = INS.match(line)
        if mi and cur is not None:
            ins.append((mi.group(2), mi.group(3).strip(), sf, sl))
    if cur and len(ins) >= minlen:
        funcs[cur] = ins
    return funcs


def disasm(path, minlen=MINLEN):
    """Return {func: [(mnem, operand, srcfile, srcline), ...]} for a binary."""
    return parse_disasm(run_objdump(path), minlen)


def opcodes(ins):
    """Mnemonic-only opcode stream for a function's instruction list."""
    return tuple(i[0] for i in ins)


def func_opcodes(funcs):
    """Map {func: instrs} -> {func: opcode-tuple}."""
    return {name: opcodes(ins) for name, ins in funcs.items()}


def func_sizes(path):
    """Symbol sizes (in bytes) for defined text symbols, via nm -S."""
    sz = {}
    for ln in subprocess.run(['nm', '-S', '--defined-only', path],
                             capture_output=True, text=True).stdout.splitlines():
        p = ln.split()
        if len(p) >= 4 and p[2] in 'Tt':
            try:
                sz[p[3]] = int(p[1], 16)
            except ValueError:
                pass
    return sz


def src_range(ins):
    """(basename, min_line, max_line) over an instruction list's source info."""
    ls = [l for (_, _, f, l) in ins if l]
    fs = [f for (_, _, f, l) in ins if f]
    if not ls:
        return ('?', 0, 0)
    return (fs[0].split('/')[-1], min(ls), max(ls))


def demangle(names):
    """Batch c++filt; returns {mangled: demangled}."""
    if not names:
        return {}
    out = subprocess.run(['c++filt'], input="\n".join(names),
                         capture_output=True, text=True).stdout.split("\n")
    return dict(zip(names, out))


def rodata(path):
    """Build a vaddr->string resolver from read-only string sections.

    Used to turn RIP-relative data loads into 'STR:<literal>' when the target
    is a printable C string. Returns a `string_at(vaddr) -> str|None` closure.
    """
    segs = []
    for sec in ('.rodata', '.rodata.str1.1', '.rodata.str1.8', '.data.rel.ro'):
        out = subprocess.run(['objdump', '-s', '-j', sec, path],
                             capture_output=True, text=True).stdout
        base = None
        buf = bytearray()
        for ln in out.splitlines():
            m = re.match(r'^\s*([0-9a-f]+)\s+((?:[0-9a-f]{2,8} ){1,4})', ln)
            if not m:
                continue
            addr = int(m.group(1), 16)
            hexb = ''.join(m.group(2).split())
            b = bytes.fromhex(hexb)
            if base is None:
                base = addr
                buf = bytearray(b)
            else:
                gap = addr - (base + len(buf))
                if 0 <= gap < 64:
                    buf += b'\x00' * gap + b
                else:
                    segs.append((base, bytes(buf)))
                    base = addr
                    buf = bytearray(b)
        if base is not None:
            segs.append((base, bytes(buf)))

    def string_at(vaddr):
        for base, buf in segs:
            if base <= vaddr < base + len(buf):
                off = vaddr - base
                end = buf.find(b'\x00', off)
                s = buf[off:end if end >= 0 else len(buf)]
                try:
                    t = s.decode('ascii')
                except UnicodeDecodeError:
                    return None
                return t if t.isprintable() and len(t) >= 2 else None
        return None
    return string_at


def null_string_at(_vaddr):
    """String resolver that never resolves -- for tests / binaries w/o rodata."""
    return None
