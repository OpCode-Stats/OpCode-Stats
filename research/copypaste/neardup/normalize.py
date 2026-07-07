"""Opcode / operand / compiler-idiom normalization.

The grouping key for a function is `norm_signature`: a tuple of per-instruction
roles where compiler idioms that mean the same source-level thing collapse
together. Concretely:
  - xor r,r / sub r,r / mov r,0  -> ('set-imm', r)   [materialize constant]
  - mov r,imm                    -> ('set-imm', r)
  - nop / data16 padding         -> ('pad',)
  - everything else              -> (mnemonic,)

`norm_operand` is the reloc-aware, position-independent operand normalizer used
to classify what actually differs between members (string / callee / data ref /
immediate).
"""
import re

REG_CANON = {}
for grp in [('rax', 'eax', 'ax', 'al', 'ah'), ('rbx', 'ebx', 'bx', 'bl', 'bh'),
            ('rcx', 'ecx', 'cx', 'cl', 'ch'), ('rdx', 'edx', 'dx', 'dl', 'dh'),
            ('rsi', 'esi', 'si', 'sil'), ('rdi', 'edi', 'di', 'dil'),
            ('rbp', 'ebp', 'bp', 'bpl'), ('rsp', 'esp', 'sp', 'spl')]:
    for r in grp:
        REG_CANON[r] = grp[0]
for _i in range(8, 16):
    for _suf in ('', 'd', 'w', 'b'):
        REG_CANON[f'r{_i}{_suf}'] = f'r{_i}'


def split_operands(op):
    """Split an operand string on top-level commas (ignoring [mem] commas)."""
    out = []
    depth = 0
    cur = ''
    for ch in op:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def canon_reg(op):
    """Canonical register class for an operand, or None if not a register."""
    return REG_CANON.get(op.strip().lower())


def is_imm(op):
    """True if the operand is a bare immediate constant."""
    return bool(re.match(r'^(?:0x[0-9a-f]+|-?\d+)$', op.strip().lower()))


def norm_inst_role(mn, op):
    """Compiler-idiom-normalized role used for family grouping.

    Compilers use xor/sub reg,reg for zero but mov reg,imm for nonzero
    constants. For wrapper code those are the same source-level role:
    materialize an argument constant in a register.
    """
    ops = split_operands(op)
    if mn in {'nop', 'data16'}:
        return ('pad',)
    if mn in {'xor', 'sub'} and len(ops) == 2:
        r0 = canon_reg(ops[0])
        r1 = canon_reg(ops[1])
        if r0 and r0 == r1:
            return ('set-imm', r0)
    if mn in {'mov', 'movabs'} and len(ops) == 2 and canon_reg(ops[0]) and is_imm(ops[1]):
        return ('set-imm', canon_reg(ops[0]))
    return (mn,)


def norm_signature(ins):
    """Normalized grouping key for a whole function."""
    return tuple(norm_inst_role(m, o) for m, o, _, _ in ins)


def norm_operand(op, fname, str_at):
    """Position-independent, reloc-aware normalization:
      - rip-relative DATA ref -> resolved string 'STR:<literal>' (or 'DATA@<addr>' if not a string)
      - branch/call to the *enclosing* function -> SELFREL (position artifact)
      - branch/call to a DISTINCT named symbol  -> SYM:<name> (genuine callee)
      - bare hex branch target                  -> REL
      - immediates                              -> kept verbatim (genuine constant diffs)"""
    if '[rip+' in op:
        c = re.search(r'#\s*([0-9a-f]+)', op)
        if c:
            v = int(c.group(1), 16)
            s = str_at(v)
            return 'STR:' + s if s else 'DATA@%x' % v
        return re.sub(r'\[rip\+0x[0-9a-f]+\]', '[rip+ADDR]', op)
    m = re.search(r'<([^>+]+)(\+0x[0-9a-f]+)?>', op)
    if m:
        sym = m.group(1)
        return 'SELFREL' if sym == fname else 'SYM:' + sym
    if re.match(r'^[0-9a-f]+$', op.strip()):
        return 'REL'
    return op


def inst_value(mn, op, fname, str_at):
    """(display_value, kind) for one instruction's differing operand.

    kind in {padding, immediate constant, string literal, callee, data ref}.
    """
    ops = split_operands(op)
    if mn in {'nop', 'data16'}:
        return ('', 'padding')
    if mn in {'xor', 'sub'} and len(ops) == 2:
        r0 = canon_reg(ops[0])
        r1 = canon_reg(ops[1])
        if r0 and r0 == r1:
            return ('0', 'immediate constant')
    if mn in {'mov', 'movabs'} and len(ops) == 2 and canon_reg(ops[0]) and is_imm(ops[1]):
        return (ops[1].lower(), 'immediate constant')
    n = norm_operand(op, fname, str_at)
    kind = ('string literal' if n.startswith('STR:')
            else 'callee' if n.startswith('SYM:')
            else 'data ref' if n.startswith('DATA@')
            else 'immediate constant')
    return (n, kind)


def pretty(nrm, raw):
    """Human display for a normalized value."""
    if nrm.startswith('STR:'):
        return '"%s"' % nrm[4:]
    if nrm.startswith(('SYM:', 'DATA@', 'SELFREL', 'REL')):
        return nrm
    if nrm == '0' or is_imm(nrm):
        return nrm
    return raw.strip()
