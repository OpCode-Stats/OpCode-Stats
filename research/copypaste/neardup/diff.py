"""Index-wise operand difference between two same-shape family members."""
from .normalize import norm_inst_role, inst_value, pretty


def diff_members(rep_ins, mem_ins, rep_name, mem_name, str_at):
    """Same normalized-shape family -> index-wise operand diff.

    This accepts small compiler idiom differences such as xor reg,reg vs
    mov reg,0/1 when their normalized roles match.

    Returns a list of (srcline, rep_display, mem_display, kind).
    """
    diffs = []
    for (rm, ro, rf, rl), (mm, mo, mf, ml) in zip(rep_ins, mem_ins):
        rr = norm_inst_role(rm, ro)
        mr = norm_inst_role(mm, mo)
        if rr == mr:
            rn, rkind = inst_value(rm, ro, rep_name, str_at)
            mn, mkind = inst_value(mm, mo, mem_name, str_at)
        else:
            rn, rkind = (rm + ' ' + ro).strip(), 'instruction'
            mn, mkind = (mm + ' ' + mo).strip(), 'instruction'
        if rn != mn and rkind != 'padding' and mkind != 'padding':
            kind = rkind if rkind == mkind else f'{rkind}/{mkind}'
            diffs.append((ml or rl, pretty(rn, ro), pretty(mn, mo), kind))
    return diffs
