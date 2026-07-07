"""neardup -- binary-guided copy-paste refactoring assistant.

Single shared library behind the neardup_report / neardup_analyze CLIs.
All disassembly parsing lives in `disasm`; there is exactly one objdump
regex parser in the codebase (this one).

Modules:
  disasm     -- objdump/nm/c++filt I/O, the canonical instruction parser,
                source-range + rodata string resolution
  normalize  -- opcode/operand/compiler-idiom normalization
  family     -- exact / normalized / fuzzy clustering into families
  diff       -- index-wise operand/value/callee/string/data-ref differences
  report     -- structured family model + text/JSON rendering
"""
