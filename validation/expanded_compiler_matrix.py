"""
expanded_compiler_matrix.py -- Expanded compiler matrix experiment.

Extends the original 5-program compiler matrix to 15+ programs compiled
with gcc and clang at five optimisation levels (-O0, -O1, -O2, -O3, -Os),
then decomposes per-metric variance into program / compiler / opt-level
components to answer: which factor dominates metric variation?

Output: validation/results/expanded_compiler_matrix.json
Binaries: validation/compiled_binaries/
"""

import sys
import json
import logging
import math
import os
import subprocess
import zlib
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, '/home/aaslyan/OpCode-Stats')

from utils.helpers import Binary, save_json, save_pickle, ensure_output_dir, build_vocabulary, encode_sequence
from extraction.disassemble import disassemble_binary
from analysis.frequency import compute_frequency_distribution, compute_rank_frequency, fit_zipf_mle
from analysis.ngrams import compute_entropy_rate, compute_shuffled_entropy_rates
from analysis.compression import compute_compression_ratios

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path('/home/aaslyan/OpCode-Stats')
COMPILED_DIR = PROJECT_ROOT / 'validation' / 'compiled_binaries'
OUTPUT_DIR = PROJECT_ROOT / 'validation' / 'results'
OUTPUT_JSON = OUTPUT_DIR / 'expanded_compiler_matrix.json'

# ---------------------------------------------------------------------------
# Compiler configurations
# ---------------------------------------------------------------------------

OPT_LEVELS = ['-O0', '-O1', '-O2', '-O3', '-Os']


def _available_compilers() -> List[str]:
    """Return list of C compiler names that are reachable on PATH."""
    compilers = []
    for cc in ['gcc', 'clang']:
        try:
            result = subprocess.run(
                [cc, '--version'], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                compilers.append(cc)
                logger.info(f"Compiler available: {cc}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info(f"Compiler not available: {cc}")
    return compilers


# ---------------------------------------------------------------------------
# Inline C programs (10 additional programs beyond the original 5)
# The originals (sort, hash, compress, search, matrix) live in
# experiments/compiler_matrix.py; we define a disjoint set here.
# ---------------------------------------------------------------------------

SOURCES: Dict[str, str] = {

"hello": r"""
#include <stdio.h>
int main(void) {
    puts("hello, world");
    return 0;
}
""",

"fibonacci": r"""
#include <stdio.h>
#include <stdint.h>

static uint64_t fib_iter(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    uint64_t a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        uint64_t c = a + b;
        a = b;
        b = c;
    }
    return b;
}

static uint64_t fib_rec(int n) {
    if (n <= 1) return (uint64_t)(n < 0 ? 0 : n);
    return fib_rec(n - 1) + fib_rec(n - 2);
}

int main(void) {
    for (int i = 0; i <= 20; i++) {
        uint64_t it = fib_iter(i);
        uint64_t rc = fib_rec(i);
        printf("fib(%2d) = %llu  match=%d\n", i, (unsigned long long)it, it == rc);
    }
    return 0;
}
""",

"bsearch": r"""
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}

static int bsearch_manual(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int main(void) {
    int N = 1024;
    int *arr = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) arr[i] = i * 3;
    qsort(arr, N, sizeof(int), cmp_int);

    int found = 0, missed = 0;
    for (int q = 0; q < N * 3; q += 2) {
        int idx = bsearch_manual(arr, N, q);
        if (idx >= 0) found++;
        else missed++;
    }
    printf("found=%d missed=%d\n", found, missed);
    free(arr);
    return 0;
}
""",

"strmanip": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static void reverse_str(char *s, int n) {
    for (int i = 0, j = n - 1; i < j; i++, j--) {
        char tmp = s[i]; s[i] = s[j]; s[j] = tmp;
    }
}

static int is_palindrome(const char *s, int n) {
    for (int i = 0, j = n - 1; i < j; i++, j--)
        if (s[i] != s[j]) return 0;
    return 1;
}

static void to_upper(char *s) {
    for (; *s; s++) *s = (char)toupper((unsigned char)*s);
}

static int count_words(const char *s) {
    int in_word = 0, count = 0;
    for (; *s; s++) {
        if (isspace((unsigned char)*s)) { in_word = 0; }
        else if (!in_word) { in_word = 1; count++; }
    }
    return count;
}

int main(void) {
    const char *words[] = {"racecar", "hello", "level", "world", "kayak", "test"};
    for (int i = 0; i < 6; i++) {
        char buf[64];
        strncpy(buf, words[i], sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        int n = (int)strlen(buf);
        int pal = is_palindrome(buf, n);
        reverse_str(buf, n);
        to_upper(buf);
        printf("'%s' rev='%s' palindrome=%d\n", words[i], buf, pal);
    }
    const char *sentence = "the quick brown fox jumps over the lazy dog";
    printf("words=%d\n", count_words(sentence));
    return 0;
}
""",

"linked_list": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node { int val; struct Node *next; } Node;

static Node *push(Node *head, int v) {
    Node *n = malloc(sizeof(Node));
    n->val = v; n->next = head;
    return n;
}

static Node *pop(Node *head, int *out) {
    if (!head) return NULL;
    *out = head->val;
    Node *next = head->next;
    free(head);
    return next;
}

static Node *insert_sorted(Node *head, int v) {
    if (!head || head->val >= v) { Node *n = malloc(sizeof(Node)); n->val = v; n->next = head; return n; }
    head->next = insert_sorted(head->next, v);
    return head;
}

static void free_list(Node *h) { while (h) { Node *t = h->next; free(h); h = t; } }

static int list_len(Node *h) { int c = 0; while (h) { c++; h = h->next; } return c; }

int main(void) {
    Node *head = NULL;
    for (int i = 0; i < 20; i++) head = push(head, i * 7 % 31);
    printf("len=%d\n", list_len(head));

    Node *sorted = NULL;
    for (int i = 0; i < 20; i++) sorted = insert_sorted(sorted, i * 3 % 17);
    int prev = -1, ok = 1;
    for (Node *p = sorted; p; p = p->next) {
        if (p->val < prev) ok = 0;
        prev = p->val;
    }
    printf("sorted_ok=%d len=%d\n", ok, list_len(sorted));

    free_list(head);
    free_list(sorted);
    return 0;
}
""",

"stack_queue": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Array-based stack */
typedef struct { int *data; int top; int cap; } Stack;

static void stack_init(Stack *s, int cap) {
    s->data = malloc(cap * sizeof(int));
    s->top = 0; s->cap = cap;
}
static void stack_push(Stack *s, int v) {
    if (s->top < s->cap) s->data[s->top++] = v;
}
static int stack_pop(Stack *s) {
    return s->top > 0 ? s->data[--s->top] : -1;
}
static void stack_free(Stack *s) { free(s->data); }

/* Circular-buffer queue */
typedef struct { int *data; int head; int tail; int size; int cap; } Queue;

static void queue_init(Queue *q, int cap) {
    q->data = malloc(cap * sizeof(int));
    q->head = q->tail = q->size = 0; q->cap = cap;
}
static void enqueue(Queue *q, int v) {
    if (q->size < q->cap) {
        q->data[q->tail] = v;
        q->tail = (q->tail + 1) % q->cap;
        q->size++;
    }
}
static int dequeue(Queue *q) {
    if (!q->size) return -1;
    int v = q->data[q->head];
    q->head = (q->head + 1) % q->cap;
    q->size--;
    return v;
}
static void queue_free(Queue *q) { free(q->data); }

int main(void) {
    Stack st; stack_init(&st, 64);
    for (int i = 0; i < 50; i++) stack_push(&st, i * i);
    int sum = 0;
    for (int i = 0; i < 50; i++) sum += stack_pop(&st);
    printf("stack_sum=%d\n", sum);
    stack_free(&st);

    Queue q; queue_init(&q, 128);
    for (int i = 0; i < 100; i++) enqueue(&q, i);
    int qsum = 0;
    while (q.size) qsum += dequeue(&q);
    printf("queue_sum=%d\n", qsum);
    queue_free(&q);
    return 0;
}
""",

"tree": r"""
#include <stdio.h>
#include <stdlib.h>

typedef struct Node { int key; struct Node *left, *right; int height; } Node;

static int ht(Node *n) { return n ? n->height : 0; }
static int max2(int a, int b) { return a > b ? a : b; }
static void update_ht(Node *n) { n->height = 1 + max2(ht(n->left), ht(n->right)); }

static Node *new_node(int k) {
    Node *n = malloc(sizeof(Node));
    n->key = k; n->left = n->right = NULL; n->height = 1;
    return n;
}

static Node *rot_right(Node *y) {
    Node *x = y->left, *T2 = x->right;
    x->right = y; y->left = T2;
    update_ht(y); update_ht(x);
    return x;
}
static Node *rot_left(Node *x) {
    Node *y = x->right, *T2 = y->left;
    y->left = x; x->right = T2;
    update_ht(x); update_ht(y);
    return y;
}

static int balance(Node *n) { return n ? ht(n->left) - ht(n->right) : 0; }

static Node *avl_insert(Node *node, int key) {
    if (!node) return new_node(key);
    if (key < node->key) node->left = avl_insert(node->left, key);
    else if (key > node->key) node->right = avl_insert(node->right, key);
    else return node;
    update_ht(node);
    int b = balance(node);
    if (b > 1 && key < node->left->key) return rot_right(node);
    if (b < -1 && key > node->right->key) return rot_left(node);
    if (b > 1 && key > node->left->key) { node->left = rot_left(node->left); return rot_right(node); }
    if (b < -1 && key < node->right->key) { node->right = rot_right(node->right); return rot_left(node); }
    return node;
}

static int in_order_check(Node *n, int *prev) {
    if (!n) return 1;
    if (!in_order_check(n->left, prev)) return 0;
    if (*prev >= n->key) return 0;
    *prev = n->key;
    return in_order_check(n->right, prev);
}

static void free_tree(Node *n) {
    if (!n) return;
    free_tree(n->left); free_tree(n->right); free(n);
}

int main(void) {
    Node *root = NULL;
    unsigned s = 0x12345678;
    for (int i = 0; i < 200; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        root = avl_insert(root, (int)(s % 1000));
    }
    int prev = -1;
    printf("avl_sorted=%d height=%d\n", in_order_check(root, &prev), ht(root));
    free_tree(root);
    return 0;
}
""",

"bitops": r"""
#include <stdio.h>
#include <stdint.h>

static int popcount32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0f0f0f0fu;
    return (int)((v * 0x01010101u) >> 24);
}

static int nlz32(uint32_t x) {
    if (!x) return 32;
    int n = 0;
    if (!(x >> 16)) { n += 16; x <<= 16; }
    if (!(x >> 24)) { n +=  8; x <<=  8; }
    if (!(x >> 28)) { n +=  4; x <<=  4; }
    if (!(x >> 30)) { n +=  2; x <<=  2; }
    if (!(x >> 31)) { n +=  1; }
    return n;
}

static uint32_t bit_reverse32(uint32_t v) {
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0f0f0f0fu) | ((v & 0x0f0f0f0fu) << 4);
    v = ((v >> 8) & 0x00ff00ffu) | ((v & 0x00ff00ffu) << 8);
    return (v >> 16) | (v << 16);
}

int main(void) {
    uint32_t s = 0xdeadbeef;
    long tot_pop = 0, tot_nlz = 0;
    for (int i = 0; i < 10000; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        tot_pop += popcount32(s);
        tot_nlz += nlz32(s);
        uint32_t rev = bit_reverse32(s);
        (void)rev;
    }
    printf("avg_popcount=%.2f avg_nlz=%.2f\n",
           (double)tot_pop / 10000.0, (double)tot_nlz / 10000.0);
    return 0;
}
""",

"primes": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int *sieve(int limit, int *out_count) {
    char *composite = calloc(limit + 1, 1);
    composite[0] = composite[1] = 1;
    for (int i = 2; i * i <= limit; i++)
        if (!composite[i])
            for (int j = i * i; j <= limit; j += i)
                composite[j] = 1;
    int cnt = 0;
    for (int i = 0; i <= limit; i++) if (!composite[i]) cnt++;
    int *primes = malloc(cnt * sizeof(int));
    int k = 0;
    for (int i = 0; i <= limit; i++) if (!composite[i]) primes[k++] = i;
    free(composite);
    *out_count = cnt;
    return primes;
}

static int miller_rabin(long long n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3 || n == 5 || n == 7) return 1;
    if (n % 2 == 0) return 0;
    long long d = n - 1; int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }
    long long witnesses[] = {2, 3, 5, 7, 11, 13};
    for (int i = 0; i < 6; i++) {
        long long a = witnesses[i];
        if (a >= n) continue;
        long long x = 1;
        long long base = a % n, exp = d;
        while (exp > 0) {
            if (exp & 1) x = (__int128)x * base % n;
            base = (__int128)base * base % n;
            exp >>= 1;
        }
        if (x == 1 || x == n - 1) continue;
        int composite = 1;
        for (int j = 0; j < r - 1; j++) {
            x = (__int128)x * x % n;
            if (x == n - 1) { composite = 0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

int main(void) {
    int cnt;
    int *primes = sieve(100000, &cnt);
    printf("primes_below_100000=%d\n", cnt);

    int agree = 0;
    for (int i = 0; i < cnt && primes[i] < 10000; i++)
        if (miller_rabin(primes[i])) agree++;
    printf("miller_rabin_agrees=%d\n", agree);
    free(primes);
    return 0;
}
""",

"graph": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXV 200
#define MAXE 1000

typedef struct { int u, v, w; } Edge;

static int parent[MAXV], rank_arr[MAXV];

static void dsu_init(int n) {
    for (int i = 0; i < n; i++) { parent[i] = i; rank_arr[i] = 0; }
}
static int dsu_find(int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}
static int dsu_union(int a, int b) {
    a = dsu_find(a); b = dsu_find(b);
    if (a == b) return 0;
    if (rank_arr[a] < rank_arr[b]) { int t = a; a = b; b = t; }
    parent[b] = a;
    if (rank_arr[a] == rank_arr[b]) rank_arr[a]++;
    return 1;
}

static int cmp_edge(const void *a, const void *b) {
    return ((Edge*)a)->w - ((Edge*)b)->w;
}

/* BFS shortest path in adjacency list */
static int adj[MAXV][MAXV], deg[MAXV];
static int dist[MAXV];
static int queue[MAXV];

static void bfs(int src, int n) {
    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[src] = 0;
    int head = 0, tail = 0;
    queue[tail++] = src;
    while (head < tail) {
        int u = queue[head++];
        for (int i = 0; i < deg[u]; i++) {
            int v = adj[u][i];
            if (dist[v] < 0) { dist[v] = dist[u] + 1; queue[tail++] = v; }
        }
    }
}

int main(void) {
    int V = 50, E_count = 0;
    Edge edges[MAXE];
    memset(adj, 0, sizeof(adj));
    memset(deg, 0, sizeof(deg));

    /* Build random graph */
    unsigned s = 0xabcdef01;
    for (int i = 0; i < 150 && E_count < MAXE; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        int u = s % V;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        int v = s % V;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        int w = (int)(s % 100) + 1;
        if (u != v) {
            edges[E_count++] = (Edge){u, v, w};
            if (deg[u] < MAXV - 1) adj[u][deg[u]++] = v;
            if (deg[v] < MAXV - 1) adj[v][deg[v]++] = u;
        }
    }

    /* Kruskal MST */
    qsort(edges, E_count, sizeof(Edge), cmp_edge);
    dsu_init(V);
    int mst_w = 0, mst_e = 0;
    for (int i = 0; i < E_count; i++) {
        if (dsu_union(edges[i].u, edges[i].v)) {
            mst_w += edges[i].w;
            mst_e++;
        }
    }

    /* BFS from vertex 0 */
    bfs(0, V);
    int reachable = 0;
    for (int i = 0; i < V; i++) if (dist[i] >= 0) reachable++;

    printf("mst_weight=%d mst_edges=%d reachable=%d\n", mst_w, mst_e, reachable);
    return 0;
}
""",

}

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _compile(src_path: Path, out_path: Path, cc: str, opt: str) -> bool:
    """Compile src_path to out_path.  Returns True on success."""
    extra = ['-lm'] if src_path.name in ('primes.c', 'matrix.c') else []
    cmd = [cc, opt, '-g0', str(src_path), '-o', str(out_path)] + extra
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            logger.warning(f"Compile failed ({' '.join(cmd)}): {result.stderr[:200]}")
            return False
        return True
    except FileNotFoundError:
        logger.warning(f"Compiler not found: {cc}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Compile timed out: {' '.join(cmd)}")
        return False
    except Exception as exc:
        logger.warning(f"Compile error: {exc}")
        return False


def build_all_binaries(compilers: List[str]) -> List[Binary]:
    """Write all C sources, compile under every (compiler, opt) pair, disassemble."""
    src_dir = COMPILED_DIR / 'src'
    bin_dir = COMPILED_DIR / 'bin'
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Write source files
    for prog_name, code in SOURCES.items():
        (src_dir / f'{prog_name}.c').write_text(code)

    binaries: List[Binary] = []
    total_attempts = len(SOURCES) * len(compilers) * len(OPT_LEVELS)
    logger.info(f"Building up to {total_attempts} binaries "
                f"({len(SOURCES)} programs × {len(compilers)} compilers × {len(OPT_LEVELS)} opt levels)")

    for prog_name in SOURCES:
        src_path = src_dir / f'{prog_name}.c'
        for cc in compilers:
            for opt in OPT_LEVELS:
                label = f'{cc}{opt}'          # e.g. gcc-O2
                bin_name = f'{prog_name}_{label}'
                out_path = bin_dir / bin_name
                logger.info(f"  Compiling {bin_name}")
                if not _compile(src_path, out_path, cc, opt):
                    continue
                binary = disassemble_binary(out_path, category=prog_name, compiler=label)
                if binary is None:
                    logger.warning(f"  Disassembly failed: {bin_name}")
                    continue
                binary.name = bin_name
                binaries.append(binary)

    logger.info(f"Successfully built {len(binaries)} / {total_attempts} binaries")
    return binaries


# ---------------------------------------------------------------------------
# Per-binary metric computation
# ---------------------------------------------------------------------------

def _zipf_alpha(binary: Binary) -> float:
    """Fit Zipf MLE to a single binary's opcode frequency distribution."""
    counts: Dict[str, int] = {}
    for op in binary.full_opcode_sequence:
        counts[op] = counts.get(op, 0) + 1
    if len(counts) < 5:
        return float('nan')
    rf = compute_rank_frequency(counts)
    result = fit_zipf_mle(rf, n_bootstrap=50, rng_seed=42)
    return float(result.get('alpha_mle', float('nan')))


def _h1(binary: Binary) -> float:
    """Unigram entropy."""
    counts: Dict[str, int] = {}
    for op in binary.full_opcode_sequence:
        counts[op] = counts.get(op, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return float('nan')
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _h2_rate(binary: Binary) -> float:
    """H2 / 2 entropy rate using a minimal single-binary wrapper."""
    class _Wrap:
        def __init__(self, b):
            self._b = b
        @property
        def full_opcode_sequence(self):
            return self._b.full_opcode_sequence

    rates = compute_entropy_rate([_Wrap(binary)], max_n=2)
    for entry in rates:
        if entry['n'] == 2:
            return float(entry['entropy_rate'])
    return float('nan')


def _zlib_ratio(binary: Binary) -> float:
    """Zlib compression ratio for a single binary."""
    seq = binary.full_opcode_sequence
    if not seq:
        return float('nan')
    # Simple byte encoding: map unique opcodes to 0..V-1
    unique = sorted(set(seq))
    idx = {op: i for i, op in enumerate(unique)}
    enc = [idx[op] for op in seq]
    try:
        if max(enc) > 255:
            raw = struct.pack(f'<{len(enc)}H', *enc)
        else:
            raw = bytes(enc)
    except (ValueError, struct.error):
        raw = bytes([x % 256 for x in enc])
    return len(zlib.compress(raw)) / len(raw)


def compute_per_binary_metrics(binaries: List[Binary]) -> List[Dict]:
    """Return a list of metric dicts, one per binary."""
    records = []
    for b in binaries:
        alpha = _zipf_alpha(b)
        h1 = _h1(b)
        h2r = _h2_rate(b)
        zr = _zlib_ratio(b)

        # Parse compiler label into cc and opt
        # binary.compiler is e.g. "gcc-O2" or "clang-Os"
        compiler_label = b.compiler or ''
        if compiler_label.startswith('clang'):
            cc = 'clang'
            opt = compiler_label[len('clang'):]
        elif compiler_label.startswith('gcc'):
            cc = 'gcc'
            opt = compiler_label[len('gcc'):]
        else:
            cc = 'unknown'
            opt = compiler_label

        records.append({
            'binary_name': b.name,
            'program': b.category or '',
            'compiler': cc,
            'opt_level': opt,
            'compiler_label': compiler_label,
            'instruction_count': b.instruction_count,
            'function_count': b.function_count,
            'zipf_alpha': alpha,
            'h1_bits': h1,
            'h2_rate_bits': h2r,
            'zlib_ratio': zr,
        })
    return records


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def _r2_categorical(y: np.ndarray, groups: List[str]) -> float:
    """
    Simple one-way ANOVA R² for a categorical grouping factor.

    R² = SS_between / SS_total.
    """
    if len(y) == 0 or len(set(groups)) < 2:
        return float('nan')

    grand_mean = float(np.mean(y))
    ss_total = float(np.sum((y - grand_mean) ** 2))
    if ss_total < 1e-15:
        return float('nan')

    unique_groups = list(set(groups))
    ss_between = 0.0
    for g in unique_groups:
        mask = np.array([g_ == g for g_ in groups])
        n_g = int(mask.sum())
        mean_g = float(np.mean(y[mask]))
        ss_between += n_g * (mean_g - grand_mean) ** 2

    return ss_between / ss_total


def variance_decomposition(records: List[Dict]) -> Dict:
    """
    For each metric decompose variance into program / compiler / opt_level.

    Uses one-way ANOVA R² for each factor independently (not ANCOVA).
    Reports also which factor explains the most variance (dominant factor)
    and whether compiler identity alone explains < 20% variance
    (compiler_invariant criterion).
    """
    metric_names = ['zipf_alpha', 'h1_bits', 'h2_rate_bits', 'zlib_ratio']
    programs = [r['program'] for r in records]
    compilers = [r['compiler'] for r in records]
    opt_levels = [r['opt_level'] for r in records]

    results = {}
    for metric in metric_names:
        values = [r[metric] for r in records]
        # Drop NaN rows
        valid = [(v, p, c, o) for v, p, c, o in zip(values, programs, compilers, opt_levels)
                 if math.isfinite(v)]
        if len(valid) < 4:
            results[metric] = {
                'program_r2': None,
                'compiler_r2': None,
                'opt_level_r2': None,
                'dominant_factor': None,
                'compiler_invariant': None,
                'n': 0,
            }
            continue

        y = np.array([v[0] for v in valid])
        p_grp = [v[1] for v in valid]
        c_grp = [v[2] for v in valid]
        o_grp = [v[3] for v in valid]

        r2_prog = _r2_categorical(y, p_grp)
        r2_comp = _r2_categorical(y, c_grp)
        r2_opt  = _r2_categorical(y, o_grp)

        factor_r2 = {
            'program': r2_prog if math.isfinite(r2_prog) else -1.0,
            'compiler': r2_comp if math.isfinite(r2_comp) else -1.0,
            'opt_level': r2_opt if math.isfinite(r2_opt) else -1.0,
        }
        dominant = max(factor_r2, key=lambda k: factor_r2[k])

        # "Compiler invariant" if compiler R² < 0.20
        comp_invariant = bool(math.isfinite(r2_comp) and r2_comp < 0.20)

        results[metric] = {
            'program_r2': round(r2_prog, 4) if math.isfinite(r2_prog) else None,
            'compiler_r2': round(r2_comp, 4) if math.isfinite(r2_comp) else None,
            'opt_level_r2': round(r2_opt, 4) if math.isfinite(r2_opt) else None,
            'dominant_factor': dominant,
            'compiler_invariant': comp_invariant,
            'n': len(valid),
        }

    return results


def group_means(records: List[Dict]) -> Dict:
    """Compute per-group mean for each metric."""
    metric_names = ['zipf_alpha', 'h1_bits', 'h2_rate_bits', 'zlib_ratio']
    out: Dict = {}

    for factor, key in [('program', 'by_program'),
                         ('compiler', 'by_compiler'),
                         ('opt_level', 'by_opt_level')]:
        out[key] = {}
        groups = sorted(set(r[factor] for r in records))
        for g in groups:
            sub = [r for r in records if r[factor] == g]
            out[key][g] = {}
            for m in metric_names:
                vals = [r[m] for r in sub if math.isfinite(r[m])]
                out[key][g][m] = round(float(np.mean(vals)), 4) if vals else None

    return out


# ---------------------------------------------------------------------------
# Key findings table
# ---------------------------------------------------------------------------

def print_findings_table(decomp: Dict) -> None:
    header = (
        f"{'metric':<18} {'prog_R2':>8} {'comp_R2':>8} {'opt_R2':>8} "
        f"{'dominant':<12} {'compiler_inv?':>13}"
    )
    print()
    print("=" * len(header))
    print("  Variance decomposition: metric ~ program + compiler + opt_level")
    print(header)
    print("-" * len(header))
    for metric, vals in decomp.items():
        p = f"{vals['program_r2']:.4f}" if vals['program_r2'] is not None else ' N/A '
        c = f"{vals['compiler_r2']:.4f}" if vals['compiler_r2'] is not None else ' N/A '
        o = f"{vals['opt_level_r2']:.4f}" if vals['opt_level_r2'] is not None else ' N/A '
        dom = vals['dominant_factor'] or '?'
        inv = 'YES' if vals['compiler_invariant'] else 'NO'
        print(f"{metric:<18} {p:>8} {c:>8} {o:>8} {dom:<12} {inv:>13}")
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== expanded_compiler_matrix.py ===")
    ensure_output_dir(OUTPUT_DIR)
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Find available compilers ───────────────────────────────────────
    compilers = _available_compilers()
    if not compilers:
        logger.error("No compilers found (need gcc or clang).  Aborting.")
        sys.exit(1)
    logger.info(f"Using compilers: {compilers}")

    # ── 2. Compile all programs ───────────────────────────────────────────
    binaries = build_all_binaries(compilers)
    if not binaries:
        logger.error("No binaries produced.  Aborting.")
        sys.exit(1)

    # Save corpus for reproducibility
    save_pickle(binaries, COMPILED_DIR / 'corpus.pkl')

    # ── 3. Per-binary metrics ─────────────────────────────────────────────
    logger.info("Computing per-binary metrics ...")
    records = compute_per_binary_metrics(binaries)

    # ── 4. Variance decomposition ─────────────────────────────────────────
    logger.info("Running variance decomposition ...")
    decomp = variance_decomposition(records)
    means = group_means(records)

    # ── 5. Print findings table ───────────────────────────────────────────
    print_findings_table(decomp)

    # ── 6. Findings narrative ─────────────────────────────────────────────
    # Identify the dominant factor across metrics
    factor_counts: Dict[str, int] = {'program': 0, 'compiler': 0, 'opt_level': 0}
    for m_vals in decomp.values():
        dom = m_vals.get('dominant_factor')
        if dom in factor_counts:
            factor_counts[dom] += 1

    overall_dominant = max(factor_counts, key=lambda k: factor_counts[k])
    n_compiler_invariant = sum(
        1 for m in decomp.values() if m.get('compiler_invariant')
    )

    findings = {
        'dominant_factor_by_metric_count': factor_counts,
        'overall_dominant_factor': overall_dominant,
        'n_metrics_compiler_invariant': n_compiler_invariant,
        'n_metrics_total': len(decomp),
        'interpretation': (
            f"The '{overall_dominant}' factor dominates in "
            f"{factor_counts[overall_dominant]}/{len(decomp)} metrics. "
            f"{n_compiler_invariant}/{len(decomp)} metrics are compiler-invariant "
            f"(compiler R² < 0.20)."
        ),
    }

    logger.info(findings['interpretation'])

    # ── 7. Save JSON ──────────────────────────────────────────────────────
    output = {
        'description': (
            'Expanded compiler matrix: '
            f'{len(SOURCES)} C programs × {len(compilers)} compilers × {len(OPT_LEVELS)} opt levels. '
            'Variance decomposition identifies which factor (program identity, '
            'compiler identity, optimisation level) explains most metric variation.'
        ),
        'compilers_used': compilers,
        'programs': list(SOURCES.keys()),
        'opt_levels': OPT_LEVELS,
        'n_binaries': len(binaries),
        'n_programs': len(SOURCES),
        'per_binary_records': records,
        'variance_decomposition': decomp,
        'group_means': means,
        'findings': findings,
    }

    def _json_safe(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(OUTPUT_JSON, 'w') as fh:
        json.dump(output, fh, indent=2, default=_json_safe)

    logger.info(f"Results written to {OUTPUT_JSON}")
    logger.info(
        f"Summary: {len(binaries)} binaries analysed, "
        f"dominant factor = {overall_dominant}, "
        f"compiler-invariant metrics = {n_compiler_invariant}/{len(decomp)}"
    )


if __name__ == '__main__':
    main()
