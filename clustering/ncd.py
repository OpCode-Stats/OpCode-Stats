"""
Normalized Compression Distance (NCD) implementation for binary similarity.

NCD uses compression to measure similarity between sequences by computing:
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

Where C(x) is the compressed size of sequence x.
"""

import zlib
import lzma
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from scipy.stats import mannwhitneyu
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, build_vocabulary, encode_sequence, save_json

logger = logging.getLogger(__name__)

class NCDCalculator:
    """Calculator for Normalized Compression Distance between binaries."""
    
    def __init__(self, compressor: str = 'zlib', max_sequence_length: int = 50000):
        """
        Initialize NCD calculator.
        
        Args:
            compressor: 'zlib' or 'lzma' - compression algorithm to use
            max_sequence_length: Maximum sequence length for performance
        """
        self.compressor = compressor
        self.max_sequence_length = max_sequence_length
        self.compression_cache = {}  # Cache compressed sizes
        
        if compressor == 'zlib':
            self.compress_func = zlib.compress
        elif compressor == 'lzma':
            self.compress_func = lzma.compress
        else:
            raise ValueError(f"Unknown compressor: {compressor}")
    
    def _encode_binary_sequence(self, binary: Binary, vocab: Dict[str, int]) -> bytes:
        """Encode binary's opcode sequence as bytes."""
        opcodes = binary.full_opcode_sequence
        
        # Sample if too long
        if len(opcodes) > self.max_sequence_length:
            logger.debug(f"Sampling {binary.name}: {len(opcodes)} -> {self.max_sequence_length}")
            step = len(opcodes) // self.max_sequence_length
            opcodes = opcodes[::step][:self.max_sequence_length]
        
        # Encode to integers
        encoded = encode_sequence(opcodes, vocab)
        
        # Convert to bytes (handle large vocabularies)
        if max(encoded) < 256:
            return bytes(encoded)
        else:
            # Use 16-bit encoding for large vocabularies
            import struct
            return struct.pack(f'<{len(encoded)}H', *encoded)
    
    def _get_compressed_size(self, data: bytes, label: str = "") -> int:
        """Get compressed size with caching."""
        data_hash = hash(data)
        cache_key = (data_hash, self.compressor)
        
        if cache_key in self.compression_cache:
            return self.compression_cache[cache_key]
        
        try:
            compressed = self.compress_func(data)
            size = len(compressed)
            self.compression_cache[cache_key] = size
            logger.debug(f"Compressed {label}: {len(data)} -> {size} bytes ({size/len(data):.3f} ratio)")
            return size
        except Exception as e:
            logger.warning(f"Compression failed for {label}: {e}")
            return len(data)  # Fallback to original size
    
    def compute_ncd(self, binary1: Binary, binary2: Binary, vocab: Dict[str, int]) -> float:
        """
        Compute Normalized Compression Distance between two binaries.
        
        Returns value between 0 (identical) and 1 (completely different).
        """
        # Encode sequences
        seq1_bytes = self._encode_binary_sequence(binary1, vocab)
        seq2_bytes = self._encode_binary_sequence(binary2, vocab)
        
        # Get individual compressed sizes
        c1 = self._get_compressed_size(seq1_bytes, binary1.name)
        c2 = self._get_compressed_size(seq2_bytes, binary2.name)
        
        # Concatenate and compress
        combined_bytes = seq1_bytes + seq2_bytes
        c12 = self._get_compressed_size(combined_bytes, f"{binary1.name}+{binary2.name}")
        
        # Compute NCD
        min_c = min(c1, c2)
        max_c = max(c1, c2)
        
        if max_c == 0:
            return 0.0
        
        ncd = (c12 - min_c) / max_c
        
        # Clamp to valid range [0, 1]
        ncd = max(0.0, min(1.0, ncd))
        
        logger.debug(f"NCD({binary1.name}, {binary2.name}) = {ncd:.3f}")
        return ncd
    
    def compute_ncd_matrix(self, binaries: List[Binary]) -> np.ndarray:
        """Compute pairwise NCD matrix for all binaries."""
        n = len(binaries)
        ncd_matrix = np.zeros((n, n))
        
        # Build vocabulary for encoding
        vocab = build_vocabulary(binaries)
        logger.info(f"Computing NCD matrix for {n} binaries using {self.compressor}")
        
        # Compute upper triangle (matrix is symmetric)
        for i in range(n):
            for j in range(i + 1, n):
                ncd = self.compute_ncd(binaries[i], binaries[j], vocab)
                ncd_matrix[i, j] = ncd
                ncd_matrix[j, i] = ncd  # Symmetric
        
        logger.info(f"NCD matrix computed: avg={np.mean(ncd_matrix[np.triu_indices(n, k=1)]):.3f}")
        return ncd_matrix

def evaluate_category_separation(ncd_matrix: np.ndarray, binaries: List[Binary]) -> Dict:
    """
    Within-vs-between category Mann-Whitney U test on NCD distances.

    Splits all pairwise distances into two groups:
      within  — both binaries share the same category label
      between — binaries have different category labels (or one has None)

    Returns the one-sided Mann-Whitney U statistic and p-value testing
    H1: within-category distances are stochastically smaller than between-category
    distances, plus rank-biserial effect size r = 1 - 2U/(n_within * n_between),
    per-category mean distances, and an interpretation string.

    Binaries with category=None are excluded from both groups.
    """
    names = [b.name for b in binaries]
    cats  = [b.category for b in binaries]
    n = len(binaries)

    within_dists:  list = []
    between_dists: list = []
    per_category: dict = {}

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cats[i], cats[j]
            if ci is None or cj is None:
                continue
            d = float(ncd_matrix[i, j])
            if ci == cj:
                within_dists.append(d)
                per_category.setdefault(ci, []).append(d)
            else:
                between_dists.append(d)

    result: Dict = {
        'n_within_pairs':  len(within_dists),
        'n_between_pairs': len(between_dists),
        'mean_within':  float(np.mean(within_dists))  if within_dists  else None,
        'mean_between': float(np.mean(between_dists)) if between_dists else None,
        'per_category_mean': {
            cat: float(np.mean(dists)) for cat, dists in per_category.items()
        },
    }

    if len(within_dists) < 2 or len(between_dists) < 2:
        result['note'] = 'Insufficient pairs for significance test'
        return result

    stat, pval = mannwhitneyu(within_dists, between_dists, alternative='less')
    n1, n2 = len(within_dists), len(between_dists)
    # rank-biserial correlation (effect size): r = 1 - 2U / (n1*n2)
    r = float(1.0 - 2.0 * stat / (n1 * n2))

    # --- Permutation test (accounts for non-independence of pairwise distances) ---
    observed_diff = float(np.mean(between_dists) - np.mean(within_dists))
    n_perm = 10000
    rng = np.random.default_rng(42)
    cat_arr = np.array(cats)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(cat_arr)
        w, b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] is None or perm[j] is None:
                    continue
                d = float(ncd_matrix[i, j])
                if perm[i] == perm[j]:
                    w.append(d)
                else:
                    b.append(d)
        if w and b:
            if (np.mean(b) - np.mean(w)) >= observed_diff:
                count_ge += 1
    perm_p = (count_ge + 1) / (n_perm + 1)

    result.update({
        'mannwhitney_U':  float(stat),
        'p_value':        float(pval),
        'p_value_note':   'Mann-Whitney treats pairwise distances as independent; see perm_p_value',
        'perm_p_value':   float(perm_p),
        'effect_size_r':  r,
        'significant':    bool(perm_p < 0.05),
        'interpretation': (
            f"Within-category NCD (mean {result['mean_within']:.3f}) is "
            f"{'significantly' if perm_p < 0.05 else 'not significantly'} smaller "
            f"than between-category NCD (mean {result['mean_between']:.3f}), "
            f"permutation p={perm_p:.4f}, MW p={pval:.4f}, r={r:.3f}."
        ),
    })

    logger.info(result['interpretation'])
    return result


def run_ncd_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete NCD analysis with multiple compressors."""
    logger.info("Running Normalized Compression Distance analysis...")
    
    if len(binaries) < 2:
        logger.warning("Need at least 2 binaries for NCD analysis")
        return {}
    
    results = {}
    
    # Analyze with different compressors
    for compressor in ['zlib', 'lzma']:
        logger.info(f"Computing NCD with {compressor} compressor...")
        
        try:
            calculator = NCDCalculator(compressor=compressor)
            ncd_matrix = calculator.compute_ncd_matrix(binaries)
            
            # Compute statistics
            upper_triangle = ncd_matrix[np.triu_indices(len(binaries), k=1)]
            
            category_sep = evaluate_category_separation(ncd_matrix, binaries)

            compressor_results = {
                'matrix': ncd_matrix.tolist(),
                'binary_names': [b.name for b in binaries],
                'statistics': {
                    'mean_distance': float(np.mean(upper_triangle)),
                    'std_distance': float(np.std(upper_triangle)),
                    'min_distance': float(np.min(upper_triangle)),
                    'max_distance': float(np.max(upper_triangle)),
                    'median_distance': float(np.median(upper_triangle))
                },
                'category_separation': category_sep,
                'most_similar_pairs': get_most_similar_pairs(ncd_matrix, binaries, top_k=5),
                'most_different_pairs': get_most_different_pairs(ncd_matrix, binaries, top_k=5)
            }
            
            results[compressor] = compressor_results
            
            # Save individual results
            save_json(compressor_results, output_dir / f"ncd_{compressor}.json")
            
        except Exception as e:
            logger.error(f"NCD analysis with {compressor} failed: {e}")
            results[compressor] = {'error': str(e)}
    
    # Compare compressors
    if 'zlib' in results and 'lzma' in results:
        results['compressor_comparison'] = compare_compressors(results['zlib'], results['lzma'])
    
    # Save combined results
    save_json(results, output_dir / "ncd_analysis.json")
    
    logger.info("NCD analysis completed")
    return results

def get_most_similar_pairs(ncd_matrix: np.ndarray, binaries: List[Binary], top_k: int = 5) -> List[Dict]:
    """Find the most similar binary pairs."""
    n = len(binaries)
    pairs = []
    
    # Get upper triangle indices (avoid diagonal and duplicates)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'binary1': binaries[i].name,
                'binary2': binaries[j].name,
                'distance': float(ncd_matrix[i, j]),
                'similarity': 1.0 - ncd_matrix[i, j]
            })
    
    # Sort by distance (ascending - most similar first)
    pairs.sort(key=lambda x: x['distance'])
    return pairs[:top_k]

def get_most_different_pairs(ncd_matrix: np.ndarray, binaries: List[Binary], top_k: int = 5) -> List[Dict]:
    """Find the most different binary pairs."""
    n = len(binaries)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'binary1': binaries[i].name,
                'binary2': binaries[j].name,
                'distance': float(ncd_matrix[i, j]),
                'similarity': 1.0 - ncd_matrix[i, j]
            })
    
    # Sort by distance (descending - most different first)  
    pairs.sort(key=lambda x: x['distance'], reverse=True)
    return pairs[:top_k]

def compare_compressors(zlib_results: Dict, lzma_results: Dict) -> Dict:
    """Compare NCD results from different compressors."""
    if 'error' in zlib_results or 'error' in lzma_results:
        return {'error': 'One or both compressors failed'}
    
    zlib_matrix = np.array(zlib_results['matrix'])
    lzma_matrix = np.array(lzma_results['matrix'])
    
    # Compute correlation between distance matrices
    zlib_upper = zlib_matrix[np.triu_indices(len(zlib_matrix), k=1)]
    lzma_upper = lzma_matrix[np.triu_indices(len(lzma_matrix), k=1)]
    
    correlation = np.corrcoef(zlib_upper, lzma_upper)[0, 1]
    
    comparison = {
        'correlation': float(correlation),
        'zlib_mean_distance': zlib_results['statistics']['mean_distance'],
        'lzma_mean_distance': lzma_results['statistics']['mean_distance'],
        'distance_difference': abs(zlib_results['statistics']['mean_distance'] - 
                                 lzma_results['statistics']['mean_distance']),
        'interpretation': interpret_compressor_comparison(correlation)
    }
    
    return comparison

def interpret_compressor_comparison(correlation: float) -> str:
    """Interpret the correlation between different compressors."""
    if correlation > 0.9:
        return "Very high agreement between compressors - robust similarity measure"
    elif correlation > 0.7:
        return "High agreement between compressors - reliable similarity measure"
    elif correlation > 0.5:
        return "Moderate agreement between compressors - some consistency"
    else:
        return "Low agreement between compressors - results may be compressor-specific"