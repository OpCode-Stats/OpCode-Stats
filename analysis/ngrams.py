"""
N-gram analysis and entropy calculation module.
"""

import numpy as np
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json

logger = logging.getLogger(__name__)

def extract_ngrams(sequence: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a sequence."""
    if len(sequence) < n:
        return []
    
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngrams.append(tuple(sequence[i:i+n]))
    
    return ngrams

def compute_ngram_frequencies(binaries: List[Binary], n: int) -> Dict[Tuple[str, ...], int]:
    """Compute n-gram frequencies across all binaries."""
    ngram_counts = defaultdict(int)
    
    for binary in binaries:
        sequence = binary.full_opcode_sequence
        ngrams = extract_ngrams(sequence, n)
        
        for ngram in ngrams:
            ngram_counts[ngram] += 1
    
    return dict(ngram_counts)

def compute_entropy(frequencies: Dict[any, int]) -> float:
    """Compute Shannon entropy from frequency distribution."""
    if not frequencies:
        return 0.0
    
    total = sum(frequencies.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in frequencies.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    return entropy

def compute_conditional_entropy(bigram_counts: Dict[Tuple[str, str], int], 
                              unigram_counts: Dict[str, int]) -> float:
    """Compute conditional entropy H(X|Y) from bigram and unigram counts."""
    if not bigram_counts or not unigram_counts:
        return 0.0
    
    # Group bigrams by first element
    conditional_counts = defaultdict(dict)
    for (first, second), count in bigram_counts.items():
        conditional_counts[first][second] = count
    
    total_unigrams = sum(unigram_counts.values())
    conditional_entropy = 0.0
    
    for first_token, second_counts in conditional_counts.items():
        # Probability of first token
        p_first = unigram_counts.get(first_token, 0) / total_unigrams
        
        if p_first > 0:
            # Conditional entropy for this first token
            total_second = sum(second_counts.values())
            h_conditional = 0.0
            
            for count in second_counts.values():
                if count > 0:
                    p_second_given_first = count / total_second
                    h_conditional -= p_second_given_first * np.log2(p_second_given_first)
            
            conditional_entropy += p_first * h_conditional
    
    return conditional_entropy

def analyze_ngrams_for_n(binaries: List[Binary], n: int) -> Dict:
    """Analyze n-grams for a specific value of n."""
    logger.info(f"Computing {n}-grams...")
    
    # Compute n-gram frequencies
    ngram_counts = compute_ngram_frequencies(binaries, n)
    
    if not ngram_counts:
        logger.warning(f"No {n}-grams found")
        return {"n": n, "count": 0, "entropy": 0.0, "top_ngrams": []}
    
    # Sort by frequency
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Compute entropy
    entropy = compute_entropy(ngram_counts)
    
    # Theoretical maximum entropy (log2 of vocabulary size)
    vocab_size = len(set(token for ngram in ngram_counts.keys() for token in ngram))
    max_entropy = np.log2(vocab_size) if vocab_size > 0 else 0.0
    
    # Top n-grams
    top_ngrams = []
    for ngram, count in sorted_ngrams[:50]:
        ngram_str = " ".join(ngram)
        total_ngrams = sum(ngram_counts.values())
        frequency = count / total_ngrams if total_ngrams > 0 else 0.0
        
        top_ngrams.append({
            "ngram": ngram_str,
            "count": count,
            "frequency": frequency
        })
    
    return {
        "n": n,
        "total_ngrams": len(ngram_counts),
        "unique_ngrams": len(ngram_counts),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "top_ngrams": top_ngrams
    }

def compute_entropy_rate(binaries: List[Binary], max_n: int = 5) -> List[Dict]:
    """Compute entropy rate for n-grams up to max_n.

    Returns per-n dictionaries with:
      - entropy_rate: H_n / n  (the standard entropy rate)
      - conditional_entropy: H_n - H_{n-1}  (new information from the n-th token)
    """
    logger.info(f"Computing entropy rates for n=1 to {max_n}")

    entropy_data = []
    previous_entropy = 0.0

    for n in range(1, max_n + 1):
        ngram_analysis = analyze_ngrams_for_n(binaries, n)
        current_entropy = ngram_analysis["entropy"]

        # Conditional entropy: H_n - H_{n-1}
        conditional_entropy = current_entropy - previous_entropy if n > 1 else current_entropy
        # Entropy rate: H_n / n
        entropy_rate = current_entropy / n

        entropy_data.append({
            "n": n,
            "entropy": current_entropy,
            "entropy_rate": entropy_rate,
            "conditional_entropy": conditional_entropy,
            "unique_ngrams": ngram_analysis["unique_ngrams"],
            "max_entropy": ngram_analysis["max_entropy"],
            "normalized_entropy": ngram_analysis["normalized_entropy"]
        })

        previous_entropy = current_entropy

    return entropy_data

def compute_shuffled_entropy_rates(binaries: List[Binary], max_n: int = 5,
                                   seed: int = 42) -> List[Dict]:
    """Compute entropy rates on shuffled sequences as a structural baseline.

    Shuffling preserves the unigram distribution but destroys all higher-order
    structure.  The gap between real and shuffled entropy rates at n>1 measures
    how much context information exists beyond unigram statistics.
    """
    import random as _random
    rng = _random.Random(seed)

    # Build shuffled copies of every binary's opcode sequence
    class _ShuffledBinary:
        def __init__(self, seq):
            self._seq = seq
        @property
        def full_opcode_sequence(self):
            return self._seq

    shuffled_binaries = []
    for b in binaries:
        seq = list(b.full_opcode_sequence)
        rng.shuffle(seq)
        shuffled_binaries.append(_ShuffledBinary(seq))

    return compute_entropy_rate(shuffled_binaries, max_n)


def compute_per_binary_entropy_rates(binaries: List[Binary], max_n: int = 5) -> Dict[str, List[Dict]]:
    """Compute entropy rates independently for each binary."""
    results = {}
    for binary in binaries:
        class _Single:
            def __init__(self, b): self._b = b
            @property
            def full_opcode_sequence(self): return self._b.full_opcode_sequence
        results[binary.name] = compute_entropy_rate([_Single(binary)], max_n)
    return results


def run_ngram_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete n-gram analysis."""
    logger.info("Running n-gram analysis...")
    
    # Compute entropy rates
    entropy_rates = compute_entropy_rate(binaries, max_n=5)

    # Shuffled baseline — shows structure beyond unigram statistics
    shuffled_rates = compute_shuffled_entropy_rates(binaries, max_n=5)

    # Per-binary entropy rates
    per_binary_rates = compute_per_binary_entropy_rates(binaries, max_n=5)

    # Detailed analysis for each n
    detailed_analysis = {}
    for n in range(2, 6):  # 2-grams to 5-grams
        detailed_analysis[f"{n}gram"] = analyze_ngrams_for_n(binaries, n)
        
        # Save top n-grams to separate file
        ngram_data = detailed_analysis[f"{n}gram"]
        save_json(ngram_data["top_ngrams"], output_dir / f"top_{n}grams.json")
    
    # Compile results
    results = {
        "entropy_analysis": {
            "entropy_rates": entropy_rates,
            "shuffled_baseline_rates": shuffled_rates,
            "entropy_gain_over_shuffled": [
                {
                    "n": real["n"],
                    "real_entropy": real["entropy"],
                    "shuffled_entropy": shuf["entropy"],
                    "gain_bits": real["entropy"] - shuf["entropy"],
                }
                for real, shuf in zip(entropy_rates, shuffled_rates)
            ],
            "per_binary_entropy_rates": per_binary_rates,
            "predictability_analysis": analyze_predictability(entropy_rates),
        },
        "detailed_ngram_analysis": detailed_analysis,
        "summary": {
            "most_predictable_n": get_most_predictable_n(entropy_rates),
            "entropy_decay": analyze_entropy_decay(entropy_rates)
        }
    }
    
    # Save results
    save_json(results, output_dir / "ngram_analysis.json")
    
    logger.info("N-gram analysis completed")
    return results

def analyze_predictability(entropy_rates: List[Dict]) -> Dict:
    """Analyze how predictable the sequences are."""
    if len(entropy_rates) < 2:
        return {"interpretation": "Insufficient data for predictability analysis"}
    
    # Get the entropy rate (how much new information each additional token provides)
    final_entropy_rate = entropy_rates[-1]["entropy_rate"]
    
    # Compare to theoretical maximum
    vocab_sizes = [er["max_entropy"] for er in entropy_rates if er["max_entropy"] > 0]
    avg_vocab_entropy = np.mean(vocab_sizes) if vocab_sizes else 0.0
    
    predictability_ratio = final_entropy_rate / avg_vocab_entropy if avg_vocab_entropy > 0 else 0.0
    
    interpretation = ""
    if predictability_ratio < 0.1:
        interpretation = "Extremely predictable - very low entropy rate"
    elif predictability_ratio < 0.3:
        interpretation = "Highly predictable - sequences are very constrained"
    elif predictability_ratio < 0.6:
        interpretation = "Moderately predictable - some structure but not rigid"
    elif predictability_ratio < 0.8:
        interpretation = "Somewhat predictable - moderate structure"
    else:
        interpretation = "Low predictability - approaching random sequences"
    
    return {
        "final_entropy_rate": final_entropy_rate,
        "predictability_ratio": predictability_ratio,
        "interpretation": interpretation
    }

def get_most_predictable_n(entropy_rates: List[Dict]) -> Dict:
    """Find the n-gram length with lowest entropy rate."""
    if not entropy_rates:
        return {"n": 0, "entropy_rate": 0.0}
    
    min_rate_entry = min(entropy_rates[1:], key=lambda x: x["entropy_rate"]) if len(entropy_rates) > 1 else entropy_rates[0]
    
    return {
        "n": min_rate_entry["n"],
        "entropy_rate": min_rate_entry["entropy_rate"],
        "interpretation": f"Context of {min_rate_entry['n']-1} previous tokens provides maximum predictability"
    }

def analyze_entropy_decay(entropy_rates: List[Dict]) -> Dict:
    """Analyze how entropy rate decays with increasing context length."""
    if len(entropy_rates) < 3:
        return {"interpretation": "Insufficient data for decay analysis"}
    
    rates = [er["entropy_rate"] for er in entropy_rates[1:]]  # Skip n=1
    
    # Simple decay analysis
    decay_rate = (rates[0] - rates[-1]) / rates[0] if rates[0] > 0 else 0.0
    
    interpretation = ""
    if decay_rate > 0.7:
        interpretation = "Rapid entropy decay - long-range dependencies"
    elif decay_rate > 0.4:
        interpretation = "Moderate entropy decay - some long-range structure"
    elif decay_rate > 0.1:
        interpretation = "Slow entropy decay - mostly local dependencies"
    else:
        interpretation = "Minimal entropy decay - little benefit from longer context"
    
    return {
        "decay_rate": decay_rate,
        "interpretation": interpretation,
        "rates_by_n": {f"{i+2}gram": rate for i, rate in enumerate(rates)}
    }