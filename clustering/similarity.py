"""
N-gram based similarity analysis using TF-IDF and cosine similarity.

This provides an alternative to compression-based similarity that captures
semantic relationships through shared n-gram patterns.
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json
from analysis.ngrams import extract_ngrams

logger = logging.getLogger(__name__)

class NgramSimilarityAnalyzer:
    """Analyzer for n-gram based binary similarity using L1-normalised term frequency."""

    def __init__(self, n: int = 3, max_features: int = 10000):
        """
        Initialize n-gram similarity analyzer.

        Args:
            n: N-gram length (default: trigrams)
            max_features: Maximum number of n-grams to consider

        Note: uses L1-normalised raw term frequency (no IDF weighting, min_df=1)
        so that every n-gram observed in at least one binary is retained.
        IDF down-weights common n-grams which is appropriate for large corpora,
        but on small corpora (< ~30 documents) it distorts distances and causes
        many n-grams to be dropped by min_df thresholds.
        """
        self.n = n
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.feature_names = None
    
    def _extract_ngram_documents(self, binaries: List[Binary]) -> List[str]:
        """Convert each binary's opcode sequence to n-gram document."""
        documents = []
        
        for binary in binaries:
            sequence = binary.full_opcode_sequence
            
            if len(sequence) < self.n:
                logger.warning(f"Binary {binary.name} too short for {self.n}-grams")
                documents.append("")
                continue
            
            # Extract n-grams; join opcodes with '_' so each n-gram becomes
            # a single token (e.g. "mov_lea_call") that the vectorizer
            # tokenizes atomically.  Space-joining within n-grams caused the
            # greedy multi-word token_pattern to swallow the entire document
            # as a single feature.
            ngrams = extract_ngrams(sequence, self.n)
            ngram_strings = ['_'.join(ngram) for ngram in ngrams]
            document = ' '.join(ngram_strings)
            documents.append(document)
        
        return documents
    
    def compute_tfidf_matrix(self, binaries: List[Binary]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute TF-IDF matrix for binary n-grams.
        
        Returns:
            tfidf_matrix: Shape (n_binaries, n_features)
            feature_names: List of n-gram features
        """
        logger.info(f"Computing TF-IDF matrix for {len(binaries)} binaries with {self.n}-grams")
        
        # Convert binaries to n-gram documents
        documents = self._extract_ngram_documents(binaries)
        
        # L1-normalised term frequency: use_idf=False + norm='l1', min_df=1
        # This is equivalent to normalised n-gram frequency histograms and
        # works correctly on corpora as small as 2 documents.
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=1,
            use_idf=False,
            norm='l1',
            ngram_range=(1, 1),   # n-grams already encoded as single tokens
            token_pattern=r'(?u)\b[\w]+(?:_[\w]+)*\b',  # matches "mov_lea_call"
        )
        
        try:
            # Fit and transform documents
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
            
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            logger.info(f"Number of unique {self.n}-grams: {len(self.feature_names)}")
            
            return tfidf_matrix.toarray(), self.feature_names
            
        except Exception as e:
            logger.error(f"TF-IDF computation failed: {e}")
            # Return empty matrix as fallback
            return np.zeros((len(binaries), 1)), ["empty"]
    
    def compute_cosine_similarity_matrix(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if tfidf_matrix.shape[1] == 1 and tfidf_matrix.sum() == 0:
            # Fallback for empty matrix
            n = tfidf_matrix.shape[0]
            return np.zeros((n, n))
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        logger.info(f"Cosine similarity matrix computed: mean={np.mean(similarity_matrix):.3f}")
        
        return similarity_matrix
    
    def get_top_features_per_binary(self, tfidf_matrix: np.ndarray, binaries: List[Binary], 
                                   top_k: int = 10) -> Dict[str, List[Dict]]:
        """Get top TF-IDF features for each binary."""
        if not self.feature_names:
            return {}
        
        binary_features = {}
        
        for i, binary in enumerate(binaries):
            # Get TF-IDF scores for this binary
            scores = tfidf_matrix[i]
            
            # Get top features
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_features = []
            
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    top_features.append({
                        'ngram': self.feature_names[idx],
                        'tfidf_score': float(scores[idx])
                    })
            
            binary_features[binary.name] = top_features
        
        return binary_features

def run_ngram_similarity_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete n-gram similarity analysis."""
    logger.info("Running n-gram similarity analysis...")
    
    if len(binaries) < 2:
        logger.warning("Need at least 2 binaries for similarity analysis")
        return {}
    
    results = {}
    
    # Analyze with different n-gram lengths
    for n in [2, 3, 4]:
        logger.info(f"Computing similarity with {n}-grams...")
        
        try:
            analyzer = NgramSimilarityAnalyzer(n=n, max_features=10000)
            tfidf_matrix, feature_names = analyzer.compute_tfidf_matrix(binaries)
            
            if tfidf_matrix.shape[1] <= 1:
                logger.warning(f"Insufficient n-grams for {n}-gram analysis")
                continue
            
            # Compute similarity matrix
            similarity_matrix = analyzer.compute_cosine_similarity_matrix(tfidf_matrix)
            
            # Convert similarities to distances (1 - similarity)
            distance_matrix = 1.0 - similarity_matrix
            
            # Compute statistics
            upper_triangle = distance_matrix[np.triu_indices(len(binaries), k=1)]
            
            # Get top features per binary
            top_features = analyzer.get_top_features_per_binary(tfidf_matrix, binaries, top_k=5)
            
            ngram_results = {
                'n': n,
                'similarity_matrix': similarity_matrix.tolist(),
                'distance_matrix': distance_matrix.tolist(),
                'binary_names': [b.name for b in binaries],
                'feature_count': len(feature_names),
                'statistics': {
                    'mean_similarity': float(np.mean(similarity_matrix)),
                    'std_similarity': float(np.std(similarity_matrix)),
                    'mean_distance': float(np.mean(upper_triangle)),
                    'std_distance': float(np.std(upper_triangle)),
                    'min_distance': float(np.min(upper_triangle)),
                    'max_distance': float(np.max(upper_triangle))
                },
                'most_similar_pairs': get_most_similar_pairs_cosine(similarity_matrix, binaries, top_k=5),
                'most_different_pairs': get_most_different_pairs_cosine(similarity_matrix, binaries, top_k=5),
                'top_features_per_binary': top_features
            }
            
            results[f'{n}gram'] = ngram_results
            
            # Save individual results
            save_json(ngram_results, output_dir / f"similarity_{n}gram.json")
            
        except Exception as e:
            logger.error(f"N-gram similarity analysis with {n}-grams failed: {e}")
            results[f'{n}gram'] = {'error': str(e)}
    
    # Find best n-gram length
    if len(results) > 1:
        results['best_ngram_analysis'] = find_best_ngram_length(results)
    
    # Save combined results
    save_json(results, output_dir / "ngram_similarity.json")
    
    logger.info("N-gram similarity analysis completed")
    return results

def get_most_similar_pairs_cosine(similarity_matrix: np.ndarray, binaries: List[Binary], 
                                 top_k: int = 5) -> List[Dict]:
    """Find most similar binary pairs using cosine similarity."""
    n = len(binaries)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'binary1': binaries[i].name,
                'binary2': binaries[j].name,
                'similarity': float(similarity_matrix[i, j]),
                'distance': 1.0 - similarity_matrix[i, j]
            })
    
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return pairs[:top_k]

def get_most_different_pairs_cosine(similarity_matrix: np.ndarray, binaries: List[Binary], 
                                   top_k: int = 5) -> List[Dict]:
    """Find most different binary pairs using cosine similarity."""
    n = len(binaries)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'binary1': binaries[i].name,
                'binary2': binaries[j].name,
                'similarity': float(similarity_matrix[i, j]),
                'distance': 1.0 - similarity_matrix[i, j]
            })
    
    pairs.sort(key=lambda x: x['similarity'])
    return pairs[:top_k]

def find_best_ngram_length(results: Dict) -> Dict:
    """Find the n-gram length that provides best separation."""
    best_n = None
    best_score = 0.0
    
    for key, result in results.items():
        if key.endswith('gram') and 'error' not in result:
            # Use standard deviation of distances as separation metric
            std_distance = result['statistics']['std_distance']
            mean_distance = result['statistics']['mean_distance']
            
            # Good separation = high std deviation relative to mean
            separation_score = std_distance / max(mean_distance, 0.001)
            
            if separation_score > best_score:
                best_score = separation_score
                best_n = result['n']
    
    return {
        'best_n': best_n,
        'separation_score': best_score,
        'interpretation': f"{best_n}-grams provide the best binary separation" if best_n else "No clear best n-gram length"
    }