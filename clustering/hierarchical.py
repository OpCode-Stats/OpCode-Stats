"""
Hierarchical clustering analysis for binary similarity matrices.

Performs agglomerative clustering and generates dendrograms to reveal
the structure of binary relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json

logger = logging.getLogger(__name__)


def compute_silhouette_score(distance_matrix: np.ndarray,
                             labels: np.ndarray) -> Optional[float]:
    """Compute silhouette score using a precomputed distance matrix."""
    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        return None
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(distance_matrix, labels, metric='precomputed'))
    except Exception as e:
        logger.warning(f"Silhouette score computation failed: {e}")
        return None


def compute_category_purity(cluster_labels: np.ndarray, binary_names: List[str],
                            categories: Dict[str, str]) -> Dict:
    """
    Compute category purity: for each cluster, what fraction of members share
    the majority category?  Mean purity across all clusters is the overall score.
    """
    from collections import Counter

    cluster_cats: Dict[int, List[str]] = {}
    for label, name in zip(cluster_labels, binary_names):
        cat = categories.get(name, 'unknown')
        cluster_cats.setdefault(int(label), []).append(cat)

    per_cluster: Dict[str, float] = {}
    purities: List[float] = []
    for label, cats in cluster_cats.items():
        majority = Counter(cats).most_common(1)[0][1]
        purity = majority / len(cats)
        purities.append(purity)
        per_cluster[f'cluster_{label}'] = round(purity, 4)

    mean_purity = float(np.mean(purities)) if purities else 0.0
    level = 'High' if mean_purity > 0.8 else ('Medium' if mean_purity > 0.5 else 'Low')
    return {
        'mean_purity': mean_purity,
        'per_cluster_purity': per_cluster,
        'interpretation': f'{level} category purity ({mean_purity:.2%} mean)'
    }

class HierarchicalClusterer:
    """Hierarchical clustering analyzer for binary distance matrices."""
    
    def __init__(self, linkage_method: str = 'ward', distance_threshold: float = 0.5):
        """
        Initialize hierarchical clusterer.
        
        Args:
            linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single')
            distance_threshold: Threshold for cutting dendrogram into clusters
        """
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
        self.linkage_matrix = None
        self.cluster_labels = None
    
    def fit_clustering(self, distance_matrix: np.ndarray, binary_names: List[str],
                       categories: Optional[Dict[str, str]] = None) -> Dict:
        """
        Perform hierarchical clustering on distance matrix.

        Args:
            distance_matrix: Square symmetric distance matrix
            binary_names: Names corresponding to matrix rows/columns
            categories: Optional name→category mapping for purity computation

        Returns:
            Dictionary with clustering results
        """
        n = len(binary_names)
        
        if n < 2:
            logger.warning("Need at least 2 binaries for clustering")
            return {}
        
        logger.info(f"Performing hierarchical clustering with {self.linkage_method} linkage")
        
        # Convert distance matrix to condensed form for scipy
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        try:
            self.linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
        except ValueError as e:
            logger.error(f"Clustering failed: {e}")
            # Try with different method as fallback
            logger.info("Trying with 'complete' linkage as fallback")
            self.linkage_matrix = linkage(condensed_distances, method='complete')
        
        # Cut tree to get cluster labels
        self.cluster_labels = fcluster(self.linkage_matrix, 
                                      t=self.distance_threshold, 
                                      criterion='distance')
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(binary_names)

        # Quality metrics
        silhouette = compute_silhouette_score(distance_matrix, self.cluster_labels)
        purity = (compute_category_purity(self.cluster_labels, binary_names, categories)
                  if categories else None)

        results = {
            'linkage_method': self.linkage_method,
            'distance_threshold': self.distance_threshold,
            'linkage_matrix': self.linkage_matrix.tolist(),
            'cluster_labels': self.cluster_labels.tolist(),
            'binary_names': binary_names,
            'cluster_analysis': cluster_analysis,
            'cophenetic_correlation': self._compute_cophenetic_correlation(distance_matrix),
            'dendrogram_info': self._get_dendrogram_info(),
            'silhouette_score': silhouette,
            'category_purity': purity
        }
        
        return results
    
    def _analyze_clusters(self, binary_names: List[str]) -> Dict:
        """Analyze the resulting clusters."""
        if self.cluster_labels is None:
            return {}
        
        # Group binaries by cluster
        clusters = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(binary_names[i])
        
        # Compute cluster statistics
        cluster_sizes = [len(members) for members in clusters.values()]
        
        analysis = {
            'num_clusters': len(clusters),
            'cluster_sizes': cluster_sizes,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
            'clusters': {f'cluster_{label}': members for label, members in clusters.items()}
        }
        
        return analysis
    
    def _compute_cophenetic_correlation(self, original_distances: np.ndarray) -> float:
        """Compute cophenetic correlation coefficient."""
        if self.linkage_matrix is None:
            return 0.0
        
        from scipy.cluster.hierarchy import cophenet
        
        try:
            cophenetic_distances, _ = cophenet(self.linkage_matrix, squareform(original_distances))
            correlation = np.corrcoef(squareform(original_distances), cophenetic_distances)[0, 1]
            return float(correlation)
        except Exception as e:
            logger.warning(f"Failed to compute cophenetic correlation: {e}")
            return 0.0
    
    def _get_dendrogram_info(self) -> Dict:
        """Get information about the dendrogram structure."""
        if self.linkage_matrix is None:
            return {}
        
        # Get dendrogram without plotting
        dend = dendrogram(self.linkage_matrix, no_plot=True)
        
        return {
            'max_distance': float(np.max(self.linkage_matrix[:, 2])),
            'min_distance': float(np.min(self.linkage_matrix[:, 2])),
            'height_range': float(np.max(self.linkage_matrix[:, 2]) - np.min(self.linkage_matrix[:, 2])),
            'num_merges': int(self.linkage_matrix.shape[0])
        }
    
    def plot_dendrogram(self, binary_names: List[str], output_path: Path, 
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot and save dendrogram."""
        if self.linkage_matrix is None:
            logger.warning("No clustering performed - cannot plot dendrogram")
            return
        
        plt.figure(figsize=figsize)
        
        # Create dendrogram
        dend = dendrogram(
            self.linkage_matrix,
            labels=binary_names,
            leaf_rotation=45,
            leaf_font_size=10
        )
        
        # Add threshold line
        plt.axhline(y=self.distance_threshold, color='red', linestyle='--', 
                   label=f'Threshold = {self.distance_threshold}')
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method} linkage)')
        plt.xlabel('Binary')
        plt.ylabel('Distance')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dendrogram to {output_path}")

def run_hierarchical_clustering(distance_matrices: Dict, binary_names: List[str],
                               output_dir: Path,
                               categories: Optional[Dict[str, str]] = None,
                               euclidean_matrices: Optional[set] = None) -> Dict:
    """
    Run hierarchical clustering on multiple distance matrices.

    Args:
        distance_matrices: Dict with matrix names as keys, matrices as values
        binary_names: Names of binaries corresponding to matrix rows/columns
        output_dir: Output directory for results and plots
        categories: Optional name→category mapping for purity computation
        euclidean_matrices: Set of matrix names known to be Euclidean.
            Ward linkage is only used for these; for non-Euclidean matrices
            (e.g. NCD, cosine-derived) Ward is skipped because it assumes
            Euclidean geometry and can distort dendrogram structure.
    """
    if euclidean_matrices is None:
        euclidean_matrices = set()

    logger.info("Running hierarchical clustering analysis...")

    results = {}

    for matrix_name, distance_matrix in distance_matrices.items():
        if isinstance(distance_matrix, list):
            distance_matrix = np.array(distance_matrix)

        logger.info(f"Clustering with {matrix_name} distance matrix...")

        try:
            # Ward requires Euclidean geometry; skip it for non-Euclidean matrices
            if matrix_name in euclidean_matrices:
                linkage_methods = ['ward', 'complete', 'average']
            else:
                linkage_methods = ['complete', 'average']
                logger.info(f"Skipping Ward linkage for '{matrix_name}' "
                            f"(non-Euclidean distance matrix)")
            method_results = {}

            for method in linkage_methods:
                clusterer = HierarchicalClusterer(linkage_method=method)
                clustering_result = clusterer.fit_clustering(
                    distance_matrix, binary_names, categories=categories)
                
                if clustering_result:  # Only add if successful
                    method_results[method] = clustering_result
                    
                    # Plot dendrogram
                    dendrogram_path = output_dir / f"dendrogram_{matrix_name}_{method}.png"
                    clusterer.plot_dendrogram(binary_names, dendrogram_path)
            
            if method_results:
                # Find best linkage method (highest cophenetic correlation)
                best_method = max(method_results.keys(), 
                                key=lambda m: method_results[m].get('cophenetic_correlation', 0))
                
                results[matrix_name] = {
                    'methods': method_results,
                    'best_method': best_method,
                    'best_result': method_results[best_method],
                    'method_comparison': compare_linkage_methods(method_results)
                }
            else:
                results[matrix_name] = {'error': 'All clustering methods failed'}
                
        except Exception as e:
            logger.error(f"Hierarchical clustering failed for {matrix_name}: {e}")
            results[matrix_name] = {'error': str(e)}
    
    # Overall analysis
    if results:
        results['summary'] = analyze_clustering_consistency(results)
    
    # Save results
    save_json(results, output_dir / "hierarchical_clustering.json")
    
    logger.info("Hierarchical clustering analysis completed")
    return results

def compare_linkage_methods(method_results: Dict) -> Dict:
    """Compare different linkage methods."""
    comparison = {}

    for method, result in method_results.items():
        if 'cluster_analysis' in result:
            comparison[method] = {
                'num_clusters': result['cluster_analysis']['num_clusters'],
                'cophenetic_correlation': result.get('cophenetic_correlation', 0.0),
                'silhouette_score': result.get('silhouette_score'),
                'category_purity': (result['category_purity']['mean_purity']
                                    if result.get('category_purity') else None),
                'largest_cluster_fraction': (result['cluster_analysis']['largest_cluster_size'] /
                                             len(result['binary_names']))
            }
    
    # Find method with best balance of cluster count and correlation
    best_method = None
    best_score = 0.0
    
    for method, stats in comparison.items():
        # Score based on cophenetic correlation and cluster balance
        score = stats['cophenetic_correlation'] * (1 - stats['largest_cluster_fraction'])
        if score > best_score:
            best_score = score
            best_method = method
    
    comparison['recommendation'] = {
        'best_method': best_method,
        'score': best_score,
        'reasoning': f"{best_method} provides the best balance of clustering quality and structure"
    }
    
    return comparison

def analyze_clustering_consistency(results: Dict) -> Dict:
    """Analyze consistency across different distance matrices."""
    matrix_names = [name for name in results.keys() if name != 'summary']
    
    if len(matrix_names) < 2:
        return {'message': 'Need multiple distance matrices to analyze consistency'}
    
    # Compare cluster assignments across matrices
    cluster_assignments = {}
    
    for matrix_name in matrix_names:
        if 'best_result' in results[matrix_name]:
            labels = results[matrix_name]['best_result'].get('cluster_labels', [])
            cluster_assignments[matrix_name] = labels
    
    if len(cluster_assignments) < 2:
        return {'message': 'Insufficient successful clustering results'}
    
    # Compute adjusted rand index between clustering results
    consistency_scores = {}
    matrix_pairs = [(m1, m2) for i, m1 in enumerate(matrix_names) 
                   for m2 in matrix_names[i+1:]]
    
    for m1, m2 in matrix_pairs:
        if m1 in cluster_assignments and m2 in cluster_assignments:
            try:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(cluster_assignments[m1], cluster_assignments[m2])
                consistency_scores[f"{m1}_vs_{m2}"] = float(ari)
            except ImportError:
                logger.warning("sklearn not available for ARI computation")
    
    # Overall consistency assessment
    if consistency_scores:
        mean_consistency = np.mean(list(consistency_scores.values()))
        interpretation = interpret_clustering_consistency(mean_consistency)
    else:
        mean_consistency = 0.0
        interpretation = "Unable to compute consistency scores"
    
    return {
        'pairwise_consistency': consistency_scores,
        'mean_consistency': mean_consistency,
        'interpretation': interpretation,
        'matrix_count': len(matrix_names)
    }

def interpret_clustering_consistency(mean_ari: float) -> str:
    """Interpret the mean adjusted rand index."""
    if mean_ari > 0.8:
        return "Very high consistency - different distance metrics produce similar clusterings"
    elif mean_ari > 0.6:
        return "High consistency - most distance metrics agree on clustering structure"
    elif mean_ari > 0.4:
        return "Moderate consistency - some agreement between distance metrics"
    elif mean_ari > 0.2:
        return "Low consistency - limited agreement between distance metrics"
    else:
        return "Very low consistency - different distance metrics produce very different clusterings"