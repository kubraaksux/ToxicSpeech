"""
Benchmark Comparison Script
Compares original vs optimized implementation performance
"""

import time
import gc
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import both implementations
from optimized_hate_speech_detection import OptimizedHateSpeechDetector, Config
from Hate_Speech_Uncertainty_Estimation import (
    get_embedding, mahalanobis_distance, 
    df_train, df_test, tokenizer, model, device
)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.config = Config()
        self.optimized_detector = OptimizedHateSpeechDetector(self.config)
        self.results = {}
    
    def benchmark_embedding_generation(self, sample_sizes: List[int] = [10, 50, 100, 500]) -> Dict:
        """Benchmark embedding generation performance."""
        print("Benchmarking Embedding Generation...")
        
        results = {
            'sample_sizes': sample_sizes,
            'original_times': [],
            'optimized_times': [],
            'speedup_ratios': []
        }
        
        # Sample texts for testing
        sample_texts = df_train['text'].tolist()
        
        for size in sample_sizes:
            print(f"\nTesting with {size} samples...")
            test_texts = sample_texts[:size]
            
            # Benchmark original implementation
            start_time = time.time()
            original_embeddings = np.vstack([
                get_embedding(text, emb_type='pooler') 
                for text in test_texts
            ])
            original_time = time.time() - start_time
            
            # Clean up memory
            gc.collect()
            
            # Benchmark optimized implementation
            start_time = time.time()
            optimized_embeddings = self.optimized_detector.embedding_generator.generate_embeddings_batch(
                test_texts, 'pooler'
            )
            optimized_time = time.time() - start_time
            
            # Calculate speedup
            speedup = original_time / optimized_time
            
            results['original_times'].append(original_time)
            results['optimized_times'].append(optimized_time)
            results['speedup_ratios'].append(speedup)
            
            print(f"Original: {original_time:.2f}s, Optimized: {optimized_time:.2f}s, Speedup: {speedup:.1f}x")
            
            # Verify embeddings are similar (within tolerance)
            similarity = np.corrcoef(
                original_embeddings.flatten(), 
                optimized_embeddings.flatten()
            )[0, 1]
            print(f"Embedding similarity: {similarity:.4f}")
            
            # Clean up memory
            del original_embeddings, optimized_embeddings
            gc.collect()
        
        return results
    
    def benchmark_mahalanobis_distance(self, feature_sizes: List[int] = [100, 500, 1000]) -> Dict:
        """Benchmark Mahalanobis distance calculation performance."""
        print("\nBenchmarking Mahalanobis Distance Calculation...")
        
        results = {
            'feature_sizes': feature_sizes,
            'original_times': [],
            'optimized_times': [],
            'speedup_ratios': []
        }
        
        # Generate sample data
        n_features = 768  # BERT embedding dimension
        n_classes = 2
        
        for size in feature_sizes:
            print(f"\nTesting with {size} samples...")
            
            # Create sample data
            train_features = np.random.randn(size, n_features)
            eval_features = np.random.randn(size // 4, n_features)
            train_labels = np.random.randint(0, n_classes, size)
            
            # Benchmark original implementation
            start_time = time.time()
            original_distances, _, _ = mahalanobis_distance(
                train_features, train_labels, eval_features, return_full=False
            )
            original_time = time.time() - start_time
            
            # Benchmark optimized implementation
            start_time = time.time()
            optimized_distances, _, _ = self.optimized_detector.distance_calculator.mahalanobis_distance_optimized(
                train_features, train_labels, eval_features, return_full=False
            )
            optimized_time = time.time() - start_time
            
            # Calculate speedup
            speedup = original_time / optimized_time
            
            results['original_times'].append(original_time)
            results['optimized_times'].append(optimized_time)
            results['speedup_ratios'].append(speedup)
            
            print(f"Original: {original_time:.2f}s, Optimized: {optimized_time:.2f}s, Speedup: {speedup:.1f}x")
            
            # Verify results are similar
            correlation = np.corrcoef(original_distances, optimized_distances)[0, 1]
            print(f"Distance correlation: {correlation:.4f}")
            
            # Clean up memory
            del train_features, eval_features, train_labels
            del original_distances, optimized_distances
            gc.collect()
        
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage improvements."""
        print("\nBenchmarking Memory Usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure original implementation memory
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run small sample with original approach (simulated)
        sample_texts = df_train['text'].tolist()[:100]
        original_embeddings = np.vstack([
            get_embedding(text, emb_type='pooler') 
            for text in sample_texts
        ])
        
        mem_after_original = process.memory_info().rss / 1024 / 1024
        original_memory = mem_after_original - mem_before
        
        # Clean up
        del original_embeddings
        gc.collect()
        
        # Measure optimized implementation memory
        mem_before = process.memory_info().rss / 1024 / 1024
        
        optimized_embeddings = self.optimized_detector.embedding_generator.generate_embeddings_batch(
            sample_texts, 'pooler'
        )
        
        mem_after_optimized = process.memory_info().rss / 1024 / 1024
        optimized_memory = mem_after_optimized - mem_before
        
        # Clean up
        del optimized_embeddings
        gc.collect()
        
        results = {
            'original_memory_mb': original_memory,
            'optimized_memory_mb': optimized_memory,
            'memory_reduction': (original_memory - optimized_memory) / original_memory * 100
        }
        
        print(f"Original memory usage: {original_memory:.1f} MB")
        print(f"Optimized memory usage: {optimized_memory:.1f} MB")
        print(f"Memory reduction: {results['memory_reduction']:.1f}%")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print("=" * 60)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        self.results['embedding_benchmark'] = self.benchmark_embedding_generation()
        self.results['distance_benchmark'] = self.benchmark_mahalanobis_distance()
        self.results['memory_benchmark'] = self.benchmark_memory_usage()
        
        self._print_summary()
        self._create_visualization()
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Embedding generation summary
        embedding_results = self.results['embedding_benchmark']
        avg_embedding_speedup = np.mean(embedding_results['speedup_ratios'])
        max_embedding_speedup = np.max(embedding_results['speedup_ratios'])
        
        print(f"\nEmbedding Generation:")
        print(f"  Average Speedup: {avg_embedding_speedup:.1f}x")
        print(f"  Maximum Speedup: {max_embedding_speedup:.1f}x")
        
        # Distance calculation summary
        distance_results = self.results['distance_benchmark']
        avg_distance_speedup = np.mean(distance_results['speedup_ratios'])
        max_distance_speedup = np.max(distance_results['speedup_ratios'])
        
        print(f"\nMahalanobis Distance Calculation:")
        print(f"  Average Speedup: {avg_distance_speedup:.1f}x")
        print(f"  Maximum Speedup: {max_distance_speedup:.1f}x")
        
        # Memory usage summary
        memory_results = self.results['memory_benchmark']
        print(f"\nMemory Usage:")
        print(f"  Memory Reduction: {memory_results['memory_reduction']:.1f}%")
        
        # Overall impact
        overall_speedup = avg_embedding_speedup * avg_distance_speedup
        print(f"\nOverall Expected Speedup: {overall_speedup:.1f}x")
        
        print("\n" + "=" * 60)
    
    def _create_visualization(self):
        """Create performance visualization charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # Embedding generation speedup
        ax1 = axes[0, 0]
        embedding_results = self.results['embedding_benchmark']
        ax1.plot(embedding_results['sample_sizes'], embedding_results['speedup_ratios'], 
                'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Speedup Ratio (x)')
        ax1.set_title('Embedding Generation Speedup')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Distance calculation speedup
        ax2 = axes[0, 1]
        distance_results = self.results['distance_benchmark']
        ax2.plot(distance_results['feature_sizes'], distance_results['speedup_ratios'], 
                'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Feature Size')
        ax2.set_ylabel('Speedup Ratio (x)')
        ax2.set_title('Mahalanobis Distance Speedup')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Time comparison for embeddings
        ax3 = axes[1, 0]
        x = np.arange(len(embedding_results['sample_sizes']))
        width = 0.35
        ax3.bar(x - width/2, embedding_results['original_times'], width, 
               label='Original', alpha=0.8, color='lightcoral')
        ax3.bar(x + width/2, embedding_results['optimized_times'], width, 
               label='Optimized', alpha=0.8, color='lightblue')
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Embedding Generation Time Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(embedding_results['sample_sizes'])
        ax3.legend()
        ax3.set_yscale('log')
        
        # Time comparison for distance calculation
        ax4 = axes[1, 1]
        x = np.arange(len(distance_results['feature_sizes']))
        ax4.bar(x - width/2, distance_results['original_times'], width, 
               label='Original', alpha=0.8, color='lightcoral')
        ax4.bar(x + width/2, distance_results['optimized_times'], width, 
               label='Optimized', alpha=0.8, color='lightblue')
        ax4.set_xlabel('Feature Size')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Distance Calculation Time Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(distance_results['feature_sizes'])
        ax4.legend()
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('performance_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'performance_benchmark_results.png'")


def main():
    """Run the complete benchmark comparison."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    
    return results


if __name__ == "__main__":
    results = main()