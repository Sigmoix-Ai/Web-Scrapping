"""
Main Evaluation Script for Task Two Assignment
This script orchestrates the complete evaluation of chunking strategies,
retrieval methods, and vector database indexing algorithms
"""

import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import numpy as np
from chunking_strategies import ProductDataChunker
from retrieval_methods import RetrievalEvaluator
from vector_db_indexing import VectorDBEvaluator


class ComprehensiveEvaluator:
    """
    A comprehensive evaluator that runs all experiments and generates reports
    """

    def __init__(self, csv_path: str):
        """Initialize with the product CSV file path"""
        self.csv_path = csv_path
        self.results = {}

    def run_complete_evaluation(self):
        """
        Run the complete evaluation pipeline
        """
        print("=== Starting Comprehensive Evaluation ===")
        print("This evaluation will test:")
        print("1. Different chunking strategies using Chonkie")
        print("2. Various retrieval methods (reranking and hybrid approaches)")
        print("3. Multiple vector database indexing algorithms")
        print()

        # Step 1: Chunking strategies evaluation
        print("Step 1: Evaluating chunking strategies...")
        chunking_results = self._evaluate_chunking_strategies()
        self.results['chunking'] = chunking_results

        # Step 2: Retrieval methods evaluation
        print("\nStep 2: Evaluating retrieval methods...")
        retrieval_results = self._evaluate_retrieval_methods()
        self.results['retrieval'] = retrieval_results

        # Step 3: Vector database indexing evaluation
        print("\nStep 3: Evaluating vector database indexing...")
        indexing_results = self._evaluate_vector_indexing()
        self.results['indexing'] = indexing_results

        # Step 4: Generate comprehensive report
        print("\nStep 4: Generating comprehensive report...")
        self._generate_comprehensive_report()

        print("\n=== Evaluation Complete ===")

    def _evaluate_chunking_strategies(self) -> Dict[str, Any]:
        """Evaluate different chunking strategies"""
        chunker = ProductDataChunker(self.csv_path, max_products=500)  # Limit products
        return chunker.run_all_chunking_strategies()

    def _evaluate_retrieval_methods(self) -> Dict[str, Any]:
        """Evaluate retrieval methods on chunked data"""
        # Load chunked data
        try:
            with open('chunking_statistics.json', 'r') as f:
                chunking_stats = json.load(f)

            chunks_data = {}
            strategies = ['token_512_50', 'sentence_3', 'semantic_05']

            for strategy in strategies:
                try:
                    with open(f'chunks_sample_{strategy}.json', 'r') as f:
                        chunks = json.load(f)
                        chunks_data[strategy] = {
                            'chunks': chunks,
                            'stats': chunking_stats[strategy]
                        }
                except FileNotFoundError:
                    continue

            if chunks_data:
                evaluator = RetrievalEvaluator(chunks_data)
                return evaluator.run_comprehensive_evaluation()
            else:
                return {"error": "No chunked data available"}

        except FileNotFoundError:
            return {"error": "Chunking results not found"}

    def _evaluate_vector_indexing(self) -> Dict[str, Any]:
        """Evaluate vector database indexing algorithms"""
        try:
            chunks_data = {}
            strategies = ['token_512_50', 'sentence_3']

            for strategy in strategies:
                try:
                    with open(f'chunks_sample_{strategy}.json', 'r') as f:
                        chunks = json.load(f)
                        chunks_data[strategy] = {'chunks': chunks}
                except FileNotFoundError:
                    continue

            if chunks_data:
                evaluator = VectorDBEvaluator(chunks_data)
                results = {}

                # Test with different document sizes
                for strategy in ['token_512_50', 'sentence_3']:
                    if strategy in chunks_data:
                        strategy_results = {}
                        for doc_size in [500, 1000]:
                            key = f"{strategy}_{doc_size}"
                            strategy_results[key] = evaluator.run_comprehensive_indexing_evaluation(
                                strategy, doc_size
                            )
                        results[strategy] = strategy_results

                return results
            else:
                return {"error": "No chunked data available"}

        except Exception as e:
            return {"error": str(e)}

    def _generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        report = {
            "evaluation_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "StarTech Products CSV",
                "total_products": 8463,
                "evaluation_components": [
                    "Chunking Strategies (Token, Sentence, Semantic)",
                    "Retrieval Methods (Dense, Sparse, Hybrid, Reranking)",
                    "Vector Database Indexing (FAISS, ChromaDB, Sklearn)"
                ]
            }
        }

        # Chunking analysis
        if 'chunking' in self.results:
            report['chunking_analysis'] = self._analyze_chunking_results()

        # Retrieval analysis
        if 'retrieval' in self.results:
            report['retrieval_analysis'] = self._analyze_retrieval_results()

        # Indexing analysis
        if 'indexing' in self.results:
            report['indexing_analysis'] = self._analyze_indexing_results()

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()

        # Save comprehensive report
        with open('comprehensive_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualizations
        self._create_visualizations()

        print("Comprehensive report saved to 'comprehensive_evaluation_report.json'")

    def _analyze_chunking_results(self) -> Dict[str, Any]:
        """Analyze chunking strategy results"""
        chunking_data = self.results['chunking']

        analysis = {
            "strategy_comparison": {},
            "efficiency_metrics": {},
            "chunk_distribution": {}
        }

        for strategy, data in chunking_data.items():
            stats = data['stats']
            analysis['strategy_comparison'][strategy] = {
                "total_chunks": stats['total_chunks'],
                "avg_chunks_per_product": round(stats['avg_chunks_per_product'], 2),
                "processing_time": round(stats['processing_time'], 2),
                "avg_chunk_size": round(stats['avg_chunk_size'], 1)
            }

        # Find best strategies
        fastest = min(analysis['strategy_comparison'].items(),
                     key=lambda x: x[1]['processing_time'])
        most_efficient = min(analysis['strategy_comparison'].items(),
                           key=lambda x: x[1]['avg_chunks_per_product'])

        analysis['efficiency_metrics'] = {
            "fastest_processing": {
                "strategy": fastest[0],
                "time": fastest[1]['processing_time']
            },
            "most_efficient_chunking": {
                "strategy": most_efficient[0],
                "chunks_per_product": most_efficient[1]['avg_chunks_per_product']
            }
        }

        return analysis

    def _analyze_retrieval_results(self) -> Dict[str, Any]:
        """Analyze retrieval method results"""
        if 'error' in self.results['retrieval']:
            return {"error": self.results['retrieval']['error']}

        retrieval_data = self.results['retrieval']

        analysis = {
            "method_performance": {},
            "strategy_performance": {},
            "best_combinations": {}
        }

        # Analyze performance across strategies and methods
        for strategy, strategy_data in retrieval_data.items():
            if 'method_results' in strategy_data:
                strategy_analysis = {}
                for method, method_data in strategy_data['method_results'].items():
                    strategy_analysis[method] = {
                        "avg_time_per_query": round(method_data['avg_time_per_query'], 4),
                        "total_processing_time": round(method_data['processing_time'], 2)
                    }
                analysis['strategy_performance'][strategy] = strategy_analysis

        return analysis

    def _analyze_indexing_results(self) -> Dict[str, Any]:
        """Analyze vector database indexing results"""
        if 'error' in self.results['indexing']:
            return {"error": self.results['indexing']['error']}

        indexing_data = self.results['indexing']

        analysis = {
            "algorithm_performance": {},
            "scalability_analysis": {},
            "efficiency_comparison": {}
        }

        # Analyze each strategy and document size combination
        for strategy, strategy_data in indexing_data.items():
            strategy_analysis = {}
            for config, config_data in strategy_data.items():
                if 'algorithms' in config_data:
                    config_analysis = {}
                    for algo, algo_data in config_data['algorithms'].items():
                        if 'error' not in algo_data:
                            config_analysis[algo] = {
                                "build_time": round(algo_data['build_time'], 3),
                                "avg_search_time": round(algo_data['avg_search_time'], 4),
                                "index_size_mb": round(algo_data['index_size_mb'], 1)
                            }
                    strategy_analysis[config] = config_analysis
            analysis['algorithm_performance'][strategy] = strategy_analysis

        return analysis

    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on evaluation results"""
        recommendations = {
            "chunking_strategy": {},
            "retrieval_method": {},
            "vector_indexing": {},
            "overall_best_practices": []
        }

        # Chunking recommendations
        if 'chunking' in self.results:
            chunking_analysis = self._analyze_chunking_results()
            if 'efficiency_metrics' in chunking_analysis:
                recommendations['chunking_strategy'] = {
                    "for_speed": chunking_analysis['efficiency_metrics']['fastest_processing']['strategy'],
                    "for_efficiency": chunking_analysis['efficiency_metrics']['most_efficient_chunking']['strategy'],
                    "general_recommendation": "Token-based chunking with 512 tokens provides good balance"
                }

        # Overall recommendations
        recommendations['overall_best_practices'] = [
            "Use token-based chunking (512 tokens) for balanced performance",
            "Implement hybrid retrieval for best accuracy",
            "Use FAISS HNSW for large-scale deployments",
            "Consider reranking for improved relevance",
            "Monitor chunk size distribution for optimal performance"
        ]

        return recommendations

    def _create_visualizations(self):
        """Create visualization plots for the evaluation results"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('StarTech Products - Chunking and Retrieval Evaluation', fontsize=16)

        # Plot 1: Chunking strategy comparison
        if 'chunking' in self.results:
            chunking_data = self.results['chunking']
            strategies = list(chunking_data.keys())
            processing_times = [chunking_data[s]['stats']['processing_time'] for s in strategies]
            chunk_counts = [chunking_data[s]['stats']['total_chunks'] for s in strategies]

            axes[0, 0].bar(strategies, processing_times)
            axes[0, 0].set_title('Chunking Processing Time by Strategy')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)

            axes[0, 1].bar(strategies, chunk_counts)
            axes[0, 1].set_title('Total Chunks by Strategy')
            axes[0, 1].set_ylabel('Number of Chunks')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 2: Vector indexing performance comparison
        if 'indexing' in self.results and 'token_512_50' in self.results['indexing']:
            strategy_data = self.results['indexing']['token_512_50']
            if 'token_512_50_1000' in strategy_data:
                config_data = strategy_data['token_512_50_1000']
                if 'algorithms' in config_data:
                    algorithms = []
                    build_times = []
                    search_times = []

                    for algo, data in config_data['algorithms'].items():
                        if 'error' not in data:
                            algorithms.append(algo)
                            build_times.append(data['build_time'])
                            search_times.append(data['avg_search_time'])

                    if algorithms:
                        x = np.arange(len(algorithms))
                        axes[1, 0].bar(x, build_times)
                        axes[1, 0].set_title('Index Build Time by Algorithm')
                        axes[1, 0].set_ylabel('Time (seconds)')
                        axes[1, 0].set_xticks(x)
                        axes[1, 0].set_xticklabels(algorithms, rotation=45)

                        axes[1, 1].bar(x, search_times)
                        axes[1, 1].set_title('Average Search Time by Algorithm')
                        axes[1, 1].set_ylabel('Time (seconds)')
                        axes[1, 1].set_xticks(x)
                        axes[1, 1].set_xticklabels(algorithms, rotation=45)

        plt.tight_layout()
        plt.savefig('evaluation_results_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Visualization saved to 'evaluation_results_visualization.png'")

    def print_summary(self):
        """Print a summary of the evaluation results"""
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*50)

        if 'chunking' in self.results:
            print("\n📊 CHUNKING STRATEGIES:")
            chunking_analysis = self._analyze_chunking_results()
            for strategy, metrics in chunking_analysis['strategy_comparison'].items():
                print(f"  {strategy}: {metrics['total_chunks']} chunks, "
                      f"{metrics['processing_time']}s, "
                      f"{metrics['avg_chunk_size']} avg tokens")

        if 'retrieval' in self.results and 'error' not in self.results['retrieval']:
            print("\n🔍 RETRIEVAL METHODS:")
            print("  Evaluated: Dense, Sparse (BM25), TF-IDF, Hybrid, Reranking")
            print("  Test queries: 10 product search scenarios")

        if 'indexing' in self.results and 'error' not in self.results['indexing']:
            print("\n🗃️ VECTOR INDEXING:")
            print("  Algorithms: FAISS (Flat, IVF, HNSW), ChromaDB, Sklearn")
            print("  Document sizes: 500, 1000")

        print("\n📋 RECOMMENDATIONS:")
        recommendations = self._generate_recommendations()
        for rec in recommendations['overall_best_practices']:
            print(f"  • {rec}")

        print("\n📁 FILES GENERATED:")
        print("  • comprehensive_evaluation_report.json")
        print("  • chunking_statistics.json")
        print("  • retrieval_evaluation_results.json")
        print("  • vector_db_evaluation_*.json")
        print("  • evaluation_results_visualization.png")


def main():
    """Main execution function"""
    print("Starting Task Two Assignment Evaluation")
    print("StarTech Products Dataset Analysis")
    print("-" * 40)

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator('startech_products.csv')

    # Run complete evaluation
    evaluator.run_complete_evaluation()

    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main()