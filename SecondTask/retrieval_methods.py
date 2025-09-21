"""
Retrieval Methods Evaluation
This module implements various retrieval methods including reranking and hybrid approaches
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import time
import re
from collections import defaultdict


class RetrievalEvaluator:
    """
    A class to evaluate different retrieval methods on chunked product data
    """

    def __init__(self, chunks_data: Dict[str, Any]):
        """Initialize with chunked data from different strategies"""
        self.chunks_data = chunks_data
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.rerank_model = SentenceTransformer('sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2')

    def prepare_corpus(self, strategy: str) -> Tuple[List[str], List[Dict]]:
        """Prepare corpus for retrieval from a specific chunking strategy"""
        chunks = self.chunks_data[strategy]['chunks']
        texts = [chunk['text'] for chunk in chunks]
        metadata = [chunk for chunk in chunks]
        return texts, metadata

    def dense_retrieval(self, query: str, corpus_texts: List[str],
                       corpus_metadata: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Dense retrieval using sentence transformers
        """
        # Encode query and corpus
        query_embedding = self.sentence_model.encode([query])
        corpus_embeddings = self.sentence_model.encode(corpus_texts)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': corpus_metadata[idx]['chunk_id'],
                'text': corpus_texts[idx],
                'score': float(similarities[idx]),
                'metadata': corpus_metadata[idx]['metadata'],
                'method': 'dense_retrieval'
            })

        return results

    def sparse_retrieval_bm25(self, query: str, corpus_texts: List[str],
                             corpus_metadata: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Sparse retrieval using BM25
        """
        # Tokenize corpus
        tokenized_corpus = [text.lower().split() for text in corpus_texts]

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)

        # Get scores
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': corpus_metadata[idx]['chunk_id'],
                'text': corpus_texts[idx],
                'score': float(scores[idx]),
                'metadata': corpus_metadata[idx]['metadata'],
                'method': 'sparse_bm25'
            })

        return results

    def tfidf_retrieval(self, query: str, corpus_texts: List[str],
                       corpus_metadata: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        TF-IDF based sparse retrieval
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

        # Fit and transform corpus
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)

        # Transform query
        query_vector = vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': corpus_metadata[idx]['chunk_id'],
                'text': corpus_texts[idx],
                'score': float(similarities[idx]),
                'metadata': corpus_metadata[idx]['metadata'],
                'method': 'tfidf'
            })

        return results

    def hybrid_retrieval(self, query: str, corpus_texts: List[str],
                        corpus_metadata: List[Dict], top_k: int = 10,
                        alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse methods
        alpha: weight for dense retrieval (1-alpha for sparse)
        """
        # Get results from both methods
        dense_results = self.dense_retrieval(query, corpus_texts, corpus_metadata, top_k * 2)
        sparse_results = self.sparse_retrieval_bm25(query, corpus_texts, corpus_metadata, top_k * 2)

        # Normalize scores
        if dense_results:
            max_dense = max([r['score'] for r in dense_results])
            min_dense = min([r['score'] for r in dense_results])
            for result in dense_results:
                if max_dense != min_dense:
                    result['normalized_score'] = (result['score'] - min_dense) / (max_dense - min_dense)
                else:
                    result['normalized_score'] = 1.0

        if sparse_results:
            max_sparse = max([r['score'] for r in sparse_results])
            min_sparse = min([r['score'] for r in sparse_results])
            for result in sparse_results:
                if max_sparse != min_sparse:
                    result['normalized_score'] = (result['score'] - min_sparse) / (max_sparse - min_sparse)
                else:
                    result['normalized_score'] = 1.0

        # Combine scores
        hybrid_scores = defaultdict(float)
        chunk_data = {}

        # Add dense scores
        for result in dense_results:
            chunk_id = result['chunk_id']
            hybrid_scores[chunk_id] += alpha * result['normalized_score']
            chunk_data[chunk_id] = result

        # Add sparse scores
        for result in sparse_results:
            chunk_id = result['chunk_id']
            hybrid_scores[chunk_id] += (1 - alpha) * result['normalized_score']
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # Sort by hybrid score
        sorted_chunks = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id].copy()
            result['score'] = score
            result['method'] = 'hybrid'
            results.append(result)

        return results

    def rerank_results(self, query: str, initial_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank initial results using a cross-encoder model
        """
        if not initial_results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result['text']) for result in initial_results]

        # Get reranking scores
        rerank_scores = self.rerank_model.encode(pairs, convert_to_tensor=True, show_progress_bar=False)

        # Apply sigmoid to get probabilities
        import torch
        if hasattr(torch.nn, 'functional'):
            rerank_scores = torch.nn.functional.sigmoid(rerank_scores).cpu().numpy()
        else:
            rerank_scores = 1 / (1 + np.exp(-rerank_scores.cpu().numpy()))

        # Update results with rerank scores
        reranked_results = []
        for i, result in enumerate(initial_results):
            new_result = result.copy()
            new_result['rerank_score'] = float(rerank_scores[i])
            new_result['original_score'] = result['score']
            new_result['method'] = result['method'] + '_reranked'
            reranked_results.append(new_result)

        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return reranked_results[:top_k]

    def evaluate_retrieval_methods(self, test_queries: List[str], strategy: str = 'token_512_50') -> Dict[str, Any]:
        """
        Evaluate all retrieval methods on test queries
        """
        print(f"Evaluating retrieval methods on {strategy} chunking strategy")

        corpus_texts, corpus_metadata = self.prepare_corpus(strategy)

        results = {
            'strategy': strategy,
            'num_documents': len(corpus_texts),
            'test_queries': test_queries,
            'method_results': {}
        }

        methods = ['dense', 'sparse_bm25', 'tfidf', 'hybrid', 'dense_reranked', 'hybrid_reranked']

        for method in methods:
            print(f"Running {method} retrieval...")
            start_time = time.time()
            method_results = []

            for query in test_queries:
                if method == 'dense':
                    query_results = self.dense_retrieval(query, corpus_texts, corpus_metadata)
                elif method == 'sparse_bm25':
                    query_results = self.sparse_retrieval_bm25(query, corpus_texts, corpus_metadata)
                elif method == 'tfidf':
                    query_results = self.tfidf_retrieval(query, corpus_texts, corpus_metadata)
                elif method == 'hybrid':
                    query_results = self.hybrid_retrieval(query, corpus_texts, corpus_metadata)
                elif method == 'dense_reranked':
                    initial_results = self.dense_retrieval(query, corpus_texts, corpus_metadata, top_k=20)
                    query_results = self.rerank_results(query, initial_results)
                elif method == 'hybrid_reranked':
                    initial_results = self.hybrid_retrieval(query, corpus_texts, corpus_metadata, top_k=20)
                    query_results = self.rerank_results(query, initial_results)

                method_results.append({
                    'query': query,
                    'results': query_results,
                    'num_results': len(query_results)
                })

            end_time = time.time()

            results['method_results'][method] = {
                'results': method_results,
                'processing_time': end_time - start_time,
                'avg_time_per_query': (end_time - start_time) / len(test_queries)
            }

        return results

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all chunking strategies and retrieval methods
        """
        # Define test queries
        test_queries = [
            "Intel Core i7 gaming desktop",
            "AMD Ryzen budget PC",
            "ASUS brand computer",
            "high performance gaming PC",
            "desktop PC under 50000 taka",
            "Core i5 desktop with SSD",
            "MSI gaming computer",
            "budget desktop PC AMD",
            "13th gen Intel desktop",
            "gaming PC with graphics card"
        ]

        evaluation_results = {}

        # Test on different chunking strategies
        strategies_to_test = ['token_512_50', 'sentence_3', 'semantic_05']

        for strategy in strategies_to_test:
            if strategy in self.chunks_data:
                evaluation_results[strategy] = self.evaluate_retrieval_methods(test_queries, strategy)

        # Save results
        with open('retrieval_evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        # Generate summary
        summary = self._generate_evaluation_summary(evaluation_results)

        with open('retrieval_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("Retrieval evaluation completed. Results saved to files.")

        return evaluation_results

    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results"""
        summary = {
            'strategies_evaluated': list(evaluation_results.keys()),
            'methods_compared': [],
            'performance_summary': {}
        }

        for strategy, strategy_results in evaluation_results.items():
            summary['methods_compared'] = list(strategy_results['method_results'].keys())
            summary['performance_summary'][strategy] = {}

            for method, method_data in strategy_results['method_results'].items():
                # Calculate average scores and response times
                all_scores = []
                for query_result in method_data['results']:
                    for result in query_result['results']:
                        if 'rerank_score' in result:
                            all_scores.append(result['rerank_score'])
                        else:
                            all_scores.append(result['score'])

                summary['performance_summary'][strategy][method] = {
                    'avg_processing_time': method_data['processing_time'],
                    'avg_time_per_query': method_data['avg_time_per_query'],
                    'avg_score': np.mean(all_scores) if all_scores else 0,
                    'max_score': np.max(all_scores) if all_scores else 0,
                    'min_score': np.min(all_scores) if all_scores else 0,
                    'std_score': np.std(all_scores) if all_scores else 0
                }

        return summary


def main():
    """Main execution function"""
    # Load chunked data (assumes chunking_strategies.py has been run)
    try:
        # Try to load existing chunked data
        with open('chunking_statistics.json', 'r') as f:
            chunking_stats = json.load(f)

        # Load sample chunks for evaluation
        chunks_data = {}
        strategies = ['token_512_50', 'token_256_25', 'sentence_3', 'sentence_5', 'semantic_05', 'semantic_07']

        for strategy in strategies:
            try:
                with open(f'chunks_sample_{strategy}.json', 'r') as f:
                    chunks = json.load(f)
                    chunks_data[strategy] = {
                        'chunks': chunks,
                        'stats': chunking_stats[strategy]
                    }
                    print(f"Loaded {len(chunks)} sample chunks for {strategy}")
            except FileNotFoundError:
                print(f"Warning: Could not find chunks for {strategy}")

        if not chunks_data:
            print("No chunked data found. Please run chunking_strategies.py first.")
            return

        # Initialize retrieval evaluator
        evaluator = RetrievalEvaluator(chunks_data)

        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()

        # Print summary
        print("\n=== Retrieval Methods Evaluation Summary ===")
        for strategy in results.keys():
            print(f"\nStrategy: {strategy.upper()}")
            for method, method_data in results[strategy]['method_results'].items():
                print(f"  {method}: {method_data['avg_time_per_query']:.3f}s per query")

    except FileNotFoundError:
        print("Please run chunking_strategies.py first to generate chunked data.")


if __name__ == "__main__":
    main()