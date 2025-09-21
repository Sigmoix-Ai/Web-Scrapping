"""
Vector Database Indexing Algorithms Evaluation
This module tests multiple vector database indexing algorithms for efficiency and accuracy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import time
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import chromadb
from chromadb.config import Settings
import os
import pickle


class VectorDBEvaluator:
    """
    A class to evaluate different vector database indexing algorithms
    """

    def __init__(self, chunks_data: Dict[str, Any]):
        """Initialize with chunked data"""
        self.chunks_data = chunks_data
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def prepare_embeddings(self, strategy: str, max_documents: int = 1000) -> Tuple[np.ndarray, List[Dict]]:
        """
        Prepare embeddings for a specific chunking strategy
        """
        print(f"Preparing embeddings for {strategy} strategy...")

        chunks = self.chunks_data[strategy]['chunks'][:max_documents]
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        start_time = time.time()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embedding_time = time.time() - start_time

        print(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")

        return embeddings, chunks

    def evaluate_faiss_flat(self, embeddings: np.ndarray, chunks: List[Dict],
                           test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate FAISS Flat (exact search) indexing
        """
        print("Evaluating FAISS Flat indexing...")

        # Build index
        start_time = time.time()
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        build_time = time.time() - start_time

        # Test search performance
        query_embeddings = self.embedding_model.encode(test_queries, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)

        search_times = []
        all_results = []

        for i, query in enumerate(test_queries):
            start_time = time.time()
            scores, indices = index.search(query_embeddings[i:i+1], k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

            # Format results
            query_results = []
            for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    query_results.append({
                        'rank': j + 1,
                        'chunk_id': chunks[idx]['chunk_id'],
                        'score': float(score),
                        'text': chunks[idx]['text'][:200] + "...",
                        'metadata': chunks[idx]['metadata']
                    })

            all_results.append({
                'query': query,
                'results': query_results
            })

        return {
            'algorithm': 'faiss_flat',
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_size_mb': index.ntotal * self.embedding_dim * 4 / (1024 * 1024),  # Approximate
            'num_vectors': index.ntotal,
            'results': all_results,
            'search_times': search_times
        }

    def evaluate_faiss_ivf(self, embeddings: np.ndarray, chunks: List[Dict],
                          test_queries: List[str], nlist: int = 100) -> Dict[str, Any]:
        """
        Evaluate FAISS IVF (approximate search) indexing
        """
        print(f"Evaluating FAISS IVF indexing with nlist={nlist}...")

        # Build index
        start_time = time.time()
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Train index
        index.train(embeddings)
        index.add(embeddings)
        build_time = time.time() - start_time

        # Set search parameters
        index.nprobe = min(10, nlist)  # Number of clusters to search

        # Test search performance
        query_embeddings = self.embedding_model.encode(test_queries, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)

        search_times = []
        all_results = []

        for i, query in enumerate(test_queries):
            start_time = time.time()
            scores, indices = index.search(query_embeddings[i:i+1], k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

            # Format results
            query_results = []
            for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    query_results.append({
                        'rank': j + 1,
                        'chunk_id': chunks[idx]['chunk_id'],
                        'score': float(score),
                        'text': chunks[idx]['text'][:200] + "...",
                        'metadata': chunks[idx]['metadata']
                    })

            all_results.append({
                'query': query,
                'results': query_results
            })

        return {
            'algorithm': 'faiss_ivf',
            'nlist': nlist,
            'nprobe': index.nprobe,
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_size_mb': index.ntotal * self.embedding_dim * 4 / (1024 * 1024),
            'num_vectors': index.ntotal,
            'results': all_results,
            'search_times': search_times
        }

    def evaluate_faiss_hnsw(self, embeddings: np.ndarray, chunks: List[Dict],
                           test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate FAISS HNSW indexing
        """
        print("Evaluating FAISS HNSW indexing...")

        # Build index
        start_time = time.time()
        index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M = 32
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16

        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        build_time = time.time() - start_time

        # Test search performance
        query_embeddings = self.embedding_model.encode(test_queries, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)

        search_times = []
        all_results = []

        for i, query in enumerate(test_queries):
            start_time = time.time()
            scores, indices = index.search(query_embeddings[i:i+1], k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

            # Format results
            query_results = []
            for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    query_results.append({
                        'rank': j + 1,
                        'chunk_id': chunks[idx]['chunk_id'],
                        'score': float(score),
                        'text': chunks[idx]['text'][:200] + "...",
                        'metadata': chunks[idx]['metadata']
                    })

            all_results.append({
                'query': query,
                'results': query_results
            })

        return {
            'algorithm': 'faiss_hnsw',
            'M': 32,
            'efConstruction': 40,
            'efSearch': 16,
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_size_mb': index.ntotal * self.embedding_dim * 4 / (1024 * 1024),
            'num_vectors': index.ntotal,
            'results': all_results,
            'search_times': search_times
        }

    def evaluate_chromadb(self, embeddings: np.ndarray, chunks: List[Dict],
                         test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate ChromaDB indexing
        """
        print("Evaluating ChromaDB indexing...")

        # Initialize ChromaDB client
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))

        # Create collection
        collection_name = f"startech_products_{int(time.time())}"
        collection = client.create_collection(name=collection_name)

        # Prepare data for ChromaDB
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [str(chunk['chunk_id']) for chunk in chunks]

        # Add documents to collection
        start_time = time.time()
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        build_time = time.time() - start_time

        # Test search performance
        search_times = []
        all_results = []

        for query in test_queries:
            start_time = time.time()
            results = collection.query(
                query_texts=[query],
                n_results=10
            )
            search_time = time.time() - start_time
            search_times.append(search_time)

            # Format results
            query_results = []
            if results['ids'] and results['ids'][0]:
                for i, (doc_id, distance, document, metadata) in enumerate(
                    zip(results['ids'][0], results['distances'][0],
                        results['documents'][0], results['metadatas'][0])):

                    query_results.append({
                        'rank': i + 1,
                        'chunk_id': int(doc_id),
                        'score': 1 - distance,  # Convert distance to similarity
                        'text': document[:200] + "...",
                        'metadata': metadata
                    })

            all_results.append({
                'query': query,
                'results': query_results
            })

        # Cleanup
        try:
            client.delete_collection(collection_name)
        except:
            pass

        return {
            'algorithm': 'chromadb',
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_size_mb': len(embeddings) * self.embedding_dim * 4 / (1024 * 1024),
            'num_vectors': len(embeddings),
            'results': all_results,
            'search_times': search_times
        }

    def evaluate_sklearn_nearestneighbors(self, embeddings: np.ndarray, chunks: List[Dict],
                                        test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate sklearn NearestNeighbors as baseline
        """
        print("Evaluating sklearn NearestNeighbors...")

        from sklearn.neighbors import NearestNeighbors

        # Build index
        start_time = time.time()
        nbrs = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='auto')
        nbrs.fit(embeddings)
        build_time = time.time() - start_time

        # Test search performance
        query_embeddings = self.embedding_model.encode(test_queries, convert_to_numpy=True)

        search_times = []
        all_results = []

        for i, query in enumerate(test_queries):
            start_time = time.time()
            distances, indices = nbrs.kneighbors(query_embeddings[i:i+1])
            search_time = time.time() - start_time
            search_times.append(search_time)

            # Format results
            query_results = []
            for j, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                query_results.append({
                    'rank': j + 1,
                    'chunk_id': chunks[idx]['chunk_id'],
                    'score': 1 - distance,  # Convert distance to similarity
                    'text': chunks[idx]['text'][:200] + "...",
                    'metadata': chunks[idx]['metadata']
                })

            all_results.append({
                'query': query,
                'results': query_results
            })

        return {
            'algorithm': 'sklearn_nn',
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_size_mb': len(embeddings) * self.embedding_dim * 8 / (1024 * 1024),  # Double precision
            'num_vectors': len(embeddings),
            'results': all_results,
            'search_times': search_times
        }

    def run_comprehensive_indexing_evaluation(self, strategy: str = 'token_512_50',
                                            max_documents: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all indexing algorithms
        """
        print(f"=== Comprehensive Vector Database Indexing Evaluation ===")
        print(f"Strategy: {strategy}, Max documents: {max_documents}")

        # Prepare data
        embeddings, chunks = self.prepare_embeddings(strategy, max_documents)

        # Test queries
        test_queries = [
            "Intel Core i7 gaming desktop",
            "AMD Ryzen budget PC",
            "ASUS brand computer",
            "high performance gaming PC",
            "desktop PC under 50000 taka"
        ]

        results = {
            'strategy': strategy,
            'num_documents': len(chunks),
            'embedding_dimension': self.embedding_dim,
            'test_queries': test_queries,
            'algorithms': {}
        }

        # Evaluate different algorithms
        algorithms = [
            ('faiss_flat', self.evaluate_faiss_flat),
            ('faiss_ivf', lambda e, c, q: self.evaluate_faiss_ivf(e, c, q, nlist=50)),
            ('faiss_hnsw', self.evaluate_faiss_hnsw),
            ('chromadb', self.evaluate_chromadb),
            ('sklearn_nn', self.evaluate_sklearn_nearestneighbors)
        ]

        for algo_name, algo_func in algorithms:
            try:
                print(f"\n--- Evaluating {algo_name} ---")
                results['algorithms'][algo_name] = algo_func(embeddings, chunks, test_queries)
                print(f"Completed {algo_name} evaluation")
            except Exception as e:
                print(f"Error evaluating {algo_name}: {str(e)}")
                results['algorithms'][algo_name] = {
                    'error': str(e),
                    'algorithm': algo_name
                }

        # Save results
        with open(f'vector_db_evaluation_{strategy}_{max_documents}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate summary
        summary = self._generate_indexing_summary(results)

        with open(f'vector_db_summary_{strategy}_{max_documents}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return results

    def _generate_indexing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of indexing evaluation results"""
        summary = {
            'strategy': results['strategy'],
            'num_documents': results['num_documents'],
            'algorithms_compared': list(results['algorithms'].keys()),
            'performance_comparison': {}
        }

        for algo_name, algo_results in results['algorithms'].items():
            if 'error' not in algo_results:
                summary['performance_comparison'][algo_name] = {
                    'build_time': algo_results['build_time'],
                    'avg_search_time': algo_results['avg_search_time'],
                    'index_size_mb': algo_results['index_size_mb'],
                    'num_vectors': algo_results['num_vectors']
                }

        # Find best performing algorithms
        if summary['performance_comparison']:
            # Best by search speed
            best_search = min(summary['performance_comparison'].items(),
                            key=lambda x: x[1]['avg_search_time'])
            summary['fastest_search'] = {
                'algorithm': best_search[0],
                'avg_search_time': best_search[1]['avg_search_time']
            }

            # Best by build time
            best_build = min(summary['performance_comparison'].items(),
                           key=lambda x: x[1]['build_time'])
            summary['fastest_build'] = {
                'algorithm': best_build[0],
                'build_time': best_build[1]['build_time']
            }

            # Smallest index size
            smallest_index = min(summary['performance_comparison'].items(),
                                key=lambda x: x[1]['index_size_mb'])
            summary['smallest_index'] = {
                'algorithm': smallest_index[0],
                'index_size_mb': smallest_index[1]['index_size_mb']
            }

        return summary

    def compare_accuracy_across_algorithms(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare accuracy across different algorithms using overlap metrics
        """
        print("Comparing accuracy across algorithms...")

        accuracy_comparison = {}
        baseline_algo = 'faiss_flat'  # Use exact search as baseline

        if baseline_algo not in results['algorithms']:
            print(f"Baseline algorithm {baseline_algo} not found")
            return {}

        baseline_results = results['algorithms'][baseline_algo]['results']

        for algo_name, algo_data in results['algorithms'].items():
            if 'error' in algo_data or algo_name == baseline_algo:
                continue

            overlaps = []
            precision_at_5 = []
            precision_at_10 = []

            for i, query_result in enumerate(algo_data['results']):
                baseline_ids = set([r['chunk_id'] for r in baseline_results[i]['results']])
                algo_ids = set([r['chunk_id'] for r in query_result['results']])

                # Calculate overlap
                overlap = len(baseline_ids.intersection(algo_ids)) / len(baseline_ids.union(algo_ids))
                overlaps.append(overlap)

                # Precision at k
                baseline_top5 = set([r['chunk_id'] for r in baseline_results[i]['results'][:5]])
                baseline_top10 = set([r['chunk_id'] for r in baseline_results[i]['results'][:10]])

                algo_top5 = set([r['chunk_id'] for r in query_result['results'][:5]])
                algo_top10 = set([r['chunk_id'] for r in query_result['results'][:10]])

                p5 = len(baseline_top5.intersection(algo_top5)) / 5 if len(algo_top5) >= 5 else 0
                p10 = len(baseline_top10.intersection(algo_top10)) / 10 if len(algo_top10) >= 10 else 0

                precision_at_5.append(p5)
                precision_at_10.append(p10)

            accuracy_comparison[algo_name] = {
                'avg_overlap': np.mean(overlaps),
                'avg_precision_at_5': np.mean(precision_at_5),
                'avg_precision_at_10': np.mean(precision_at_10),
                'std_overlap': np.std(overlaps)
            }

        return accuracy_comparison


def main():
    """Main execution function"""
    # Load chunked data
    try:
        chunks_data = {}
        strategies = ['token_512_50', 'sentence_3', 'semantic_05']

        for strategy in strategies:
            try:
                with open(f'chunks_sample_{strategy}.json', 'r') as f:
                    chunks = json.load(f)
                    chunks_data[strategy] = {'chunks': chunks}
                    print(f"Loaded {len(chunks)} chunks for {strategy}")
            except FileNotFoundError:
                print(f"Warning: Could not find chunks for {strategy}")

        if not chunks_data:
            print("No chunked data found. Please run chunking_strategies.py first.")
            return

        # Initialize evaluator
        evaluator = VectorDBEvaluator(chunks_data)

        # Run evaluation on different strategies and document sizes
        document_sizes = [500, 1000]
        strategies_to_test = ['token_512_50', 'sentence_3']

        for strategy in strategies_to_test:
            if strategy in chunks_data:
                for doc_size in document_sizes:
                    print(f"\n=== Evaluating {strategy} with {doc_size} documents ===")
                    results = evaluator.run_comprehensive_indexing_evaluation(strategy, doc_size)

                    # Compare accuracy
                    accuracy_results = evaluator.compare_accuracy_across_algorithms(results)

                    if accuracy_results:
                        with open(f'accuracy_comparison_{strategy}_{doc_size}.json', 'w') as f:
                            json.dump(accuracy_results, f, indent=2)

                    # Print summary
                    print(f"\nSummary for {strategy} ({doc_size} docs):")
                    for algo, data in results['algorithms'].items():
                        if 'error' not in data:
                            print(f"  {algo}: Build={data['build_time']:.3f}s, "
                                  f"Search={data['avg_search_time']:.3f}s, "
                                  f"Size={data['index_size_mb']:.1f}MB")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()