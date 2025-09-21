"""
Simple Task Two Implementation - Chunking, Retrieval, and Vector DB Evaluation
Demonstrates the assignment requirements without complex dependencies
"""

import pandas as pd
import numpy as np
import json
import time
import re
from collections import defaultdict


class TaskTwoDemo:
    """
    Simple implementation of Task Two assignment requirements:
    1. Different chunking strategies
    2. Various retrieval methods
    3. Vector database indexing evaluation
    """

    def __init__(self, csv_path):
        """Initialize with the StarTech products dataset"""
        # Load data with proper encoding handling
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"[OK] Loaded {len(self.df)} products from StarTech dataset")

    def clean_text(self, text):
        """Clean text for processing"""
        if pd.isna(text):
            return ""
        # Keep only ASCII characters to avoid encoding issues
        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', str(text))
        return cleaned.strip()

    def prepare_products(self, sample_size=200):
        """Prepare product texts for chunking experiments"""
        sample_df = self.df.head(sample_size)
        products = []

        for idx, row in sample_df.iterrows():
            # Create product description
            parts = []

            name = self.clean_text(row['name'])
            brand = self.clean_text(row['brand'])
            category = self.clean_text(row['category'])
            subcategory = self.clean_text(row['subcategory'])
            price = self.clean_text(row['price'])

            if name:
                parts.append(f"Product name: {name}")
            if brand:
                parts.append(f"Brand: {brand}")
            if category:
                parts.append(f"Category: {category}")
            if subcategory:
                parts.append(f"Subcategory: {subcategory}")
            if price:
                parts.append(f"Price: {price}")

            full_text = ". ".join(parts)

            products.append({
                'id': idx,
                'text': full_text,
                'name': name,
                'brand': brand,
                'category': category
            })

        return products

    def experiment_chunking_strategies(self, products):
        """
        Experiment with different chunking strategies (Requirement 1)
        """
        print("\n=== EXPERIMENT 1: CHUNKING STRATEGIES ===")

        results = {}

        # Strategy 1: Token-based chunking
        print("Testing Token-based chunking...")
        start_time = time.time()
        token_chunks = []

        for product in products:
            words = product['text'].split()
            chunk_size = 50  # tokens
            overlap = 10

            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                token_chunks.append({
                    'product_id': product['id'],
                    'text': chunk_text,
                    'method': 'token',
                    'size': len(words[start:end])
                })

                start = end - overlap
                if start >= len(words):
                    break

        token_time = time.time() - start_time
        results['token_based'] = {
            'chunks': len(token_chunks),
            'time': token_time,
            'avg_size': np.mean([c['size'] for c in token_chunks])
        }

        # Strategy 2: Sentence-based chunking
        print("Testing Sentence-based chunking...")
        start_time = time.time()
        sentence_chunks = []

        for product in products:
            sentences = [s.strip() for s in product['text'].split('.') if s.strip()]
            sentences_per_chunk = 3

            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                chunk_text = ". ".join(chunk_sentences)

                sentence_chunks.append({
                    'product_id': product['id'],
                    'text': chunk_text,
                    'method': 'sentence',
                    'size': len(chunk_sentences)
                })

        sentence_time = time.time() - start_time
        results['sentence_based'] = {
            'chunks': len(sentence_chunks),
            'time': sentence_time,
            'avg_size': np.mean([c['size'] for c in sentence_chunks])
        }

        # Strategy 3: Semantic-based chunking (simplified)
        print("Testing Semantic-based chunking...")
        start_time = time.time()
        semantic_chunks = []

        for product in products:
            # Group sentences by keyword similarity
            sentences = [s.strip() for s in product['text'].split('.') if s.strip()]

            if len(sentences) <= 1:
                semantic_chunks.append({
                    'product_id': product['id'],
                    'text': product['text'],
                    'method': 'semantic',
                    'size': len(sentences)
                })
                continue

            current_group = [sentences[0]]
            current_keywords = set(sentences[0].lower().split())

            for sentence in sentences[1:]:
                sent_keywords = set(sentence.lower().split())
                # Simple similarity based on shared keywords
                shared = len(current_keywords.intersection(sent_keywords))
                total = len(current_keywords.union(sent_keywords))
                similarity = shared / total if total > 0 else 0

                if similarity > 0.3:  # Semantic similarity threshold
                    current_group.append(sentence)
                    current_keywords.update(sent_keywords)
                else:
                    # Save current group
                    chunk_text = ". ".join(current_group)
                    semantic_chunks.append({
                        'product_id': product['id'],
                        'text': chunk_text,
                        'method': 'semantic',
                        'size': len(current_group)
                    })

                    current_group = [sentence]
                    current_keywords = sent_keywords

            # Save last group
            if current_group:
                chunk_text = ". ".join(current_group)
                semantic_chunks.append({
                    'product_id': product['id'],
                    'text': chunk_text,
                    'method': 'semantic',
                    'size': len(current_group)
                })

        semantic_time = time.time() - start_time
        results['semantic_based'] = {
            'chunks': len(semantic_chunks),
            'time': semantic_time,
            'avg_size': np.mean([c['size'] for c in semantic_chunks])
        }

        # Print results
        print(f"\nCHUNKING RESULTS:")
        print(f"Token-based:    {results['token_based']['chunks']:4d} chunks, {results['token_based']['time']:.3f}s")
        print(f"Sentence-based: {results['sentence_based']['chunks']:4d} chunks, {results['sentence_based']['time']:.3f}s")
        print(f"Semantic-based: {results['semantic_based']['chunks']:4d} chunks, {results['semantic_based']['time']:.3f}s")

        return results, token_chunks

    def experiment_retrieval_methods(self, chunks):
        """
        Evaluate various retrieval methods (Requirement 2)
        """
        print("\n=== EXPERIMENT 2: RETRIEVAL METHODS ===")

        # Test queries
        test_queries = [
            "Intel Core i7 gaming PC",
            "AMD Ryzen desktop computer",
            "budget desktop PC",
            "ASUS brand computer",
            "high performance gaming"
        ]

        results = {}

        # Method 1: Basic keyword matching (sparse retrieval)
        print("Testing Sparse retrieval (BM25-like)...")
        start_time = time.time()

        def score_document(query, doc_text):
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())
            intersection = query_words.intersection(doc_words)
            return len(intersection) / len(query_words) if query_words else 0

        sparse_results = []
        for query in test_queries:
            query_results = []
            for chunk in chunks:
                score = score_document(query, chunk['text'])
                if score > 0:
                    query_results.append((chunk, score))

            # Sort by score and take top 10
            query_results.sort(key=lambda x: x[1], reverse=True)
            sparse_results.append(query_results[:10])

        sparse_time = time.time() - start_time
        results['sparse_retrieval'] = {
            'time': sparse_time,
            'avg_results': np.mean([len(r) for r in sparse_results])
        }

        # Method 2: TF-IDF like scoring
        print("Testing TF-IDF-like retrieval...")
        start_time = time.time()

        # Build term frequencies
        term_freq = defaultdict(lambda: defaultdict(int))
        doc_freq = defaultdict(int)

        for i, chunk in enumerate(chunks):
            words = chunk['text'].lower().split()
            unique_words = set(words)
            for word in words:
                term_freq[i][word] += 1
            for word in unique_words:
                doc_freq[word] += 1

        def tfidf_score(query, doc_id):
            score = 0
            query_words = query.lower().split()
            for word in query_words:
                tf = term_freq[doc_id][word]
                idf = np.log(len(chunks) / (doc_freq[word] + 1))
                score += tf * idf
            return score

        tfidf_results = []
        for query in test_queries:
            query_results = []
            for i, chunk in enumerate(chunks):
                score = tfidf_score(query, i)
                if score > 0:
                    query_results.append((chunk, score))

            query_results.sort(key=lambda x: x[1], reverse=True)
            tfidf_results.append(query_results[:10])

        tfidf_time = time.time() - start_time
        results['tfidf_retrieval'] = {
            'time': tfidf_time,
            'avg_results': np.mean([len(r) for r in tfidf_results])
        }

        # Method 3: Hybrid approach (combining sparse and dense-like features)
        print("Testing Hybrid retrieval...")
        start_time = time.time()

        def hybrid_score(query, chunk):
            # Combine keyword matching with text length and product category features
            keyword_score = score_document(query, chunk['text'])

            # Bonus for product name matches
            name_bonus = 0.5 if any(word in chunk['text'].lower() for word in query.lower().split()) else 0

            # Text length normalization
            length_norm = min(1.0, len(chunk['text']) / 200)

            return keyword_score + name_bonus * length_norm

        hybrid_results = []
        for query in test_queries:
            query_results = []
            for chunk in chunks:
                score = hybrid_score(query, chunk)
                if score > 0:
                    query_results.append((chunk, score))

            query_results.sort(key=lambda x: x[1], reverse=True)
            hybrid_results.append(query_results[:10])

        hybrid_time = time.time() - start_time
        results['hybrid_retrieval'] = {
            'time': hybrid_time,
            'avg_results': np.mean([len(r) for r in hybrid_results])
        }

        # Method 4: Reranking simulation
        print("Testing Reranking approach...")
        start_time = time.time()

        def rerank_score(query, chunk):
            # Simulate cross-encoder reranking with more sophisticated scoring
            base_score = score_document(query, chunk['text'])

            # Position of query words in text
            query_words = query.lower().split()
            text_words = chunk['text'].lower().split()

            position_bonus = 0
            for word in query_words:
                if word in text_words:
                    pos = text_words.index(word)
                    # Earlier positions get higher scores
                    position_bonus += (len(text_words) - pos) / len(text_words)

            return base_score + 0.3 * position_bonus / len(query_words)

        rerank_results = []
        for query in test_queries:
            # First get initial results
            initial_results = []
            for chunk in chunks:
                score = score_document(query, chunk['text'])
                if score > 0:
                    initial_results.append((chunk, score))

            initial_results.sort(key=lambda x: x[1], reverse=True)
            top_candidates = initial_results[:20]  # Get top 20 for reranking

            # Rerank top candidates
            reranked = []
            for chunk, _ in top_candidates:
                new_score = rerank_score(query, chunk)
                reranked.append((chunk, new_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            rerank_results.append(reranked[:10])

        rerank_time = time.time() - start_time
        results['reranking'] = {
            'time': rerank_time,
            'avg_results': np.mean([len(r) for r in rerank_results])
        }

        # Print results
        print(f"\nRETRIEVAL RESULTS:")
        for method, data in results.items():
            print(f"{method:20}: {data['time']:.3f}s, avg {data['avg_results']:.1f} results/query")

        return results

    def experiment_vector_indexing(self, chunks):
        """
        Test multiple vector database indexing algorithms (Requirement 3)
        """
        print("\n=== EXPERIMENT 3: VECTOR INDEXING ALGORITHMS ===")

        # Simulate vector embeddings (normally would use sentence transformers)
        print("Creating simulated embeddings...")

        # Simple bag-of-words vector representation
        vocab = set()
        for chunk in chunks:
            vocab.update(chunk['text'].lower().split())
        vocab = sorted(list(vocab))
        vocab_size = len(vocab)

        print(f"Vocabulary size: {vocab_size}")

        def text_to_vector(text):
            words = text.lower().split()
            vector = np.zeros(vocab_size)
            for word in words:
                if word in vocab:
                    idx = vocab.index(word)
                    vector[idx] += 1
            return vector

        # Create vectors for all chunks
        vectors = []
        for chunk in chunks:
            vectors.append(text_to_vector(chunk['text']))
        vectors = np.array(vectors)

        print(f"Created {len(vectors)} vectors of dimension {vocab_size}")

        results = {}

        # Algorithm 1: Brute Force (Flat index)
        print("Testing Brute Force indexing...")
        start_time = time.time()

        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            return dot_product / norm_product if norm_product != 0 else 0

        # Test search
        query = "Intel Core i7 gaming"
        query_vector = text_to_vector(query)

        similarities = []
        for vector in vectors:
            sim = cosine_similarity(query_vector, vector)
            similarities.append(sim)

        # Get top 10
        top_indices = np.argsort(similarities)[::-1][:10]

        brute_force_time = time.time() - start_time
        results['brute_force'] = {
            'time': brute_force_time,
            'top_score': similarities[top_indices[0]] if top_indices.size > 0 else 0
        }

        # Algorithm 2: Approximate search (LSH-like)
        print("Testing Approximate indexing...")
        start_time = time.time()

        # Simple random projection for approximation
        projection_dim = min(100, vocab_size // 4)
        projection_matrix = np.random.randn(vocab_size, projection_dim)

        # Project all vectors
        projected_vectors = vectors @ projection_matrix
        projected_query = query_vector @ projection_matrix

        # Search in reduced space
        approx_similarities = []
        for proj_vector in projected_vectors:
            sim = cosine_similarity(projected_query, proj_vector)
            approx_similarities.append(sim)

        approx_top_indices = np.argsort(approx_similarities)[::-1][:10]

        approx_time = time.time() - start_time
        results['approximate'] = {
            'time': approx_time,
            'top_score': approx_similarities[approx_top_indices[0]] if approx_top_indices.size > 0 else 0
        }

        # Algorithm 3: Hierarchical clustering approach
        print("Testing Hierarchical indexing...")
        start_time = time.time()

        # Simple clustering: group vectors by their dominant features
        num_clusters = min(20, len(vectors) // 10)
        cluster_centers = vectors[np.random.choice(len(vectors), num_clusters, replace=False)]

        # Assign each vector to nearest cluster
        cluster_assignments = []
        for vector in vectors:
            distances = [np.linalg.norm(vector - center) for center in cluster_centers]
            cluster_assignments.append(np.argmin(distances))

        # Search within clusters
        query_distances = [np.linalg.norm(query_vector - center) for center in cluster_centers]
        nearest_cluster = np.argmin(query_distances)

        # Search only within the nearest cluster
        cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == nearest_cluster]
        cluster_similarities = []

        for idx in cluster_indices:
            sim = cosine_similarity(query_vector, vectors[idx])
            cluster_similarities.append((idx, sim))

        cluster_similarities.sort(key=lambda x: x[1], reverse=True)

        hierarchical_time = time.time() - start_time
        results['hierarchical'] = {
            'time': hierarchical_time,
            'top_score': cluster_similarities[0][1] if cluster_similarities else 0,
            'cluster_size': len(cluster_indices)
        }

        # Print results
        print(f"\nVECTOR INDEXING RESULTS:")
        print(f"Brute Force:   {results['brute_force']['time']:.4f}s, score: {results['brute_force']['top_score']:.3f}")
        print(f"Approximate:   {results['approximate']['time']:.4f}s, score: {results['approximate']['top_score']:.3f}")
        print(f"Hierarchical:  {results['hierarchical']['time']:.4f}s, score: {results['hierarchical']['top_score']:.3f}")

        return results

    def run_complete_evaluation(self):
        """Run the complete Task Two evaluation"""
        print("STARTECH PRODUCTS - TASK TWO ASSIGNMENT")
        print("=" * 50)
        print("Evaluating: Chunking + Retrieval + Vector Indexing")
        print("=" * 50)

        # Prepare data
        products = self.prepare_products(200)
        print(f"[OK] Prepared {len(products)} product descriptions")

        # Run all experiments
        chunking_results, chunks = self.experiment_chunking_strategies(products)
        retrieval_results = self.experiment_retrieval_methods(chunks[:500])  # Use subset for retrieval
        indexing_results = self.experiment_vector_indexing(chunks[:200])    # Use subset for indexing

        # Compile final results
        final_results = {
            'dataset_info': {
                'total_products': len(self.df),
                'sample_products': len(products),
                'total_chunks': len(chunks)
            },
            'chunking_strategies': chunking_results,
            'retrieval_methods': retrieval_results,
            'vector_indexing': indexing_results,
            'summary': {
                'best_chunking': min(chunking_results.items(), key=lambda x: x[1]['time'])[0],
                'fastest_retrieval': min(retrieval_results.items(), key=lambda x: x[1]['time'])[0],
                'fastest_indexing': min(indexing_results.items(), key=lambda x: x[1]['time'])[0]
            }
        }

        # Save results
        with open('task_two_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"\n" + "=" * 50)
        print("TASK TWO ASSIGNMENT COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"[OK] Tested 3 chunking strategies")
        print(f"[OK] Evaluated 4 retrieval methods")
        print(f"[OK] Compared 3 vector indexing algorithms")
        print(f"[OK] Results saved to 'task_two_results.json'")
        print(f"\nBest performers:")
        print(f"  Chunking: {final_results['summary']['best_chunking']}")
        print(f"  Retrieval: {final_results['summary']['fastest_retrieval']}")
        print(f"  Indexing: {final_results['summary']['fastest_indexing']}")

        return final_results


def main():
    """Main execution function"""
    demo = TaskTwoDemo('startech_products.csv')
    results = demo.run_complete_evaluation()


if __name__ == "__main__":
    main()