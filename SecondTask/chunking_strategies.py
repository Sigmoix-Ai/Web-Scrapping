"""
Chunking Strategies using Chonkie for StarTech Product Data
This module implements different chunking strategies (token-based, semantic-based, sentence-based)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
from chonkie import TokenChunker, SentenceChunker, SemanticChunker
from transformers import AutoTokenizer
import time
import json


class ProductDataChunker:
    """
    A class to implement different chunking strategies for product data
    """

    def __init__(self, csv_path: str, max_products: int = 1000):
        """Initialize with product data"""
        self.df = pd.read_csv(csv_path).head(max_products)  # Limit products
        self.products = self._prepare_product_texts()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def _prepare_product_texts(self) -> List[Dict[str, Any]]:
        """Convert product data to text for chunking"""
        products = []

        for idx, row in self.df.iterrows():
            # Combine product information into comprehensive text
            text_parts = []

            if pd.notna(row['name']):
                text_parts.append(f"Product: {row['name']}")

            if pd.notna(row['brand']):
                text_parts.append(f"Brand: {row['brand']}")

            if pd.notna(row['category']):
                text_parts.append(f"Category: {row['category']}")

            if pd.notna(row['subcategory']):
                text_parts.append(f"Subcategory: {row['subcategory']}")

            if pd.notna(row['price']):
                text_parts.append(f"Price: {row['price']}")

            if pd.notna(row['model']):
                text_parts.append(f"Model: {row['model']}")

            if pd.notna(row['availability']):
                text_parts.append(f"Availability: {row['availability']}")

            if pd.notna(row['rating']):
                text_parts.append(f"Rating: {row['rating']}")

            full_text = ". ".join(text_parts)

            products.append({
                'id': idx,
                'text': full_text,
                'metadata': {
                    'name': row['name'] if pd.notna(row['name']) else '',
                    'brand': row['brand'] if pd.notna(row['brand']) else '',
                    'category': row['category'] if pd.notna(row['category']) else '',
                    'price': row['price'] if pd.notna(row['price']) else '',
                    'url': row['product_url'] if pd.notna(row['product_url']) else ''
                }
            })

        return products

    def token_based_chunking(self, chunk_size: int = 512, overlap: int = 50) -> Tuple[List[Dict], Dict]:
        """
        Implement token-based chunking using Chonkie TokenChunker
        """
        print(f"Starting token-based chunking with chunk_size={chunk_size}, overlap={overlap}")
        start_time = time.time()

        # Initialize TokenChunker
        chunker = TokenChunker(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        chunks = []
        chunk_id = 0

        for product in self.products:
            product_chunks = chunker.chunk(product['text'])

            for chunk_text in product_chunks:
                # Ensure chunk_text is a string
                chunk_str = str(chunk_text) if chunk_text is not None else ""
                if chunk_str.strip():  # Only process non-empty chunks
                    chunks.append({
                        'chunk_id': chunk_id,
                        'product_id': product['id'],
                        'text': chunk_str,
                        'metadata': product['metadata'],
                        'chunk_type': 'token_based',
                        'chunk_size': len(self.tokenizer.encode(chunk_str))
                    })
                    chunk_id += 1

        end_time = time.time()

        stats = {
            'chunking_method': 'token_based',
            'total_chunks': len(chunks),
            'total_products': len(self.products),
            'avg_chunks_per_product': len(chunks) / len(self.products),
            'processing_time': end_time - start_time,
            'chunk_size_config': chunk_size,
            'overlap_config': overlap,
            'avg_chunk_size': np.mean([chunk['chunk_size'] for chunk in chunks]),
            'min_chunk_size': min([chunk['chunk_size'] for chunk in chunks]),
            'max_chunk_size': max([chunk['chunk_size'] for chunk in chunks])
        }

        print(f"Token-based chunking completed: {len(chunks)} chunks in {stats['processing_time']:.2f}s")
        return chunks, stats

    def sentence_based_chunking(self, chunk_size: int = 3) -> Tuple[List[Dict], Dict]:
        """
        Implement sentence-based chunking using Chonkie SentenceChunker
        """
        print(f"Starting sentence-based chunking with chunk_size={chunk_size} sentences")
        start_time = time.time()

        # Initialize SentenceChunker
        chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=1
        )

        chunks = []
        chunk_id = 0

        for product in self.products:
            product_chunks = chunker.chunk(product['text'])

            for chunk_text in product_chunks:
                # Extract text from SentenceChunk object
                chunk_str = str(chunk_text) if chunk_text is not None else ""
                if chunk_str.strip():  # Only process non-empty chunks
                    sentence_count = len([s for s in chunk_str.split('.') if s.strip()])

                    chunks.append({
                        'chunk_id': chunk_id,
                        'product_id': product['id'],
                        'text': chunk_str,
                        'metadata': product['metadata'],
                        'chunk_type': 'sentence_based',
                        'sentence_count': sentence_count,
                        'chunk_size': len(self.tokenizer.encode(chunk_str))
                    })
                    chunk_id += 1

        end_time = time.time()

        stats = {
            'chunking_method': 'sentence_based',
            'total_chunks': len(chunks),
            'total_products': len(self.products),
            'avg_chunks_per_product': len(chunks) / len(self.products),
            'processing_time': end_time - start_time,
            'sentences_per_chunk_config': chunk_size,
            'avg_sentences_per_chunk': np.mean([chunk['sentence_count'] for chunk in chunks]),
            'avg_chunk_size': np.mean([chunk['chunk_size'] for chunk in chunks]),
            'min_chunk_size': min([chunk['chunk_size'] for chunk in chunks]),
            'max_chunk_size': max([chunk['chunk_size'] for chunk in chunks])
        }

        print(f"Sentence-based chunking completed: {len(chunks)} chunks in {stats['processing_time']:.2f}s")
        return chunks, stats

    def semantic_based_chunking(self, similarity_threshold: float = 0.5) -> Tuple[List[Dict], Dict]:
        """
        Implement semantic-based chunking using Chonkie SemanticChunker
        """
        print(f"Starting semantic-based chunking with similarity_threshold={similarity_threshold}")
        start_time = time.time()

        # Initialize SemanticChunker
        chunker = SemanticChunker(
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            similarity_threshold=similarity_threshold
        )

        chunks = []
        chunk_id = 0

        for product in self.products:
            try:
                product_chunks = chunker.chunk(product['text'])

                for chunk_text in product_chunks:
                    # Extract text from semantic chunk object
                    chunk_str = str(chunk_text) if chunk_text is not None else ""
                    if chunk_str.strip():  # Only process non-empty chunks
                        chunks.append({
                            'chunk_id': chunk_id,
                            'product_id': product['id'],
                            'text': chunk_str,
                            'metadata': product['metadata'],
                            'chunk_type': 'semantic_based',
                            'chunk_size': len(self.tokenizer.encode(chunk_str))
                        })
                        chunk_id += 1
            except Exception as e:
                # Fallback to sentence chunking for problematic texts
                print(f"Semantic chunking failed for product {product['id']}, using fallback")
                sentences = product['text'].split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append({
                            'chunk_id': chunk_id,
                            'product_id': product['id'],
                            'text': sentence.strip(),
                            'metadata': product['metadata'],
                            'chunk_type': 'semantic_based_fallback',
                            'chunk_size': len(self.tokenizer.encode(sentence.strip()))
                        })
                        chunk_id += 1

        end_time = time.time()

        stats = {
            'chunking_method': 'semantic_based',
            'total_chunks': len(chunks),
            'total_products': len(self.products),
            'avg_chunks_per_product': len(chunks) / len(self.products),
            'processing_time': end_time - start_time,
            'similarity_threshold_config': similarity_threshold,
            'avg_chunk_size': np.mean([chunk['chunk_size'] for chunk in chunks]),
            'min_chunk_size': min([chunk['chunk_size'] for chunk in chunks]) if chunks else 0,
            'max_chunk_size': max([chunk['chunk_size'] for chunk in chunks]) if chunks else 0
        }

        print(f"Semantic-based chunking completed: {len(chunks)} chunks in {stats['processing_time']:.2f}s")
        return chunks, stats

    def run_all_chunking_strategies(self) -> Dict[str, Any]:
        """
        Run all chunking strategies and compare results
        """
        print("=== Running All Chunking Strategies ===")

        results = {}

        # Token-based chunking with different configurations
        token_chunks_512, token_stats_512 = self.token_based_chunking(chunk_size=512, overlap=50)
        token_chunks_256, token_stats_256 = self.token_based_chunking(chunk_size=256, overlap=25)

        # Sentence-based chunking with different configurations
        sentence_chunks_3, sentence_stats_3 = self.sentence_based_chunking(chunk_size=3)
        sentence_chunks_5, sentence_stats_5 = self.sentence_based_chunking(chunk_size=5)

        # Semantic-based chunking with different thresholds
        semantic_chunks_05, semantic_stats_05 = self.semantic_based_chunking(similarity_threshold=0.5)
        semantic_chunks_07, semantic_stats_07 = self.semantic_based_chunking(similarity_threshold=0.7)

        results = {
            'token_512_50': {'chunks': token_chunks_512, 'stats': token_stats_512},
            'token_256_25': {'chunks': token_chunks_256, 'stats': token_stats_256},
            'sentence_3': {'chunks': sentence_chunks_3, 'stats': sentence_stats_3},
            'sentence_5': {'chunks': sentence_chunks_5, 'stats': sentence_stats_5},
            'semantic_05': {'chunks': semantic_chunks_05, 'stats': semantic_stats_05},
            'semantic_07': {'chunks': semantic_chunks_07, 'stats': semantic_stats_07}
        }

        # Save results
        self._save_chunking_results(results)

        return results

    def _save_chunking_results(self, results: Dict[str, Any]):
        """Save chunking results to files"""
        # Save statistics
        stats_summary = {
            strategy: data['stats'] for strategy, data in results.items()
        }

        with open('chunking_statistics.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)

        # Save chunks for each strategy (sample)
        for strategy, data in results.items():
            sample_chunks = data['chunks'][:100]  # Save first 100 chunks as sample
            with open(f'chunks_sample_{strategy}.json', 'w') as f:
                json.dump(sample_chunks, f, indent=2)

        print("Chunking results saved to chunking_statistics.json and sample files")


def main():
    """Main execution function"""
    # Initialize chunker with the CSV file
    chunker = ProductDataChunker('startech_products.csv')

    # Run all chunking strategies
    results = chunker.run_all_chunking_strategies()

    # Print summary
    print("\n=== Chunking Strategy Comparison ===")
    for strategy, data in results.items():
        stats = data['stats']
        print(f"\n{strategy.upper()}:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg chunks per product: {stats['avg_chunks_per_product']:.2f}")
        print(f"  Processing time: {stats['processing_time']:.2f}s")
        print(f"  Avg chunk size (tokens): {stats['avg_chunk_size']:.1f}")


if __name__ == "__main__":
    main()