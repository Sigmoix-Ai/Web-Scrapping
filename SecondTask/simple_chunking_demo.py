"""
Simple demonstration of different chunking strategies using Chonkie
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import json
import re

# Import Chonkie chunkers - updated for latest version
from chonkie import TokenChunker, SentenceChunker, SemanticChunker


class SimpleChunker:
    """A simple chunker to demonstrate different strategies"""

    def __init__(self, csv_path: str):
        """Initialize with CSV data"""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} products from {csv_path}")

    def prepare_sample_data(self, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Prepare a sample of product data for chunking"""
        sample_df = self.df.head(sample_size)
        products = []

        for idx, row in sample_df.iterrows():
            text_parts = []

            # Build product description
            if pd.notna(row['name']):
                text_parts.append(f"Product: {row['name']}")
            if pd.notna(row['brand']):
                text_parts.append(f"Brand: {row['brand']}")
            if pd.notna(row['category']):
                text_parts.append(f"Category: {row['category']}")
            if pd.notna(row['price']):
                text_parts.append(f"Price: {row['price']}")

            full_text = ". ".join(text_parts)

            products.append({
                'id': idx,
                'text': full_text,
                'metadata': {
                    'name': str(row['name']) if pd.notna(row['name']) else '',
                    'brand': str(row['brand']) if pd.notna(row['brand']) else '',
                    'category': str(row['category']) if pd.notna(row['category']) else '',
                    'price': str(row['price']) if pd.notna(row['price']) else ''
                }
            })

        return products

    def token_chunking_demo(self, products: List[Dict], chunk_size: int = 100):
        """Demonstrate token-based chunking"""
        print(f"\n=== Token-based Chunking (chunk_size={chunk_size}) ===")
        start_time = time.time()

        # Initialize token chunker
        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=20)

        all_chunks = []
        for product in products:
            try:
                chunks = chunker.chunk(product['text'])
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'product_id': product['id'],
                        'chunk_id': len(all_chunks),
                        'text': chunk,
                        'method': 'token',
                        'length': len(chunk),
                        'metadata': product['metadata']
                    })
            except Exception as e:
                print(f"Error chunking product {product['id']}: {e}")

        end_time = time.time()

        print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.2f} seconds")
        print(f"Average chunks per product: {len(all_chunks) / len(products):.2f}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"  Chunk {i+1}: {chunk['text'][:100]}...")

        return all_chunks

    def sentence_chunking_demo(self, products: List[Dict], chunk_size: int = 2):
        """Demonstrate sentence-based chunking"""
        print(f"\n=== Sentence-based Chunking (chunk_size={chunk_size} sentences) ===")
        start_time = time.time()

        # Initialize sentence chunker
        chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=1)

        all_chunks = []
        for product in products:
            try:
                chunks = chunker.chunk(product['text'])
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'product_id': product['id'],
                        'chunk_id': len(all_chunks),
                        'text': chunk,
                        'method': 'sentence',
                        'length': len(chunk),
                        'metadata': product['metadata']
                    })
            except Exception as e:
                print(f"Error chunking product {product['id']}: {e}")

        end_time = time.time()

        print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.2f} seconds")
        print(f"Average chunks per product: {len(all_chunks) / len(products):.2f}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"  Chunk {i+1}: {chunk['text'][:100]}...")

        return all_chunks

    def semantic_chunking_demo(self, products: List[Dict]):
        """Demonstrate semantic-based chunking"""
        print(f"\n=== Semantic-based Chunking ===")
        start_time = time.time()

        try:
            # Initialize semantic chunker with basic settings
            chunker = SemanticChunker(chunk_size=512, similarity_threshold=0.5)

            all_chunks = []
            for product in products[:10]:  # Test with fewer products for semantic chunking
                try:
                    chunks = chunker.chunk(product['text'])
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            'product_id': product['id'],
                            'chunk_id': len(all_chunks),
                            'text': chunk,
                            'method': 'semantic',
                            'length': len(chunk),
                            'metadata': product['metadata']
                        })
                except Exception as e:
                    print(f"Error chunking product {product['id']}: {e}")
                    # Fallback to simple splitting
                    sentences = product['text'].split('. ')
                    for j, sentence in enumerate(sentences):
                        if sentence.strip():
                            all_chunks.append({
                                'product_id': product['id'],
                                'chunk_id': len(all_chunks),
                                'text': sentence.strip(),
                                'method': 'semantic_fallback',
                                'length': len(sentence.strip()),
                                'metadata': product['metadata']
                            })

            end_time = time.time()

            print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.2f} seconds")
            print(f"Average chunks per product: {len(all_chunks) / 10:.2f}")

            # Show sample chunks
            print("\nSample chunks:")
            for i, chunk in enumerate(all_chunks[:3]):
                print(f"  Chunk {i+1}: {chunk['text'][:100]}...")

            return all_chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}")
            print("Falling back to sentence-based chunking...")
            return self.sentence_chunking_demo(products[:10])

    def compare_chunking_strategies(self, sample_size: int = 50):
        """Compare all chunking strategies"""
        print(f"=== Comparing Chunking Strategies on {sample_size} products ===")

        # Prepare sample data
        products = self.prepare_sample_data(sample_size)

        # Test all strategies
        results = {}

        # Token chunking with different sizes
        results['token_100'] = self.token_chunking_demo(products, chunk_size=100)
        results['token_200'] = self.token_chunking_demo(products, chunk_size=200)

        # Sentence chunking with different sizes
        results['sentence_2'] = self.sentence_chunking_demo(products, chunk_size=2)
        results['sentence_3'] = self.sentence_chunking_demo(products, chunk_size=3)

        # Semantic chunking
        results['semantic'] = self.semantic_chunking_demo(products)

        # Generate comparison summary
        print(f"\n=== Chunking Strategy Comparison Summary ===")
        for strategy, chunks in results.items():
            avg_length = np.mean([chunk['length'] for chunk in chunks])
            print(f"{strategy:15} | {len(chunks):4d} chunks | avg length: {avg_length:6.1f} chars")

        # Save results
        summary = {}
        for strategy, chunks in results.items():
            summary[strategy] = {
                'total_chunks': len(chunks),
                'avg_chunk_length': float(np.mean([chunk['length'] for chunk in chunks])),
                'sample_chunks': chunks[:5]  # Save first 5 as sample
            }

        with open('chunking_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to 'chunking_demo_results.json'")
        return results


def main():
    """Main demonstration function"""
    print("StarTech Products Chunking Strategy Demonstration")
    print("=" * 50)

    # Initialize chunker
    chunker = SimpleChunker('startech_products.csv')

    # Run comparison with sample data
    results = chunker.compare_chunking_strategies(sample_size=50)

    print(f"\nDemonstration completed successfully!")
    print(f"Tested chunking strategies on 50 StarTech products")
    print(f"Total strategies tested: {len(results)}")


if __name__ == "__main__":
    main()