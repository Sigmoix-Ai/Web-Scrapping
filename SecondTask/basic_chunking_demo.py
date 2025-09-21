"""
Basic Chunking Demonstration for StarTech Products
This demonstrates different chunking strategies without complex dependencies
"""

import pandas as pd
import numpy as np
import time
import json
import re


class BasicChunker:
    """Basic implementation of different chunking strategies"""

    def __init__(self, csv_path: str):
        """Initialize with CSV data"""
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"Loaded {len(self.df)} products from {csv_path}")

    def clean_text(self, text):
        """Clean text for better processing"""
        if pd.isna(text):
            return ""
        # Remove special characters that might cause encoding issues
        text = str(text).encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    def prepare_sample_data(self, sample_size: int = 100):
        """Prepare sample product data"""
        sample_df = self.df.head(sample_size)
        products = []

        for idx, row in sample_df.iterrows():
            text_parts = []

            # Build product description with cleaned text
            name = self.clean_text(row['name'])
            brand = self.clean_text(row['brand'])
            category = self.clean_text(row['category'])
            price = self.clean_text(row['price'])

            if name:
                text_parts.append(f"Product: {name}")
            if brand:
                text_parts.append(f"Brand: {brand}")
            if category:
                text_parts.append(f"Category: {category}")
            if price:
                text_parts.append(f"Price: {price}")

            full_text = ". ".join(text_parts)

            products.append({
                'id': idx,
                'text': full_text,
                'metadata': {
                    'name': name,
                    'brand': brand,
                    'category': category,
                    'price': price
                }
            })

        return products

    def token_based_chunking(self, products, chunk_size=100, overlap=20):
        """Simple token-based chunking"""
        print(f"\n=== Token-based Chunking (size: {chunk_size}, overlap: {overlap}) ===")
        start_time = time.time()

        all_chunks = []
        for product in products:
            text = product['text']
            words = text.split()

            # Create overlapping chunks
            start = 0
            chunk_id = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                all_chunks.append({
                    'product_id': product['id'],
                    'chunk_id': len(all_chunks),
                    'text': chunk_text,
                    'method': 'token',
                    'word_count': len(words[start:end]),
                    'char_count': len(chunk_text),
                    'metadata': product['metadata']
                })

                # Move to next chunk with overlap
                start = end - overlap
                if start >= len(words):
                    break

        end_time = time.time()

        print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.3f} seconds")
        print(f"Average chunks per product: {len(all_chunks) / len(products):.2f}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
            print(f"  Chunk {i+1}: {preview}")

        return all_chunks

    def sentence_based_chunking(self, products, sentences_per_chunk=2, overlap=1):
        """Simple sentence-based chunking"""
        print(f"\n=== Sentence-based Chunking (sentences: {sentences_per_chunk}, overlap: {overlap}) ===")
        start_time = time.time()

        all_chunks = []
        for product in products:
            text = product['text']
            # Split by periods and clean up
            sentences = [s.strip() for s in text.split('.') if s.strip()]

            # Create chunks of sentences
            start = 0
            while start < len(sentences):
                end = min(start + sentences_per_chunk, len(sentences))
                chunk_sentences = sentences[start:end]
                chunk_text = ". ".join(chunk_sentences) + "."

                all_chunks.append({
                    'product_id': product['id'],
                    'chunk_id': len(all_chunks),
                    'text': chunk_text,
                    'method': 'sentence',
                    'sentence_count': len(chunk_sentences),
                    'char_count': len(chunk_text),
                    'metadata': product['metadata']
                })

                # Move to next chunk with overlap
                start = end - overlap
                if start >= len(sentences):
                    break

        end_time = time.time()

        print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.3f} seconds")
        print(f"Average chunks per product: {len(all_chunks) / len(products):.2f}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
            print(f"  Chunk {i+1}: {preview}")

        return all_chunks

    def semantic_based_chunking_simple(self, products, similarity_threshold=0.7):
        """Simple semantic chunking based on keyword similarity"""
        print(f"\n=== Semantic-based Chunking (similarity: {similarity_threshold}) ===")
        start_time = time.time()

        all_chunks = []
        for product in products:
            text = product['text']
            sentences = [s.strip() for s in text.split('.') if s.strip()]

            if len(sentences) <= 1:
                # Single sentence or very short text
                all_chunks.append({
                    'product_id': product['id'],
                    'chunk_id': len(all_chunks),
                    'text': text,
                    'method': 'semantic',
                    'sentence_count': len(sentences),
                    'char_count': len(text),
                    'metadata': product['metadata']
                })
                continue

            # Group similar sentences based on shared keywords
            current_chunk = [sentences[0]]
            chunk_keywords = set(sentences[0].lower().split())

            for sentence in sentences[1:]:
                sentence_keywords = set(sentence.lower().split())
                # Calculate Jaccard similarity
                intersection = chunk_keywords.intersection(sentence_keywords)
                union = chunk_keywords.union(sentence_keywords)
                similarity = len(intersection) / len(union) if union else 0

                if similarity >= similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(sentence)
                    chunk_keywords.update(sentence_keywords)
                else:
                    # Start new chunk
                    chunk_text = ". ".join(current_chunk) + "."
                    all_chunks.append({
                        'product_id': product['id'],
                        'chunk_id': len(all_chunks),
                        'text': chunk_text,
                        'method': 'semantic',
                        'sentence_count': len(current_chunk),
                        'char_count': len(chunk_text),
                        'metadata': product['metadata']
                    })

                    current_chunk = [sentence]
                    chunk_keywords = set(sentence.lower().split())

            # Add the last chunk
            if current_chunk:
                chunk_text = ". ".join(current_chunk) + "."
                all_chunks.append({
                    'product_id': product['id'],
                    'chunk_id': len(all_chunks),
                    'text': chunk_text,
                    'method': 'semantic',
                    'sentence_count': len(current_chunk),
                    'char_count': len(chunk_text),
                    'metadata': product['metadata']
                })

        end_time = time.time()

        print(f"Generated {len(all_chunks)} chunks in {end_time - start_time:.3f} seconds")
        print(f"Average chunks per product: {len(all_chunks) / len(products):.2f}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
            print(f"  Chunk {i+1}: {preview}")

        return all_chunks

    def compare_chunking_strategies(self, sample_size=50):
        """Compare all chunking strategies"""
        print(f"=== Comparing Chunking Strategies on {sample_size} products ===")

        # Prepare sample data
        products = self.prepare_sample_data(sample_size)

        results = {}

        # Token-based chunking with different sizes
        results['token_50'] = self.token_based_chunking(products, chunk_size=50, overlap=10)
        results['token_100'] = self.token_based_chunking(products, chunk_size=100, overlap=20)

        # Sentence-based chunking
        results['sentence_2'] = self.sentence_based_chunking(products, sentences_per_chunk=2, overlap=1)
        results['sentence_3'] = self.sentence_based_chunking(products, sentences_per_chunk=3, overlap=1)

        # Semantic chunking
        results['semantic_07'] = self.semantic_based_chunking_simple(products, similarity_threshold=0.7)
        results['semantic_05'] = self.semantic_based_chunking_simple(products, similarity_threshold=0.5)

        # Generate comparison summary
        print(f"\n=== Chunking Strategy Comparison Summary ===")
        print(f"{'Strategy':<15} | {'Chunks':<6} | {'Avg Length':<10} | {'Avg Chunks/Product'}")
        print("-" * 60)

        summary_stats = {}
        for strategy, chunks in results.items():
            total_chunks = len(chunks)
            avg_length = np.mean([chunk['char_count'] for chunk in chunks])
            avg_chunks_per_product = total_chunks / len(products)

            print(f"{strategy:<15} | {total_chunks:<6} | {avg_length:<10.1f} | {avg_chunks_per_product:<15.2f}")

            summary_stats[strategy] = {
                'total_chunks': total_chunks,
                'avg_chunk_length': float(avg_length),
                'avg_chunks_per_product': float(avg_chunks_per_product),
                'total_products': len(products),
                'sample_chunks': [
                    {
                        'text': chunk['text'],
                        'char_count': chunk['char_count'],
                        'metadata': chunk['metadata']
                    }
                    for chunk in chunks[:3]  # Save first 3 as sample
                ]
            }

        # Save results
        with open('basic_chunking_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to 'basic_chunking_results.json'")

        # Analysis insights
        print(f"\n=== Analysis Insights ===")

        # Find most efficient strategies
        efficiency_scores = {}
        for strategy, stats in summary_stats.items():
            # Lower chunks per product = more efficient
            efficiency_scores[strategy] = stats['avg_chunks_per_product']

        most_efficient = min(efficiency_scores.items(), key=lambda x: x[1])
        least_efficient = max(efficiency_scores.items(), key=lambda x: x[1])

        print(f"Most efficient (fewest chunks): {most_efficient[0]} ({most_efficient[1]:.2f} chunks/product)")
        print(f"Least efficient (most chunks): {least_efficient[0]} ({least_efficient[1]:.2f} chunks/product)")

        # Find optimal chunk sizes
        chunk_sizes = [(strategy, stats['avg_chunk_length']) for strategy, stats in summary_stats.items()]
        chunk_sizes.sort(key=lambda x: x[1])

        print(f"Smallest chunks: {chunk_sizes[0][0]} ({chunk_sizes[0][1]:.1f} chars)")
        print(f"Largest chunks: {chunk_sizes[-1][0]} ({chunk_sizes[-1][1]:.1f} chars)")

        return results


def analyze_product_categories(csv_path):
    """Analyze product categories in the dataset"""
    df = pd.read_csv(csv_path, encoding='utf-8')

    print("=== Product Category Analysis ===")

    # Clean and analyze categories
    categories = df['category'].fillna('Unknown').str.strip()
    category_counts = categories.value_counts()

    print(f"Total products: {len(df)}")
    print(f"Unique categories: {len(category_counts)}")
    print("\nTop 10 categories:")
    for cat, count in category_counts.head(10).items():
        print(f"  {cat}: {count} products")

    # Analyze brands
    brands = df['brand'].fillna('Unknown').str.strip()
    brand_counts = brands.value_counts()

    print(f"\nUnique brands: {len(brand_counts)}")
    print("\nTop 10 brands:")
    for brand, count in brand_counts.head(10).items():
        print(f"  {brand}: {count} products")


def main():
    """Main demonstration function"""
    print("StarTech Products Basic Chunking Strategy Demonstration")
    print("=" * 60)

    # Analyze dataset first
    analyze_product_categories('startech_products.csv')

    print("\n" + "=" * 60)

    # Initialize chunker
    chunker = BasicChunker('startech_products.csv')

    # Run comparison with sample data
    results = chunker.compare_chunking_strategies(sample_size=100)

    print(f"\nDemonstration completed successfully!")
    print(f"Tested {len(results)} chunking strategies on 100 StarTech products")
    print("Check 'basic_chunking_results.json' for detailed results")


if __name__ == "__main__":
    main()