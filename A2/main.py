import argparse
import random
from collections import Counter
from models import NgramModel, NgramModelSmoothing, InterpolatedModel
import re

def get_vocab(sentences, min_count=3):
    token_count = Counter(word for sentence in sentences for word in sentence)
    vocab = {word for word, count in token_count.items() if count >= min_count}
    vocab.add('<UNK>')
    vocab.add('<STOP>')
    vocab_size = len(vocab)
    return vocab, vocab_size, token_count

def replace_tokens(sentences, vocab):
    return [[word if word in vocab else '<UNK>' for word in sentence] + ['<STOP>'] for sentence in sentences]

def get_sentences(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            tokens = line.strip().split() 
            sentences.append(tokens)
    return sentences

def get_every_other_sentence(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as infile:
        # Read the entire file content
        content = infile.read()
        # Split content into sentences using regex to match sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', content)
        # Take every other sentence (starting from index 0)
        every_other_sentence = [s.split() for s in sentences[::2]]  # Tokenize every other sentence
    return every_other_sentence

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str, default='Ngrams',
                        choices=['Ngrams', 'Interpolation'], help="model selection [Ngrams, Interpolation]")
    parser.add_argument('--feature', '-f', type=str, default='woSmoothing',
                        choices=['woSmoothing', 'wSmoothing', 'halfData'], help="ngrams with or without smoothing | or get half data")
    parser.add_argument('--path', type=str, default='./A2-Data/', help="path to dataset")
    parser.add_argument('--lambda1', type=float, help="Weight for unigram model in interpolation")
    parser.add_argument('--lambda2', type=float, help="Weight for bigram model in interpolation")
    parser.add_argument('--lambda3', type=float, help="Weight for trigram model in interpolation")
    args = parser.parse_args()
    
    if all([args.lambda1 is not None, args.lambda2 is not None, args.lambda3 is not None]):
        if args.lambda1 + args.lambda2 + args.lambda3 != 1:
            raise ValueError("Lambda values must sum to 1")
        custom_lambdas = (args.lambda1, args.lambda2, args.lambda3)
    else:
        custom_lambdas = None
    
    # Load data based on the feature flag
    try:
        if args.feature == 'halfData':
            # Load only every other sentence for halfData feature
            train_data = get_every_other_sentence(args.path + "1b_benchmark.train.tokens")
            dev_data = get_every_other_sentence(args.path + "1b_benchmark.dev.tokens")
            test_data = get_every_other_sentence(args.path + "1b_benchmark.test.tokens")
        else:
            # Load all sentences
            train_data = get_sentences(args.path + "1b_benchmark.train.tokens")
            dev_data = get_sentences(args.path + "1b_benchmark.dev.tokens")
            test_data = get_sentences(args.path + "1b_benchmark.test.tokens")
        
        # Build vocab from train set and replace tokens in all sets
        vocab, vocab_size, _ = get_vocab(train_data)
        train_data = replace_tokens(train_data, vocab)
        dev_data = replace_tokens(dev_data, vocab)
        test_data = replace_tokens(test_data, vocab)
    except Exception as error:
        print(f"Error loading data: {error}")
    
    # Initialize models based on smoothing feature
    if args.feature == 'wSmoothing':
        unigram_model = NgramModelSmoothing(1)
        bigram_model = NgramModelSmoothing(2)
        trigram_model = NgramModelSmoothing(3)
    else:
        unigram_model = NgramModel(1)
        bigram_model = NgramModel(2)
        trigram_model = NgramModel(3)

    # Set vocab size for each model
    unigram_model.set_vocab_size(vocab_size)
    bigram_model.set_vocab_size(vocab_size)
    trigram_model.set_vocab_size(vocab_size)

    # Train each model
    unigram_model.train(train_data)
    bigram_model.train(train_data)
    trigram_model.train(train_data)

    # Evaluate models
    for model_name, model in [("Unigram", unigram_model), ("Bigram", bigram_model), ("Trigram", trigram_model)]:
        print(f"{model_name} Model Perplexities:")
        print("Training Set:", model.perplexity(train_data))
        print("Development Set:", model.perplexity(dev_data))
        print("Test Set:", model.perplexity(test_data))
        print()

    test_input = [["HDTV", "."]]
    print("Unigram Model Perplexity on 'HDTV .':", unigram_model.perplexity(test_input))
    print("Bigram Model Perplexity on 'HDTV .':", bigram_model.perplexity(test_input))
    print("Trigram Model Perplexity on 'HDTV .':", trigram_model.perplexity(test_input))
    print()

    # Interpolation Model
    if args.model == 'Interpolation':
        best_dev_perplexity = float('inf')
        best_lambdas = custom_lambdas if custom_lambdas else (args.lambda1, args.lambda2, args.lambda3)
        
        # Iterate over lambda values if custom lambdas not provided
        if not custom_lambdas:
            for lambda1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for lambda2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    lambda3 = 1 - lambda1 - lambda2
                    if lambda3 < 0:
                        continue
                    
                    interpolated_model = InterpolatedModel(unigram_model, bigram_model, trigram_model, lambda1, lambda2, lambda3)
                    
                    # Calculate perplexity on dev set
                    dev_perplexity = interpolated_model.perplexity(dev_data)
                    print(f"λ1={lambda1}, λ2={lambda2}, λ3={lambda3} - Dev Perplexity: {dev_perplexity}")
                    
                    # Track best dev perplexity and lambda values
                    if dev_perplexity < best_dev_perplexity:
                        best_dev_perplexity = dev_perplexity
                        best_lambdas = (lambda1, lambda2, lambda3)
        else:
            print(f"Using user-provided lambdas: λ1={best_lambdas[0]}, λ2={best_lambdas[1]}, λ3={best_lambdas[2]}")
        
        # Evaluate final interpolated model on all sets
        best_lambda1, best_lambda2, best_lambda3 = best_lambdas
        final_interpolated_model = InterpolatedModel(unigram_model, bigram_model, trigram_model, best_lambda1, best_lambda2, best_lambda3)
        
        train_perplexity = final_interpolated_model.perplexity(train_data)
        dev_perplexity = final_interpolated_model.perplexity(dev_data)
        test_perplexity = final_interpolated_model.perplexity(test_data)
        
        print(f"\nFinal Interpolated Model with Optimal Lambdas: λ1={best_lambda1}, λ2={best_lambda2}, λ3={best_lambda3}")
        print(f"Training Set Perplexity: {train_perplexity}")
        print(f"Development Set Perplexity: {dev_perplexity}")
        print(f"Test Set Perplexity: {test_perplexity}\n")

if __name__ == "__main__":
    main()
