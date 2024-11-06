import argparse
import random
from collections import Counter
from models import NgramModel, NgramModelSmoothing, InterpolatedModel

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str, default='Ngrams',
                        choices=['Ngrams', 'Interpolation'], help="model selection [Ngrams, Interpolation]")
    parser.add_argument('--feature', '-f', type=str, default='woSmoothing',
                        choices=['woSmoothing', 'wSmoothing'], help="ngrams with or without smoothing")
    parser.add_argument('--path', type=str, default='./A2-Data/', help="path to dataset")
    args = parser.parse_args()

    # load our data provided its the given .tokens from the assignment
    try:
        if args.path:
            provided_train_data = get_sentences(args.path + "1b_benchmark.train.tokens")
            provided_test_data = get_sentences(args.path + "1b_benchmark.test.tokens")
        
            # build our vocab
            train_vocab, train_vocab_size, train_token_count = get_vocab(provided_train_data)
            test_vocab, test_vocab_size, test_token_count = get_vocab(provided_test_data)

            # replace tokens in train and test with <UNK> where required
            provided_train_data = replace_tokens(provided_train_data, train_vocab)
            provided_test_data = replace_tokens(provided_test_data, test_vocab)
    except Exception as error:
        print(f"Error from args.path | {error}")
    

    if args.feature == 'wSmoothing':
        unigram_model = NgramModelSmoothing(1)
        bigram_model = NgramModelSmoothing(2)
        trigram_model = NgramModelSmoothing(3)
    elif args.feature == 'woSmoothing':
        unigram_model = NgramModel(1)
        bigram_model = NgramModel(2)
        trigram_model = NgramModel(3)


    unigram_model.set_vocab_size(train_vocab_size)
    bigram_model.set_vocab_size(train_vocab_size)
    trigram_model.set_vocab_size(train_vocab_size)

    unigram_model.train(provided_train_data)
    bigram_model.train(provided_train_data)
    trigram_model.train(provided_train_data)

    for model_name, model in [("Unigram", unigram_model), ("Bigram", bigram_model), ("Trigram", trigram_model)]:
        print(f"{model_name} Model Perplexities:")
        print("Training Set:", model.perplexity(provided_train_data))
        print("Test Set:", model.perplexity(provided_test_data))
        print()

    test_input = [["HDTV", "."]]
    print("Unigram Model Perplexity on 'HDTV .':", unigram_model.perplexity(test_input))
    print("Bigram Model Perplexity on 'HDTV .':", bigram_model.perplexity(test_input))
    print("Trigram Model Perplexity on 'HDTV .':", trigram_model.perplexity(test_input))

    if args.model == 'Interpolation':
        lambda1 = 0.1
        lambda2 = 0.3
        lambda3 = 0.6
        # Initialize Interpolated Model with generated lambdas
        interpolated_model = InterpolatedModel(unigram_model, bigram_model, trigram_model, lambda1, lambda2, lambda3)
        
        # calc perplexities
        train_perplexity = interpolated_model.perplexity(provided_train_data)
        test_perplexity = interpolated_model.perplexity(provided_test_data)
        
        # Print results
        print(f"\nLambda set: λ1={lambda1:.2f}, λ2={lambda2:.2f}, λ3={lambda3:.2f}")
        print(f"Training Set Perplexity: {train_perplexity}")
        print(f"Test Set Perplexity: {test_perplexity}\n")


        

    print(args)


if __name__ == "__main__":
    main()