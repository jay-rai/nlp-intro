import math
from collections import defaultdict

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab_size = 0 

    def train(self, sentences):
        for sentence in sentences:
            tokens = ['<START>'] * (self.n - 1) + sentence
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size 

    def probability(self, ngram):
        context = ngram[:-1]
        if self.context_counts[context] == 0:
            return 1 / self.vocab_size
        return self.ngram_counts[ngram] / self.context_counts[context]

    def perplexity(self, sentences):
        log_prob_sum = 0
        total_tokens = 0
        for sentence in sentences:
            tokens = ['<START>'] * (self.n - 1) + sentence + ['<STOP>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                prob = self.probability(ngram)
                log_prob_sum += math.log(prob if prob > 0 else 1 / self.vocab_size)
            total_tokens += len(sentence) + 1
        return math.exp(-log_prob_sum / total_tokens)
    
class NgramModelSmoothing:
    def __init__(self, n, alpha=1.0):
        self.n = n
        self.alpha = alpha
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab_size = 0

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def train(self, sentences):
        for sentence in sentences:
            tokens = ['<START>'] * (self.n - 1) + sentence
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def probability(self, ngram):
        context = ngram[:-1]
        smoothed_count = self.ngram_counts[ngram] + self.alpha
        smoothed_context_count = self.context_counts[context] + (self.alpha * self.vocab_size)
        return smoothed_count / smoothed_context_count

    def perplexity(self, sentences):
        log_prob_sum = 0
        total_tokens = 0
        for sentence in sentences:
            tokens = ['<START>'] * (self.n - 1) + sentence + ['<STOP>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                prob = self.probability(ngram)
                log_prob_sum += math.log(prob if prob > 0 else 1 / self.vocab_size)
            total_tokens += len(sentence) + 1
        return math.exp(-log_prob_sum / total_tokens)
    

class InterpolatedModel:
    def __init__(self, unigram_model, bigram_model, trigram_model, lambda1=0.33, lambda2=0.33, lambda3=0.34):
        self.unigram_model = unigram_model
        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def probability(self, ngram):
        unigram_prob = self.unigram_model.probability((ngram[-1],))
        bigram_prob = self.bigram_model.probability(ngram[-2:]) if len(ngram) > 1 else 0
        trigram_prob = self.trigram_model.probability(ngram) if len(ngram) > 2 else 0

        interpolated_prob = (self.lambda1 * unigram_prob +
                             self.lambda2 * bigram_prob +
                             self.lambda3 * trigram_prob)
        return interpolated_prob

    def perplexity(self, sentences, epsilon=1e-10):
        log_prob_sum = 0
        total_tokens = 0
        for sentence in sentences:
            tokens = ['<START>', '<START>'] + sentence + ['<STOP>']
            for i in range(2, len(tokens)):
                ngram = tuple(tokens[max(0, i-2):i+1])
                prob = self.probability(ngram)
                
                log_prob_sum += math.log(prob + epsilon)
            total_tokens += len(sentence) + 1
        return math.exp(-log_prob_sum / total_tokens)