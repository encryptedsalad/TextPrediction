# I want to create an abstract base class that contains:
# 1: The number of possible tokens
# 2: A function that can turn a plaintext string into a list of tokens
# 3: A function that can turn a list of tokens into a plaintext string
from abc import ABC, abstractmethod
import json

class Tokenizer(ABC):
    num_unique_tokens: int
    
    @abstractmethod
    def str_to_stream(self, input: str) -> list[int]:
        pass
    
    @abstractmethod
    def stream_to_str(self, tokens: list[int]) -> str:
        pass

class ASCIITokenizer(Tokenizer):
    num_unique_tokens = 256
    
    def str_to_stream(self, input: str) -> list[int]:
        out = []
        for char in input:
            out.append(ord(char))
        return out
    
    def stream_to_str(self, input: list[int]) -> str:
        out = ""
        for token in input:
            out.append(chr(token))
        return out
    
class AlphabetTokenizer(Tokenizer):
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz:.?!()\n\t '
    num_unique_tokens = len(alphabet) + 1
    
    def str_to_stream(self, input: str) -> list[int]:
        out = []
        for char in input:
            out.append(self.alphabet.find(input[0]) + 1)
        return out
    
    def stream_to_str(self, input: list[int]) -> str:
        out = ""
        for token in input:
            if token <= 0 or token > len(self.alphabet):
            # arbitrarily decided that vertical bar is the standard ascii null char
                out.append("|")
            out.append(self.alphabet[token - 1])
        return out

def find_most_freq_words(data: str = ""):
    data = open("data/all_shakespeare.txt", "rb").read()
    words_list = data.split()
    words_dict = {}

    for word in words_list:
        word = word.decode("utf-8", errors="ignore").lower()
        if words_dict.get(word) == None:
            words_dict[word] = 1
        else:
            words_dict[word] = words_dict[word] + 1
        
    unique_words = list(words_dict.keys())

    unique_words.sort(reverse = True, key = lambda x: words_dict[x])
    return unique_words

class WordTokenizer(Tokenizer):
    words : list[int]
    num_unique_tokens : int
    
    def __init__(self, top_n: int = 10000, words: list[str] = None):
        # if we are given words, then we just use them
        if words == None:
            words = find_most_freq_words()
        self.words = words[:top_n]
        self.num_unique_tokens = len(self.words) + 1
        
    def str_to_stream(self, input: str) -> list[int]:
        return list(map(lambda x: self.words.index(x) + 1 if x in self.words else 0, input.split()))
    
    def stream_to_str(self, input: list[int]) -> str:
        return " ".join(list(map(lambda x: self.words[x - 1] if x != 0 else "FILL", input)))
    