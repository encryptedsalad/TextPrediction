import torch
from torch import nn
import torch.nn.functional as F
from Tokens import Tokenizer
from RNN import RNN
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class SimpleRNN(RNN):
    def __init__(self, tokenizer: Tokenizer, hidden_dim: int):
        super().__init__(tokenizer)
        self.initial_h = nn.Parameter(torch.zeros(hidden_dim))
        self.h = [self.initial_h]
        self.W_xh = nn.Parameter(torch.empty(hidden_dim, self.num_tokens))
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hy = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
        self.B_h = nn.Parameter(torch.zeros(hidden_dim))
        self.h = [self.initial_h]
        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_hy)
        
    def first_state(self) -> torch.Tensor:
        return self.initial_h @ self.W_hy
    
    def next_state(self, cur_token: torch.Tensor) -> torch.Tensor:
        x_state = self.W_xh @ cur_token
        h_state = self.W_hh @ self.h[-1]
        self.h.append(torch.tanh(x_state + h_state + self.B_h))
        return self.h[-1] @ self.W_hy
