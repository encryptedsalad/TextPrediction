import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Tokens import Tokenizer
from abc import ABC, abstractmethod
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class RNN(ABC):
    hidden_states : list[torch.Tensor]
    tokenizer: Tokenizer

    @abstractmethod
    def next_token(cur_token: torch.Tensor) -> torch.Tensor:
        pass


class RNNModel(nn.Module):
    def __init__(self, tokenizer: Tokenizer, hidden_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_tokens = tokenizer.num_unique_tokens
        self.initial_h = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xh = nn.Parameter(torch.empty(hidden_dim, self.num_tokens))
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hy = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
        self.B_h = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_hy)
        
    # TODO make this use the vectors that the GRU does
    def forward(self, token_stream: list[int]):
        hidden_states = []
        outputs = []
        
        cur_state = self.initial_h.to(device)
        outputs.append(self.W_hy @ cur_state)
        
        # keep list of hidden states so we aren't modifying h in place
        for token in token_stream:
            x = F.one_hot(torch.tensor([token]), self.num_tokens).float().squeeze(0).to(device)
            x_state = self.W_xh @ x
            h_state = self.W_hh @ cur_state
            cur_state = torch.tanh(x_state + h_state + self.B_h)
            hidden_states.append(cur_state)
        
        for hidden_state in hidden_states:
            outputs.append(self.W_hy @ hidden_state)

        return torch.stack(outputs)
    
    def autoregress(self, len: int, temperature = 1.0):
        cur_state = self.initial_h.to(device)
        next_token = self.tokenizer.str_to_stream("the quick brown fox")[0]
        stream = [next_token]
        for _ in range(len):
            x = F.one_hot(torch.tensor([next_token]), self.num_tokens).float().squeeze(0).to(device)
            state_1 = self.W_xh @ x
            state_2 = self.W_hh @ cur_state
            cur_state = torch.tanh(state_1 + state_2 + self.B_h).to(device)
            logits = self.W_hy @ cur_state
            probs = F.softmax(logits / temperature, dim = 0)
            next_token = torch.multinomial(probs, 1)
            stream.append(next_token)
        
        return stream

    def train_with_data(self, 
            trn_data_path: str, 
            batch_size: int, 
            loss_fn = nn.CrossEntropyLoss, 
            optimizer = None):

        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters())

        with open(trn_data_path, 'r', encoding='utf-8') as f:
            token_stream = self.tokenizer.str_to_stream(f.read())
        
        parts = [token_stream[i:i + batch_size] for i in range(0, len(token_stream), batch_size)]
        
        self.train()
        for batch_num, batch in enumerate(parts):
            pred = self(batch)[:-1]
            expected = torch.tensor(batch).to(device)
            loss = loss_fn(pred, expected)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_num % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}  [{batch_num:>5d}/{len(parts):>5d}]")
                self.eval()
                print("sample output:  ", end="")
                with torch.no_grad():
                    print(self.tokenizer.stream_to_str(self.autoregress(100, 0.3)))
                self.train()