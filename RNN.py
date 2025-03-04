import torch
from torch import nn
import torch.nn.functional as F
from Tokens import Tokenizer
from abc import ABC, abstractmethod
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class RNN(ABC, nn.Module):
    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer):
        nn.Module.__init__(self)
        self.tokenizer = tokenizer
        self.num_tokens = tokenizer.num_unique_tokens
        
    @abstractmethod
    def first_state(self) -> torch.Tensor:
        """
        Gets the initial state assuming no input.
        """
        pass
    
    @abstractmethod
    def next_state(self, cur_token: torch.Tensor) -> torch.Tensor:
        """
        Takes a single input state and produces a single output state. States are arbitrary 1d tensors. 
        Implementer is responsible for keeping track of hidden states.
        """
        pass

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of input states and produces a tensor of output states.
        The input tensor is assumed to be a 2d tensor gotten from torch.stack() of a list of 1d state tensors. 
        The output tensor is a 2d tensor of feeding the batch of input tensors into next_token, and then stacking them.
        """
        out_states = [self.first_state()]
        tokens = batch.unbind()
        for token in tokens:
            out_states.append(self.next_state(token))
        return torch.stack(out_states)[:-1]
    
    def get_token_from_state(self, state: torch.Tensor, temperature: float) -> int:
        probs = F.softmax(state / temperature, dim = 0)
        return torch.multinomial(probs, 1)

    def autoregress(self, len: int, start: torch.Tensor = None) -> list[int]:
        if start != None:
            tokens = start.unbind()
            for token in tokens:
                cur_state = self.next_state(token)

        out = []
        cur_state = self.first_state()
        for _ in range(len):
            out.append(self.get_token_from_state(cur_state, 0.3))
            cur_state = self.next_state(F.one_hot(torch.tensor([out[-1]]), self.num_tokens).float().squeeze(0).to(device))
        return out
            
    def train_with_data(self, 
            trn_data_path: str, 
            batch_size: int, 
            loss_fn = nn.CrossEntropyLoss(), 
            optimizer = None):

        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters())

        with open(trn_data_path, 'r', encoding='utf-8') as f:
            token_stream = self.tokenizer.str_to_stream(f.read())
        
        parts = [token_stream[i:i + batch_size] for i in range(0, len(token_stream), batch_size)]
        
        self.train()
        losses = []
        for batch_num, batch in enumerate(parts):
            states = self.tokenizer.stream_to_states(batch)
            pred = self(states)
            expected = torch.tensor(batch).to(device)
            loss = loss_fn(pred, expected)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_num % 100 == 0:
                loss = loss.item()
                losses.append(loss)
                recent_loss = sum(losses[-10:]) / len(losses[-10:])
                print(f"loss  : {loss:>7f}  [{batch_num:>5d}/{len(parts):>5d}]")
                print(f"recent: {recent_loss:>7f}  [{batch_num:>5d}/{len(parts):>5d}]")
                self.eval()
                print("sample output:  ", end="")
                with torch.no_grad():
                    print(self.tokenizer.stream_to_str(self.autoregress(100)))
                self.train()      