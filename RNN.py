import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Tokens import Tokenizer

device = torch.device("mps")

HIDDEN_DIM = 1000

# TODO we switched the tokenizer API so now we need to update this code. 
class RNNModel(nn.Module):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_tokens = tokenizer.num_unique_tokens
        self.initial_h = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.W_xh = nn.Parameter(torch.empty(HIDDEN_DIM, self.num_tokens))
        self.W_hh = nn.Parameter(torch.empty(HIDDEN_DIM, HIDDEN_DIM))
        self.W_hy = nn.Parameter(torch.empty(self.num_tokens, HIDDEN_DIM))
        self.B_h = nn.Parameter(torch.zeros(HIDDEN_DIM))
        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_hy)
        
    def forward(self, token_stream: list[int]):
        hidden_states = []
        outputs = []
        
        cur_state = self.initial_h.to(device)
        outputs.append(self.W_hy @ cur_state)
        
        # compute all of the hidden states so that we aren't modifying h in place
        for token in token_stream:
            x = F.one_hot(torch.tensor([token]), self.num_tokens).float().squeeze(0).to(device)
            x_state = self.W_xh @ x
            h_state = self.W_hh @ cur_state
            cur_state = torch.tanh(x_state + h_state + self.B_h)
            hidden_states.append(cur_state)
        
        for hidden_state in hidden_states:
            outputs.append(self.W_hy @ hidden_state)

        return torch.stack(outputs)
    
    def autoregress(self, len: int):
        # TODO implement temperature
        cur_state = self.initial_h.to(device)
        stream = [1]
        next_token = 1
        for i in range(len):
            # TODO package this into a function, since both this and forward use it
            x = F.one_hot(torch.tensor([next_token]), self.num_tokens).float().squeeze(0).to(device)
            state_1 = self.W_xh @ x
            state_2 = self.W_hh @ cur_state
            cur_state = torch.tanh(state_1 + state_2 + self.B_h).to(device)
            logits = self.W_hy @ cur_state
            _,next_token = torch.max(logits, dim = 0)
            stream.append(next_token)
        
        return stream

def train(trn_data_path: str, model: RNNModel, loss_fn, optimizer):
    # first, we want to get the training data from the file
    with open(trn_data_path, 'r', encoding='utf-8') as f:
        token_stream = model.tokenizer.str_to_stream(f.read())

    BATCH_LEN = 30
    
    parts = [token_stream[i:i + BATCH_LEN] for i in range(0, len(token_stream), BATCH_LEN)]
    
    model.train()
    for batch_num, batch in enumerate(parts):
        pred = model(batch)[:-1]
        expected = torch.tensor(batch).to(device)
        loss = loss_fn(pred, expected)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_num % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_num:>5d}/{len(parts):>5d}]")
            model.eval()
            print("sample output:  ", end="")
            with torch.no_grad():
                print(model.tokenizer.stream_to_str(model.autoregress(10)))
            model.train()
            

            

            