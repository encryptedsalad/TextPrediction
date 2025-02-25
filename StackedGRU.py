import torch
from torch import nn
import torch.nn.functional as F
from Tokens import Tokenizer
from GRU import GRUModel
from torch import gru

device = torch.device("cuda")

class StackedGRU(nn.Module):
    def __init__(self, tokenizer: Tokenizer, hidden_dim: int):
        # h is h in the wikipedia article, k is h hat in the wikipedia article.
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.num_tokens = tokenizer.num_unique_tokens
        self.l1 = GRUModel(tokenizer, hidden_dim)
        self.l2 = GRUModel(tokenizer, hidden_dim)

    def package_batch(self, batch: list[int]) -> torch.Tensor:
        out_list = []
        for token in batch:
            out_list.append(F.one_hot(torch.tensor([token]), self.num_tokens).float().squeeze(0).to(device))
        return torch.stack(out_list)

    def forward(self, input: torch.Tensor):
        y1 = self.l1(input)
        return self.l2(y1)
    
    # TODO refactor this to take a single token at a time so that we can stack it natively and save lines of code
    def autoregress(self, len: int, temperature = 1.0) -> list[int]:
        h1 = self.l1.initial_h.to(device)
        z1 = self.l1.initial_z.to(device)
        r1 = self.l1.initial_r.to(device)
        k1 = self.l1.initial_k.to(device)
        h2 = self.l2.initial_h.to(device)
        z2 = self.l2.initial_z.to(device)
        r2 = self.l2.initial_r.to(device)
        k2 = self.l2.initial_k.to(device)

        next_token = self.tokenizer.str_to_stream("the quick brown fox")[0]
        stream = [next_token]

        # keep list of hidden states so we aren't modifying h in place
        for _ in range(len):
            x = F.one_hot(torch.tensor([next_token]), self.num_tokens).float().squeeze(0).to(device)
            z1 = torch.sigmoid(self.l1.W_xz @ x + self.l1.W_hz @ h1 + self.l1.B_z)
            r1 = torch.sigmoid(self.l1.W_xr @ x + self.l1.W_hr @ h1 + self.l1.B_r)
            k1 = torch.tanh(self.l1.W_xk @ x + self.l1.W_hk @ (r1 * h1) + self.l1.B_k)
            h1 = ((torch.ones(self.hidden_dim).to(device) - z1) * h1) + (z1 * k1)
            y1 = self.l1.W_hy @ h1

            z2 = torch.sigmoid(self.l2.W_xz @ y1 + self.l2.W_hz @ h2 + self.l2.B_z)
            r2 = torch.sigmoid(self.l2.W_xr @ y1 + self.l2.W_hr @ h2 + self.l2.B_r)
            k2 = torch.tanh(self.l2.W_xk @ x + self.l2.W_hk @ (r2 * h2) + self.l2.B_k)
            h2 = ((torch.ones(self.hidden_dim).to(device) - z2) * h2) + (z2 * k2)

            logits = self.l2.W_hy @ h2
            probs = F.softmax(logits / temperature, dim = 0)
            next_token = torch.multinomial(probs, 1)
            stream.append(next_token)
            
        return stream

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
        for batch_num, batch in enumerate(parts):
            pred = self(self.package_batch(batch))
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