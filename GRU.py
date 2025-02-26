import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Tokens import Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self, tokenizer: Tokenizer, hidden_dim: int):
        # h is h in the wikipedia article, k is h hat in the wikipedia article.
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.num_tokens = tokenizer.num_unique_tokens
        self.initial_h = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_z = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_r = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_k = nn.Parameter(torch.zeros(hidden_dim))
        self.W_xz = nn.Parameter(torch.empty(hidden_dim, self.num_tokens))
        self.W_xr = nn.Parameter(torch.empty(hidden_dim, self.num_tokens))
        self.W_xk = nn.Parameter(torch.empty(hidden_dim, self.num_tokens))
        self.W_hz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hk = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hy = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
        self.B_k = nn.Parameter(torch.zeros(hidden_dim))
        self.B_z = nn.Parameter(torch.zeros(hidden_dim))
        self.B_r = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_normal_(self.W_xk)
        nn.init.xavier_normal_(self.W_xz)
        nn.init.xavier_normal_(self.W_xr)
        nn.init.xavier_normal_(self.W_hk)
        nn.init.xavier_normal_(self.W_hz)
        nn.init.xavier_normal_(self.W_hr)
        nn.init.xavier_normal_(self.W_hy)

    def package_batch(self, batch: list[int]) -> torch.Tensor:
        out_list = []
        for token in batch:
            out_list.append(F.one_hot(torch.tensor([token]), self.num_tokens).float().squeeze(0).to(device))
        return torch.stack(out_list)

    def forward(self, input: torch.Tensor):
        h = self.initial_h.to(device)
        z = self.initial_z.to(device)
        r = self.initial_r.to(device)
        k = self.initial_k.to(device)
        
        hidden_h = [h]
        hidden_z = [z]
        hidden_r = [r]
        hidden_k = [k]
        
        outputs = [self.W_hy @ h]
        
        # keep list of hidden states so we aren't modifying h in place
        tokens = input.unbind()
        for x in tokens:
            z = torch.sigmoid(self.W_xz @ x + self.W_hz @ h + self.B_z)
            r = torch.sigmoid(self.W_xr @ x + self.W_hr @ h + self.B_r)
            k = torch.tanh(self.W_xk @ x + self.W_hk @ (r * h) + self.B_k)
            h = ((torch.ones(self.hidden_dim).to(device) - z) * h) + (z * k) 
            
            hidden_z.append(z)
            hidden_r.append(r)
            hidden_k.append(k)
            hidden_h.append(h)
            
            outputs.append(self.W_hy @ h)

        return torch.stack(outputs[:-1])
    
    def autoregress(self, len: int, temperature = 1.0) -> list[int]:
        h = self.initial_h.to(device)
        z = self.initial_z.to(device)
        r = self.initial_r.to(device)
        k = self.initial_k.to(device)

        next_token = self.tokenizer.str_to_stream("the quick brown fox")[0]
        stream = [next_token]

        # keep list of hidden states so we aren't modifying h in place
        for _ in range(len):
            x = F.one_hot(torch.tensor([next_token]), self.num_tokens).float().squeeze(0).to(device)
            z = torch.sigmoid(self.W_xz @ x + self.W_hz @ h + self.B_z)
            r = torch.sigmoid(self.W_xr @ x + self.W_hr @ h + self.B_r)
            k = torch.tanh(self.W_xk @ x + self.W_hk @ (r * h) + self.B_k)
            h = ((torch.ones(self.hidden_dim).to(device) - z) * h) + (z * k)
            
            logits = self.W_hy @ h
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