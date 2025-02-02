import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda")
alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz:.?!()\n\t '

def to_alphabet(c: str):
    return alphabet.find(c[0]) + 1

def from_alphabet(c: int):
    if c <= 0 or c > len(alphabet):
        # arbitrarily decided that vertical bar is the standard ascii null char
        return "|"
    return alphabet[c - 1]

# TODO use the alphabet 
# TODO fix the off by one error, or at least debug to determine if it is actually happening

# use the full ascii character set, option to change it later.
NUM_CHARS = len(alphabet) + 1
HIDDEN_DIM = 1000

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_h = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.W_xh = nn.Parameter(torch.empty(HIDDEN_DIM, NUM_CHARS))
        self.W_hh = nn.Parameter(torch.empty(HIDDEN_DIM, HIDDEN_DIM))
        self.W_hy = nn.Parameter(torch.empty(NUM_CHARS, HIDDEN_DIM))
        self.B_h = nn.Parameter(torch.zeros(HIDDEN_DIM))
        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_hy)
        
    # TODO add in a partial hidden state vector
    def forward(self, x_seq: str):
        hidden_states = []
        outputs = []
        
        cur_state = self.initial_h.to(device)
        outputs.append(self.W_hy @ cur_state)
        # pre-compute all of the hidden states so that we aren't modifying h in place
        for letter in x_seq:
            # TODO technically, this predicts the second token on the first run.
            x = F.one_hot(torch.tensor([to_alphabet(letter)]), NUM_CHARS).float().squeeze(0).to(device)
            state_1 = self.W_xh @ x
            state_2 = self.W_hh @ cur_state
            hidden_states.append(torch.tanh(state_1 + state_2 + self.B_h))
            cur_state = hidden_states[-1].to(device)
        
        for hidden_state in hidden_states:
            outputs.append(self.W_hy @ hidden_state)

        return torch.stack(outputs)
    
    def autoregress(self, len: int, temp: float):
        # TODO implement temperature
        cur_state = self.initial_h.to(device)
        next_letter = "a"
        for i in range(len):
            # TODO package this into a function, maybe?
            x = F.one_hot(torch.tensor([to_alphabet(next_letter)]), NUM_CHARS).float().squeeze(0).to(device)
            state_1 = self.W_xh @ x
            state_2 = self.W_hh @ cur_state
            cur_state = torch.tanh(state_1 + state_2 + self.B_h).to(device)
            logits = self.W_hy @ cur_state
            _,index = torch.max(logits, dim = 0)
            next_letter = from_alphabet(index)
            print(next_letter, end="")

    def get_expected(x_seq: str):
        expecteds = []
        for letter in x_seq:
            expecteds.append(to_alphabet(letter))
        return torch.tensor(expecteds)

def train(trn_data_path: str, model: RNNModel, loss_fn, optimizer):
    # first, we want to get the training data from the file
    with open(trn_data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # TODO this might need some cleaning up to make it start on a paragraph
    BATCH_LEN = 200
    parts = [text[i:i + BATCH_LEN] for i in range(0, len(text), BATCH_LEN)]
    
    model.train()
    for batch, part in enumerate(parts):
        pred = model(part)[:-1]
        expected = RNNModel.get_expected(part).to(device)
        loss = loss_fn(pred, expected)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{len(parts):>5d}]")
            model.eval()
            print("sample output:  ", end="")
            with torch.no_grad():
                model.autoregress(100)
            model.train()
            print()
            

            