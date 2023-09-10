import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW 

# main parameters
block_size = 64
batch_size = 16 # blocks to process in parallel
max_iters = 150
d_model = 256  
nhead = 4      # self-attention heads
num_encoder_layers = 4  
lr = 1e-3
eval_iters = 140
eval_interval = 10
#______
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# a list containing each unique character in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# using dictionaries to create a map for characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # takes a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # takes a list of integers, output a string

# encode the entire text
data = torch.tensor(encode(text), dtype=torch.long)
# training 90% of the data and leaving the rest 10% to validate the data later on
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# using the GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print()

train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

# loading the data by small batches of inputs x and targets y
def get_batch(split):
    if split == 'train': data = train_data 
    else: data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y =x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X, Y)  # Forward pass to get logits
            loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))  # Calculate the loss
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output

xb, yb = get_batch('train')


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        logits = self.fc(output)
        return logits  
    def generate(self, starting_token, max_new_tokens):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            current_token = torch.tensor([[starting_token]], dtype=torch.long).to(device) # Note the extra dimension
            generated_tokens = [starting_token]
            for _ in range(max_new_tokens):
                logits = self(current_token, current_token)  # Use self-attention for autoregressive generation
                probs = F.softmax(logits[0, -1], dim=-1)  # Take the last token's probabilities
                next_token = torch.multinomial(probs, num_samples=1).item()
                next_token = int(next_token)  # Convert to Python integer

                generated_tokens.append(next_token)
                current_token = torch.cat([current_token, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        self.train()  # Set the model back to training mode
        return generated_tokens



model = TransformerLanguageModel(vocab_size, d_model, nhead, num_encoder_layers)
model.to(device)

starting_token = stoi['h']  # Use the index for the starting character
generated_tokens = model.generate(starting_token, max_new_tokens=300)
generated_text = decode(generated_tokens)
# print(generated_text)




# now that we got a result that is random, let's optimize it
# in this case, we will use the AdamW optimizer
def evaluate(model, split='val'):
    model.eval()
    loss = 0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x, y)
            loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return loss / eval_iters

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
best_val_loss = float('inf')

for iter in range(max_iters):
    xb, yb = get_batch('train')
    optimizer.zero_grad()
    logits = model(xb, yb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        val_loss = evaluate(model)
        print(f"Iteration {iter}, Validation loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
print("----------")
print("Final loss: ", loss.item())
starting_token = stoi['T']
print(decode(model.generate(starting_token, max_new_tokens=125)))
