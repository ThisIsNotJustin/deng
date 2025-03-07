import pandas as pd
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = pd.read_csv('eng-dutch.tsv', sep='\t', header=None, names=['eng_id', 'eng', 'd-id', 'dutch'])

pairs = []
for _, row in data.iterrows():
    pairs.append({'src': f'<2nl> {row.eng}', 'trg': f'<2en> {row.dutch}'})
    pairs.append({'src': f'<2en> {row.dutch}', 'trg': f'<2nl> {row.eng}'})

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=['<pad>', '<sos>', '<eos>', '<unk>', '<2en>', '<2nl>']
)

corpus = [pair['src'] for pair in pairs] + [pair['trg'] for pair in pairs]
tokenizer.train_from_iterator(corpus, trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=100):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src = self.tokenizer.encode(pair['src']).ids[:self.max_length]
        trg = [self.tokenizer.token_to_id('<sos>')] + self.tokenizer.encode(pair['trg']).ids[:self.max_length-2] + [self.tokenizer.token_to_id('<eos>')]
        return torch.tensor(src), torch.tensor(trg)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=tokenizer.token_to_id('<pad>'), batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=tokenizer.token_to_id('<pad>'), batch_first=True)
    return src_padded, trg_padded

dataset = TranslationDataset(pairs, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, trg, src_padding_mask=None, trg_padding_mask=None, trg_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        trg_emb = self.embedding(trg) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        trg_emb = self.pos_encoder(trg_emb)
        
        output = self.transformer(
            src_emb, trg_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask
        )
        return self.fc_out(output)

vocab_size = tokenizer.get_vocab_size()
model = Transformer(vocab_size).to(device)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))

model.train()
for epoch in range(10):
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        src_padding_mask = (src == tokenizer.token_to_id('<pad>')).to(device)
        trg_padding_mask = (trg_input == tokenizer.token_to_id('<pad>')).to(device)
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(1)).to(device)

        optimizer.zero_grad()
        output = model(src, trg_input, src_padding_mask, trg_padding_mask, trg_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), trg_output.reshape(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

def translate(model, tokenizer, src_sentence, direction='en-nl', max_length=50):
    model.eval()
    src_text = f'<2nl> {src_sentence}' if direction == 'en-nl' else f'<2en> {src_sentence}'
    src_ids = tokenizer.encode(src_text).ids
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    trg_ids = [tokenizer.token_to_id('<sos>')]
    for _ in range(max_length):
        trg = torch.tensor(trg_ids).unsqueeze(0).to(device)
        output = model(src, trg)
        next_id = output.argmax(2)[:, -1].item()
        trg_ids.append(next_id)
        if next_id == tokenizer.token_to_id('<eos>'):
            break
    return tokenizer.decode(trg_ids, skip_special_tokens=True)

print(translate(model, tokenizer, 'Hello world!', 'eng-nl'))