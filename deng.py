import pandas as pd
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer uninitialized.")
    
    return tokenizer

def init_tokenizer(data_file):
    global tokenizer

    data = pd.read_csv(data_file, sep='\t', header=None, names=['eng_id', 'eng', 'd-id', 'dutch'])

    pairs = []
    for _, row in data.iterrows():
        pairs.append({'src': f'<2nl> {row.eng}', 'trg': f'<2en> {row.dutch}'})
        pairs.append({'src': f'<2en> {row.dutch}', 'trg': f'<2nl> {row.eng}'})

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        special_tokens=['<pad>', '<sos>', '<eos>', '<unk>', '<2en>', '<2nl>']
    )

    corpus = [pair['src'] for pair in pairs] + [pair['trg'] for pair in pairs]
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return pairs

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
        trg = [self.tokenizer.token_to_id('<sos>')] + \
        self.tokenizer.encode(pair['trg']).ids[:self.max_length-2] + \
        [self.tokenizer.token_to_id('<eos>')]
        return torch.tensor(src), torch.tensor(trg)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=tokenizer.token_to_id('<pad>'), batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=tokenizer.token_to_id('<pad>'), batch_first=True)
    return src_padded, trg_padded

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
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
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
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_dir='checkpoint'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, val_loss)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, val_loss)
            self.counter = 0
    
    def save_checkpoint(self, model, optimizer, epoch, loss):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_checkpoint_{timestamp}_loss_{loss:.4f}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,

        }, filepath)
        print(f"Saved checkpoint to {filepath}")

def train_model(model, dataloader, num_epochs=10, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0

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

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f'Epoch {epoch}, Loss: {loss.item()}, Avg Loss: {avg_loss}')

        early_stopping(avg_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def load_best_model(model, checkpoint_dir='checkpoint'):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return
    
    best = sorted(checkpoints, key=lambda x: float(x.split('_loss_')[1].split('.pt')[0]))[0]
    checkpoint_path = os.path.join(checkpoint_dir, best)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

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

if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    pairs = init_tokenizer('eng-dutch.tsv')
    dataset = TranslationDataset(pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(vocab_size).to(device)
    model.to(device)

    train_model(model, dataloader)