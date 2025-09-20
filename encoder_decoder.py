
import random
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 12
EMBED_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 1e-3
EPOCHS = 10
MAX_SEQ_LEN = 8
PAD_TOKEN = 8
SOS_TOKEN = 1
EOS_TOKEN = 2


class ReverseDataset(Dataset):
    def __init__(self, n_samples:int, max_len:int):
        self.n_samples = n_samples
        self.max_len = max_len
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        L = random.randint(1, self.max_len-2)
        seq = [random.randint(3, VOCAB_SIZE-1) for _ in range(L)]
        src = seq +[EOS_TOKEN]
        tgt = [SOS_TOKEN]+list(reversed(seq))+[EOS_TOKEN]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
    


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    srcs, tgts = zip(*batch)
    src_lens = [s.size(0) for s in srcs]
    tgt_lens = [t.size(0) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    padded_src = torch.full((len(batch), max_src), PAD_TOKEN,dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), PAD_TOKEN, dtype=torch.long)
    for i, (s,t) in enumerate(zip(srcs, tgts)):
        padded_src[i: s.size(0)]=s
        padded_tgt[i: t.size(0)]=t
    return padded_src, torch.tensor(src_lens), padded_tgt, torch.tensor(tgt_lens)




class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs, mask=None):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len,1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0,2,1)
        v = self.v.repeat(encoder_outputs.size(0),1).unsqueeze(1)
        scores = torch.bmm(v, energy).squeeze(1)
        if mask is not None:
            scores - scores.masked_fill(mask==0, -1e9)
        return torch.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_TOKEN)
        self.use_attention = use_attention
        self.gru = nn.GRU(embed_size+ (hidden_size if use_attention else 0), hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size*(2 if use_attention else 1), vocab_size)
        if use_attention:
            self.attention = Attention(hidden_size)
    
    def forward(self, input_step, last_hidden, encoder_outputs, mask=None):
        embedded = self.embedding(input_step)
        embedded = embedded.squeeze(1)
        
        if self.use_attention:
            dec_hidden = last_hidden[-1]
            attn_weights = self.attention(dec_hidden, encoder_outputs, mask=mask)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            rnn_input = torch.cat((embedded, attn_applied), dim=1).unsqueeze(1)
        else:
            rnn_input = embedded.unsqueeze(1)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)
        if self.use_attention:
            output = torch.cat((output, attn_applied), dim=1)
        output = self.out(output)
        return output, hidden
            

#         return outputs
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoded = decoder
        
    def create_mask(self, src, src_lengths):
        mask = (src != PAD_TOKEN).to(DEVICE)
        return mask
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(0)
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(DEVICE)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src, src_lengths)
        input_step = tgt[:,0].unsqueeze(1)
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_step, hidden, encoder_outputs, mask=mask)
            outputs[:,t]=output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_step = tgt[:,t].unsqueeze(1) if teacher_force else top1


def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for src, src_lens, tgt, tgt_lens in tqdm(dataloader):
        src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(src, src_lens, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        outputs = outputs[:,1:].contiguous().view(-1, VOCAB_SIZE)
        target = tgt[:,1:].contiguous().view(-1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
    return total_loss/len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in dataloader:
            src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)
            outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            preds = outputs.argmax(-1)
            mask = (tgt != PAD_TOKEN)
            total_correct += (preds==tgt).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
    return total_correct/ total_tokens



def main():
    random.seed(42)
    torch.manual_seed(42)
    train_ds = ReverseDataset(2000, MAX_SEQ_LEN)
    val_ds = ReverseDataset(400, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, use_attention=True).to(DEVICE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val token accuracy: {acc*100:.2f}%")
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            src, _, tgt, _ = val_ds[random.randint(0, len(val_ds)-1)]
            src_b = src.unsqueeze(0).to(DEVICE)
            src_l = torch.tensor([src.size(0)]).to(DEVICE)
            outputs = model(src_b, src_l, tgt.unsqueeze(0).to(DEVICE), teacher_forcing_ratio=0.0)
            pred = outputs.argmax(-1).squeeze(0).cpu().tolist()
            print("SRC: ", src.cpu().tolist())
            print("TGT: ", tgt.cpu().tolist())
            print('PRED: ', pred)
            print("____")

if __name__ == "__main__":
    main()