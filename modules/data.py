import json, torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset



def read_json(f_name):
    with open(f"data/{f_name}", 'r') as f:
        data = json.load(f)
    return data


class ChatDataset(Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.model_name = config.model_name
        self.data = read_json(f'{split}.json')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src_ids']
        if self.model_name == 'transformer':
            trg = self.data[idx]['trg'][:-1]
        else:
            trg = self.data[idx]['trg']
        return src, trg 


def collate_fn(batch):
    src_batch, trg_batch = [], []
    
    for src, trg in batch:
        src_batch.append(torch.LongTensor(src))
        trg_batch.append(torch.LongTensor(trg))
    
    batch_size = len(src_batch)
    eos_batch = torch.LongTensor([3 for _ in range(batch_size)])
    
    src_batch = pad_sequence(src_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    trg_batch = pad_sequence(trg_batch, 
                             batch_first=True, 
                             padding_value=1)
    
    trg_batch = torch.column_stack((trg_batch, eos_batch))

    return {'src': src_batch, 'trg': trg_batch}



def load_dataloader(config, split):
    dataset = ChatDataset(config, split)
    
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False, 
                      collate_fn=collate_fn, 
                      num_workers=2)