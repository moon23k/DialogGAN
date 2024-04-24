import json, torch, datasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split):
        self.tokenizer = tokenizer
        self.data = self.load_data(split)
    
    @staticmethod
    def load_data(split):
        with open(f'data/{split}.json', 'r') as f:
            data = json.load(f)
        return data


    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.data[idx]['x'], return_tensors='pt')        
        labels = self.tokenizer.encode(self.data[idx]['y'], return_tensors='pt')

        return map(lambda x: x.squeeze(), (inputs['input_ids'], inputs['attention_mask'], labels))

    def __len__(self):
        return len(self.data)
                


class Collator(object):

    def __init__(self, pad_token_id, decoder_start_token_id):
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id


    def __call__(self, batch):

        input_batch, mask_batch, label_batch = zip(*batch)

        input_ids = self.pad_batch(input_batch)
        attention_mask = self.pad_batch(mask_batch)
        
        labels = self.pad_batch(label_batch)
        labels.masked_fill_(labels == self.pad_token_id, -100)
        decoder_input_ids = self.shift_right(labels)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'decoder_input_ids': decoder_input_ids 
                }


    def shift_right(self, labels):
        dec_input_ids = labels.new_zeros(labels.shape)
        dec_input_ids[..., 1:] = labels[..., :-1].clone()
        dec_input_ids[..., 0] = self.decoder_start_token_id
        dec_input_ids.masked_fill_(dec_input_ids == -100, self.pad_token_id)
        return dec_input_ids


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )


def load_dataloader(config, tokenizer, split):
    
    dataset = CustomDataset(tokenizer, split)
    collator = Collator(config.pad_token_id, config.decoder_start_token_id)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=config.mode=='finetune',
        collate_fn=collator,
        pin_memory=True,
        num_workers=2
    )    
    
    return dataloader