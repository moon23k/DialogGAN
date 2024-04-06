import torch, datasets



class Dataset(torch.utils.data.Dataset):

    def __init__(self, x_encodings, y_encodings):
        self.x_encoding = x_encodings
        self.y_encoding = y_encodings

    def __getitem__(self, idx):
        data = {'input_ids': self.x_encoding[idx].ids,
                'attention_mask': self.x_encoding[idx].attention_mask,
                'labels': self.y_encoding[idx].ids}
        return data

    def __len__(self):
        return self.x_encoding['input_ids'].size(0)
        


def load_dataset(tokenizer, split):    
    dataset = datasets.Dataset.from_json(f'data/{split}.json')

    x_encodings = tokenizer(dataset['x'], padding=True, truncation=True, return_tensors='pt')
    y_encodings = tokenizer(dataset['x'], padding=True, truncation=True, return_tensors='pt')
    
    dataset = Dataset(x_encodings, y_encodings)
    
    return dataset