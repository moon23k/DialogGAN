import json, argparse, torch
from module import load_dataloader, load_model, Trainer
from peft import TaskType, LoraConfig
from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig

from module import (
    load_model, 
    load_dataloader,
    Trainer,
    Tester
)




class Config(object):
    def __init__(self, args):

        #Common attributes
        self.mode = args.mode
        self.strategy = args.strategy        
        self.mname = 'google/flan-t5-base'
        
        self.max_len = 512
        self.batch_size = 32

        self.ckpt = f"ckpt/{self.strategy}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Args for Tokenizer
        self.tok_args = {
            'pretrained_model_name_or_path': self.mname,
            'device_map': self.device,
            'load_in_8bit': True,
            'model_max_length': self.max_len
        }

        #Args for Model
        self.model_args = {
            'pretrained_model_name_or_path': self.mname,
            'device_map': self.device,
            'torch_dtype': torch.bfloat16,
            'trust_remote_code': True,
            'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        }
        
        #Args for PEFT
        self.peft_config = LoraConfig(
            r=16, #8~32
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )        

        #Args for Training
        self.training_args = {
            'lr': 1e-5,
            'n_epochs': 10,
            'clip': 1,
            'ckpt': self.ckpt,  
            'iters_to_generate': 10,
            'iters_to_accumulate': 4,
            'early_stop': True,
            'patience': 3
        }


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def main(args):
    #Prerequisites
    set_seed(42)
    config = Config(args)
    tokenizer = AutoTokenizer.from_pretrained(**config.tok_args)
    model = load_model(config)

    if config.mode == 'finetune':
        train_dataset = load_dataloader(tokenizer, 'train')
        valid_dataset = load_dataloader(tokenizer, 'valid')

        trainer = Trainer(config, model, tokenizer, train_dataset, valid_dataset)    
        trainer.train()

    elif config.mode == 'test':
        test_dataloader = load_dataloader(tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
    elif config.mode == 'inference':
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-strategy', required=True)

    args = parser.parse_args()
    assert args.mode.lower() in ['finetune', 'test', 'inference']
    assert args.strategy.lower() in ['std', 'gen']

    main(args)