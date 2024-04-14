import torch
from transformers import AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training, get_peft_model




def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")



def load_model(config):

    model = AutoModelForSeq2SeqLM.from_pretrained(**config.model_args)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.peft_config)
    model.config.use_cache = False

    #load finetuned states
    if config.mode != 'finetune':
        pass

    setattr(config, 'pad_token_id', model.config.pad_token_id)
    setattr(config, 'decoder_start_token_id', model.config.decoder_start_token_id)

    return model    