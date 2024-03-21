## LLM Efficient Fine-Tuning (LEFT)
Recently, Large Language Models (LLMs) have exhibited outstanding performance. However, their sheer scale presents challenges, making even fine-tuning difficult in typical computing environments.
This repository shares a series of codes incorporating various methodologies such as **Efficient Training Strategies** and **PEFT** to enable efficient fine-tuning of LLMs. 
Hence, the repository is named **Large Language Model Efficient Fine Tuning**, or simply **LEFT**. The strategies, usage guidelines, and brief experimental results employed in the code are further detailed below.

<br><br> 

## Efficient Training

**Mixed Precision Training** <br>
> In the training process of most deep learning models, both data and model parameters are typically represented in float32 type. In contrast, Mixed precision training entails training the model by mixing single precision (FP32) and half-precision (FP16). Utilizing FP16 offers advantages in both memory efficiency and training time efficiency, thus enhancing overall efficacy.

<br>

**Gradient Accumulation** <br> 

<br><br> 

## Parameter Efficient Fine-Tuning (PEFT)

**Prompt Tuning** <br> 
> Prompting helps guide language model behavior by adding some input text specific to a task. Prompt tuning is an additive method for only training and updating the newly added prompt tokens to a pretrained model. This way, you can use one pretrained model whose weights are frozen, and train and update a smaller set of prompt parameters for each downstream task instead of fully finetuning a separate model. As models grow larger and larger, prompt tuning can be more efficient, and results are even better as model parameters scale.

<br>

**Prefix Tuning** <br> 
> Prefix tuning is an additive method where only a sequence of continuous task-specific vectors is attached to the beginning of the input, or prefix. Only the prefix parameters are optimized and added to the hidden states in every layer of the model. The tokens of the input sequence can still attend to the prefix as virtual tokens. As a result, prefix tuning stores 1000x fewer parameters than a fully finetuned model, which means you can use one large language model for many tasks.

<br>

**P Tuning** <br> 
> It is challenging to finetune large language models for downstream tasks because they have so many parameters. To work around this, you can use prompts to steer the model toward a particular downstream task without fully finetuning a model. Typically, these prompts are handcrafted, which may be impractical because you need very large validation sets to find the best prompts. P-tuning is a method for automatically searching and optimizing for better prompts in a continuous space.

<br>

**LoRA** <br> 
> Low-Rank Adaptation (LoRA) is a reparametrization method that aims to reduce the number of trainable parameters with low-rank representations. The weight matrix is broken down into low-rank matrices that are trained and updated. All the pretrained model parameters remain frozen. After training, the low-rank matrices are added back to the original weights. This makes it more efficient to store and train a LoRA model because there are significantly fewer parameters.

<br>

**IA3** <br> 
> IA3 refers to "Infused Adapter by Inhibiting and Amplifying Inner Activations".

<br><br>

## Setup

<br><br>

## Results

<br><br>

## How to Uses
```
git clone 
```

## Reference

<br>
