## C2_Bot
$C^2 Bot$ referes to a **C**haracteristic **C**hat **Bot**

This repo covers implementation of seqGAN with my own edits. SeqGAN arcitecture apply GAN to NLG Task via Reinforcement Learning Technique. Main idea borrowed from seqGAN, but the ways of configuring models and Loss functions are somewhat different. In the original paper, Policy Gradient was used by getting rewards from discriminator and Roll-outs. But In my case, I rather used output of discriminator as penalty than rewards. Though these two approaches look different, the main goal to draw is the same.

<br>

## Model desc
Just like in GAN, C2_Bot also consists of Generator and Discriminator. But C2_Bot uses Policy Gradient for Adversarial Learning.

**Generator**

<br>

**Disciminator**
* Variant of Transformer Architecture
* Use of BERT Binary Classification
<br>

**C2 Bot**

<br>
<br>

## Configurations
**Training flows**
1. Train Generator (훈련종료시점은 epoch별로 혹은 기준 Loss이하인 경우 자동 종료 둘중 하나로)
2. Generate Samples 
3. Train Discriminator
4. Train C2 Bot on new Dataset (Use of Pre-Trained Generator and Discriminator)

<br>
<br>

## Reference

<br>
