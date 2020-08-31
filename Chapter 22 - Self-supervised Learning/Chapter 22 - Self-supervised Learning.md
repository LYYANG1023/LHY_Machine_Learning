# Chapter 22 - Self-supervised Learning

[Abstract](#Abstract)

[1.Self-supervised Learning](#1)

​		[1.1 Self-supervised Learning的常见模型](#1.1)

[2.Reconstruction Task](#2)

​		[2.1 Reconstruction on Text](#2.1)

​		[2.2 Reconstruction on Image](#2.2)

[3.Contrastive Learning](#3)

​		[3.1 CPC和SimCLR](#3.1)

[4.Reference](#4)



#### <span name="Abstract">Abstract：在Supervised Learning中，如果Labeled ata足够多，那么NN可以达到不错的效果。与此同时，一些隐藏层的Representation蕴含了相当多的信息，可以进行二次利用。由于Labeled Data的成本是很高的，但Unlabelled Data确实取之不尽，用之不竭的。所以Unsupervised Learning有着很大的研究空间。</span>



#### <span name="1">1.Self-supervised Learning</span>

1. <span name="1.1">Self-supervised Learning的常见模型</span>

   - Self-supervised Learning是一种Unsupervised Learning，一般是使用一部分数据进行训练，然后对剩余的一部分数据进行预测。

   - Self-supervised Learning常见的方法有：Reconstruct from a corrupted (or partial) data、Visual common sense tasks、Contrastive Learning。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811212323959.png" alt="image-20200811212323959" style="zoom:50%;" />

     

#### <span name="2">2.Reconstruction Task</span>

2. <span name="2.1">Reconstruction on Text</span>

   - Reconstruct from a corrupted (or partial) data on Text相当于一个Denoising Auto Encoder。在Denoising Auto Encoder中使用的是Encoder生成的Compressed Representation，重心在Encoder。但是在Reconstruction中需要整个Model，将整个模型拿到下一个任务做FIne-tune。

     ![image-20200811212557656](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811212557656.png)
   
   - BERT-Family：
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811213614623.png" alt="image-20200811213614623" style="zoom:50%;" />
   
   - Language Model：A statistical language model is a probability distribution over sequences of words. Given such a sequence, say of length m, it assigns a probability $P(w_1,\dots,w_m)$ to the whole sequence。例如 $P(“Is it raining now?”) > P(“Is it raining yesterday?”)$。具体的计算方式有N-gram和Neural Network。
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811215008738.png" alt="image-20200811215008738" style="zoom:50%;" />
   
     
   
   - ELMO
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811215311312.png" alt="image-20200811215311312" style="zoom:50%;" />
   
     
   
   - GPT
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811215415639.png" alt="image-20200811215415639" style="zoom:50%;" />
   
     
   
   - BERT，使用Masked LM。
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811215550170.png" alt="image-20200811215550170" style="zoom:50%;" />
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811215738196.png" alt="image-20200811215738196" style="zoom:50%;" />
   
     
   
   - GPT、ELMO属于Autoregressive Language Model（ARLM），不存在data corruption的问题（即Pre-train和Fine-tune阶段数据是一样的，不会有没见过的），只能是单向的；BERT属于Autoencoding Language Model（AELM a.k.a MaskedLM），可以是双向的预测。
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811220203649.png" alt="image-20200811220203649" style="zoom:50%;" />
   
     
   
   - XLNet - Permutation LM（随机置换输入的顺序 + Predicting next word）
   
     ![image-20200811220350630](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811220350630.png)
   
     
   
   - BART：Encoder部分和BERT相同，在Decoder部分进行AutoRegressive LM的步骤。
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811220819223.png" alt="image-20200811220819223" style="zoom:50%;" />
   
     
   
   - ELECTRA
   
     ![image-20200811220937566](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811220937566.png)
   
     
   
   - RoBERTa：A Robustly Optimized BERT Pretraining Approach
   
   - ALBERT：A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
   
     
   
2. <span name="2.2">Reconstruction on Image</span>

   - Predict missing pieces：挖掉图片的一部分，然后进行复原。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811221218901.png" alt="image-20200811221218901" style="zoom:50%;" />

   - Solving Jigsaw Puzzles

   - Rotation

     ![image-20200811221407021](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811221407021.png)

   




#### <span name="3">3.Contrastive Learning</span>

1. <span name="3.1">CPC和SimCLR</span>

   - Contrastive Learning的一个例子是Contrastive Predictive Coding（CPC）。CPC不在进行的是Auto regressive任务中的进行$x_{t+1}$的预测，而是预测$x_{t+2}$。类似于Word2Vec的想法。

     ![image-20200811221755563](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811221755563.png)

     

   - Contrastive Learning的另一个例子是SimCLR

     ![image-20200811222042723](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 22 - Self-supervised Learning/image-20200811222042723.png)




#### <span name="4">4.Reference</span>

- CS294-158 Deep Unsupervised Learning Lecture 7 
- AAAI 2020 Keynotes Turing Award Winners Event 
- Learning From Text - OpenAI
- Learning from Unlabeled Data - Thang Luong