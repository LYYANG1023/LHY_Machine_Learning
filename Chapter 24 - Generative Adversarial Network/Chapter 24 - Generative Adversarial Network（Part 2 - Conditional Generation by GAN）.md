# Chapter 24 - Generative Adversarial Network（Part 2 - Conditional Generation by GAN）

[1.Conditional Generation by GAN](#1)

​		[1.1 Conditional GAN中Generator与Discriminator的设计](#1.1)

​		[1.2 Conditional GAN 的实际应用](#1.2)

[2.Unsupervised Conditional Generation by GAN](#2)

​		[2.1 Unsupervised Conditional Generation by GAN的提出背景](#2.1)

​		[2.2 实现方案一：Direct transformation](#2.2)

​		[2.3 实现方案二：Projection to Common Space](#2.3)

​		[2.4 Application](#2.4)



#### <span name="1">1.Conditional Generation by GAN</span>

1. <span name="1.1">Conditional GAN中Generator与Discriminator的设计</span>

   - 以Text-to-Image为例，介绍Conditional Generation by GAN的基本原理。

   - 使用Traditional Supervised Approach解决Text-to-Image问题，输入就是一段文字，输出就是对应的图片。这样解决存在的问题是，比如输入是“Train”，在训练集中对应的图片可能有许多种，有正面的火车，侧面的火车等等。那么在测试中，如果输入“Train”，产生的输出如果是正面或侧面的火车还可以接受，实际上产生的一般是各类火车图片的综合，那么可能就是一个四不像。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814154646982.png" alt="image-20200814154646982" style="zoom:33%;" />
     
     
     
   - Traditional GAN中Generator的输入是一个Normal Distribution $z$，输出是对应的Image。对于Conditional  GAN，Generator的输入是除了Normal Distribution $z$ 只要，还有一个Text $c$，输出不变。（[Scott Reed, et al, ICML, 2016]）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814155629036.png" alt="image-20200814155629036" style="zoom:33%;" />
     
   - Traditional GAN中Discriminator的输入是一张Image $x$，输出是对其的布尔值判断。对于Conditional  GAN，这种要求是不够的，因为Discriminator忽略了conditional input。Discriminator不但应该关注Generator的输出，还应该关注Discriminator的输入。只有文字和生成的图片对应起来，且图片质量很好时，才回被判定为1。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814155608599.png" alt="image-20200814155608599" style="zoom:33%;" />
     
     
     
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814155418617.png" alt="image-20200814155418617" style="zoom:33%;" />
     
     
     
   - Conditional  GAN的训练算法伪代码如下：

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814162927403.png" alt="image-20200814162927403" style="zoom: 67%;" />
     
   - Conditional  GAN中DIscriminator的具体实现方案有以下两种：

     - 第一种是将object $x$和condition $c$分别送入一个网络进行Embedding，然后将二者的结果送入另一个网络得到评分。
     
       ![image-20200814163236030](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814163236030.png)
       
     - 第二种是object $x$经过一个网络得到Embedding，根据这个Embedding直接得出“x is realistic or not ”的评分，然后将Embedding和condition $c$结合起来送入另一个网络，得到“c and x are matched or not”的评分。这种架构，通过拆分两个评分标准，表现也会更出色一些。（[Han Zhang, et al., arXiv, 2017]、[Takeru Miyato, et al., ICLR, 2018] 、[Augustus Odena et al., ICML, 2017]）
     
       ![image-20200814163518860](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814163518860.png)
     
       

2. <span name="1.2">Conditional GAN 的实际应用</span>

   - Stack GAN将生成图片的部分拆成两个阶段，先产生较小的Image，然后才产生较大的Image（Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas, “StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks”, ICCV, 2017）

     ![image-20200814163837148](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814163837148.png)

     

   - Image to Image（https://arxiv.org/pdf/1611.07004）：给图片上色。Traditional supervised approach就是收集图片对，然后作为输入输出，这种方法产生的图片比较模糊（类似于上文中提到的火车的例子）。因此可以使用Conditional GAN解决这个问题，具体还要加上很多约束，可以参见论文。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814164104032.png" alt="image-20200814164104032" style="zoom: 25%;" />

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814164230405.png" alt="image-20200814164230405" style="zoom: 25%;" />

     

   - Patch GAN：要求Discriminator每次只检查Image的一部分，增加Discriminator的甄别能力。（https://arxiv.org/pdf/1611.07004.pdf）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814164431763.png" alt="image-20200814164431763" style="zoom: 25%;" />

   - Speech Enhancement指去除一段声音讯号的背景噪声，增强主体声音。Conditional GAN的作用就是在去除噪声的同时，保证主体的声音与原讯号（Condition）一致。
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814164557866.png" alt="image-20200814164557866" style="zoom: 25%;" />
   
   - Video Generation
   
     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814164718801.png" alt="image-20200814164718801" style="zoom: 25%;" />
   
     

#### <span name="2">2.Unsupervised Conditional Generation by GAN</span>

1. <span name="2.1">Unsupervised Conditional Generation by GAN的提出背景</span>

   - Unsupervised Conditional Generation by GAN的一种典型案例就是风格转换，比如输入是油画，输出就是仿古画；输入的是男声，输出的是女生。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814192528619.png" alt="image-20200814192528619" style="zoom:33%;" />

     

   - Unsupervised Conditional Generation by GAN的实现方法有两种，第一种是Direct transformation，输入是Domain X，然后通过训练使模型直接输出Domain Y，这种方法一般只是小幅度的修改，适用于质地或改变颜色等等；第二种是Projection to Common Space，适用于变化比较大的Task。首先通过Encoder获得Domain X的脸部特征，然后使用Decoder根据获得的脸部特征生成一个动画图案。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814193046023.png" alt="image-20200814193046023" style="zoom:33%;" />

       

2. <span name="2.2">Direct Transformation</span>

   - 已知Domain X和Domain Y的数据，但是不知道两者之间的对应关系。Direct transformation的基本思路为：首先有一个Discriminator $D_Y$能够判断一张Image是否属于Domain Y。然后训练Generator $G_{X\rightarrow Y}$使得其输出能够骗过Discriminator $D_Y$，这样Generator $G_{X\rightarrow Y}$产生的东西就会类似Domain Y的东西。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814193709245.png" alt="image-20200814193709245" style="zoom: 33%;" />

   - 上述方法存在一个问题，如果Generator $G_{X\rightarrow Y}$只学会生成类似Domain Y的技能，那么如果其生成了一些与输入无关但是又很像Domain Y的东西，也是没有意义的。因为Generator $G_{X\rightarrow Y}$忽略了Domain  X的输入。

   - 此问题有很多解决方案，最暴力的就是忽略该问题，模型也可以训练的起来，可能的原因是因为Generator $G_{X\rightarrow Y}$在层数不多的情况下，输入和输出不会有很大的差异。[Tomer Galanti, et al. ICLR, 2018]

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814202005271.png" alt="image-20200814202005271" style="zoom:33%;" />

     

   - 第二种解决方案是，使用一个Pre-trained的神经网络，将Generator的输入和输出作为其输入，得到两个Embedding。此时不但需要Generator的输出要尽可能的骗过Discriminator，还要求Generator输入和输出的Embedding要尽可能的相似。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814202302974.png" alt="image-20200814202302974" style="zoom:33%;" />

   - 第三种解决方案是Cycle GAN，需要训练Generator $G_{X\rightarrow Y}$和Generator $G_{Y \rightarrow X}$。Generator $G_{X\rightarrow Y}$能够将Domain X转换为Domain Y，Generator $G_{Y \rightarrow X}$能够将Domain Y还原回Domain $X'$。此时要求Generator $G_{X\rightarrow Y}$的输出不但要能骗过Discriminator，还要求$X'$尽可能的与X相似，只有这样才能保证Generator $G_{X\rightarrow Y}$产生的东西不会和输入有过多的变化。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814205523956.png" alt="image-20200814205523956" style="zoom: 25%;" />

   - Cycle GAN还可以改进为双向的。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814205736201.png" alt="image-20200814205736201" style="zoom:25%;" />

   - Cycle GAN进行发色转换的例子，可见https://github.com/Aixile/chainer-cyclegan

   - Cycle GAN存在一个问题，就是会将Input的一些特征隐藏起来。如下图所示，左边的原图和右边的重构图白色楼房上都有黑点，但是中间的图片却没有。可能的原因是，Cycle GAN将一些特征以很小的数值隐藏在图中，那么这就失去了Cycle GAN的意义，比如模型将很大改的改动都隐藏起来。该问题的更深入本质还有待研究。[Casey Chu, et al., NIPS workshop, 2017] 

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814210127306.png" alt="image-20200814210127306" style="zoom:33%;" />

   - 与Cycle GAN很相似的还有很多，例如Disco GAN、Dual GAN等等（不同的人在同一时间提出了同样的想法）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814210335022.png" alt="image-20200814210335022" style="zoom:25%;" />

   - 鉴于Cycle GAN只能将Domain X转换为Domain Y，但是有一些场景可能需要多个Domain之间的转换，这是就可以使用starGAN，使用一个Generator实现多个Domain的互转。[Yunjey Choi, arXiv, 2017]

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814210536059.png" alt="image-20200814210536059" style="zoom: 33%;" />

   - starGAN的Discriminator不但要鉴别输入的真假，还要鉴别输入属于哪一个Domain。Generator的输入为Input Image和Target Domain，得到的输出和Original Domain作为输出在此送入Generator，要求其重构图像。最后希望Discriminator认为Fake Image足够真实，然后还属于目标Domain。

     ![image-20200814210902735](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814210902735.png)

     

   - StarGAN的实际案例如下：Domain的区别可以通过 $[Blac/Blond/Brown/Male/Young]$ 的向量描述。

     ![image-20200814211005909](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814211005909.png)

     ![image-20200814211325694](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814211325694.png)

     

3. <span name="2.3">Projection to Common Space</span>

   - Projection to Comon Space的核心思想是将Input Object映射到某一个Latent Space，在用Decoder将其重构回来。首先如果输入是Domain X，那么$Encoder X$会抽取其Latent Attribute，如果输入是Domain Y，$Encoder Y$会抽取其Latent Attribute。$Encoder X$和$Encoder Y$是两个网络（参数不同）。Latent Attribute如果进入$Decoder X$就会得到真实人物的Image，如果进入$Decoder Y$就会得到动画人物的Image。在风格转换的过程中，按照红色箭头的方向进行就可以得到预期的输出结果。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814221930896.png" alt="image-20200814221930896" style="zoom:33%;" />

   - 因为只有Domain X和Domain Y的数据，没有二者之间的联系，所以需要进行Unsupervised Learning。Projection to Comon Space的训练方法是，将$Encoder X$和$Decoder X$组成一个Auto-Encoder，利用Domain X的数据进行训练；将$Encoder Y$和$Decoder Y$组成一个Auto-Encoder，利用Domain Y的数据进行训练。除此之外，还可以再加入一个Discriminator $D_X$和$D_Y$，要求输出必须和原始的Domain相同。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814225047267.png" alt="image-20200814225047267" style="zoom:33%;" />

   - $Encoder X +Decoder X + D_X$可以看做一个VAE GAN。因为蓝色部分和绿色部分是分开进行训练的，所以上下两部分没有任何关联。两个Encoder生成的Latent Space Vector并不在一个空间中。可以理解为$Decoder Y$看不懂$Encoder X$的表达方式。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814225912968.png" alt="image-20200814225912968" style="zoom:33%;" />

   - 无关联问题的第一种解决方案为Couple GAN [Ming-Yu Liu, et al., NIPS, 2016]和UNIT [Ming-Yu Liu, et al., NIPS, 2017]。基本思想为将两个Encoder的部分参数绑定在一起，比如后几层Hidden Layer共用参数；将两个Decoder的部分参数绑定在一起，前几层Hidden Layer共用参数。通过参数绑定的方法要求两个Encoder尽可能的将输入压缩到同样的Latent Space。共享参数的极端情况就是两个Encoder变为一个Encoder，再输入不同的Domain Object时，给一个标志位让网络知道输入是来自哪个Domain的。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814231355720.png" alt="image-20200814231355720" style="zoom:33%;" />

   - 此时可以增加一个Domain Discriminator，用于判断Encoder输出的Latent Space Vector属于哪一个，Encoder的一个目标就是骗过Discriminator，只有在Discriminator无法辨别Latent Space Vector属于哪一个Domain时，就可以认为两个Encoder将输入压缩到了一个Latent Space中。[Guillaume Lample, et al., NIPS, 2017]

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814232357165.png" alt="image-20200814232357165" style="zoom:33%;" />

   - 无关联问题的第二种解决方案为Cycle Consistency。$Encoder X \rightarrow Latent\ Space\ Vector \ \rightarrow Decoder Y $$\rightarrow Encoder Y \rightarrow Latent\ Space\ VectorDecoder X \rightarrow D_X$。要求整个流程的输入和输出的Reconstruction Error最小化。这样的训练方法还用在ComboGAN [Asha Anoosheh, et al., arXiv, 017]中。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814233612354.png" alt="image-20200814233612354" style="zoom:33%;" />

   - 除了Cycle Consistency之外，还有Semantic Consistence，要求两次Latent Space Vector尽可能的相似。Cycle Consistency比较的是两个Image的Picel-wised Error，而Semantic Consistence比较的更多是Latent Space中的语义误差。这种思想被应用在UDTN [Yaniv Taigman, et al., ICLR, 2017]和XGAN [Amélie Royer, et al., arXiv, 2017]中。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814234033755.png" alt="image-20200814234033755" style="zoom:33%;" />

     

4. <span name="2.4">Application</span>

   - 世界二次元化：https://github.com/Hi-king/kawaii_creator 

   - Voice Conversion

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 24 - Generative Adversarial Network/image-20200814234626293.png" alt="image-20200814234626293" style="zoom:33%;" />

   - Reference：

     - Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros, Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV, 2017
     - Zili Yi, Hao Zhang, Ping Tan, Minglun Gong, DualGAN: Unsupervised Dual Learning for Image-to-Image Translation, ICCV, 2017
     - Tomer Galanti, Lior Wolf, Sagie Benaim, The Role of Minimal Complexity Functions in Unsupervised Learning of Semantic Mappings, ICLR, 2018
     - Yaniv Taigman, Adam Polyak, Lior Wolf, Unsupervised Cross-Domain Image Generation, ICLR, 2017
     - Asha Anoosheh, Eirikur Agustsson, Radu Timofte, Luc Van Gool, ComboGAN: Unrestrained Scalability for Image Domain Translation, arXiv, 2017
     - Amélie Royer, Konstantinos Bousmalis, Stephan Gouws, Fred Bertsch, Inbar Mosseri, Forrester Cole, Kevin Murphy, XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings, arXiv, 2017
     - Guillaume Lample, Neil Zeghidour, Nicolas Usunier, Antoine Bordes, Ludovic Denoyer, Marc'Aurelio Ranzato, Fader Networks: Manipulating Images by Sliding Attributes, NIPS, 2017
     - Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim, Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, ICML, 2017
     - Ming-Yu Liu, Oncel Tuzel, “Coupled Generative Adversarial Networks”, NIPS, 2016
     - Ming-Yu Liu, Thomas Breuel, Jan Kautz, Unsupervised Image-to-Image Translation Networks, NIPS, 2017
     - Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo, StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, arXiv, 2017