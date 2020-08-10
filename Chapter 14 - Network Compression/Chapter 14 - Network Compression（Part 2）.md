# Chapter 14 - Network Compression（Part 2）

[Abstract](#Abstract)

[1.Network Compression Review](#1)

​		[1.1 Network Compression常用的解决办法](#1.1)

[2.Knowledge Distillation（知识蒸馏）](#2)

​		[2.1 Knowledge Distillation基本原理](#2.1)

​		[2.2 Logits Distillation](#2.2)

​		[2.3 Feature Distillation](#2.3)

​		[2.4 Relational Distillation](#2.4)

[3.Network Purning](#3)

​		[3.1 Network Purning Case](#3.1)

​		[3.2 Evaluate Importance](#3.2)

​		[3.3 More About Lottery Ticket Hypothesis](#3.3)

[4.Reference](#4)



#### <span name="Abstract">Abstract：由于一些智能家居或移动设备的资源有限性，比如有限的存储空间、有限的计算能力等等，这些设备上搭载的神经网络不能够太大太复杂，否则设备可能会过载运转。因此需要研究网络压缩技术降低神经网络的规模。</span>



#### <span name="1">1.Network Compression Review</span>

1. <span name="1.1">Network Compression常用的解决办法</span>

   - Network Compression的主要方法包括：Network Pruning、Knowledge Distillation、Architecture Design、parameter Quantization

      <img src="./image-20200727202307473.png" alt="image-20200727202307473" style="zoom:33%;" /><img src="./image-20200727202333562.png" alt="image-20200727202333562" style="zoom:33%;" />

   - 网络压缩领域的方法多种多样，在实际应用中，可以综合多种手段进行

      <img src="./image-20200727202741725.png" alt="image-20200727202741725" style="zoom:33%;" />

      


#### <span name="2">2.Knowledge Distillation（知识蒸馏）</span>

1. <span name="2.1">Knowledge Distillation基本原理</span>

   - Knowledge Distillation中“蒸馏”的东西到底是什么？目前主要分为两个方向，一个是输出值（学习输出值的分布），一个是中间值（学习特征是如何转换的）

     <img src="./image-20200727203219387.png" alt="image-20200727203219387" style="zoom:33%;" />

   - Knowledge Distillation可以学习到类别之间的隐藏关系，例如“1”、“7”、“9”之间的相似性等等

   - 一般的模型是存在Incompleteness和Inconsistency的问题。如下图所示，图片的label是一只猫，但是机器学出来的结果可能是“猫”：0.9、“球”：0.2，这就是Overfitting（或是Incompleteness）的一种表现。一般在模型训练中还会采用Crop方式，即在原始输入中裁剪出一些小的片段，如果裁剪出的刚好是球，这就导致了Inconsistency的问题。

     <img src="./image-20200728114932580.png" alt="image-20200728114932580" style="zoom:33%;" />

   - 解决上述问题的一种方式是Label Refinery，即对输入的标签进行精细化。每次Refined的時候都可以降低Incompleteness & Inconsistency, 并且还可以学到label之间的关系。

     <img src="./image-20200728114847809.png" alt="image-20200728114847809" style="zoom:50%;" />

   - 在Knowledge Distillation中，即使Teacher Network是很大的模型，Student Network是很小的模型。但是Teacher Network的Accuracy并不是Student Network的上界，Student Network可以做的更好。

     

2. <span name="2.2">Logits Distillation</span>

   - KD中的Baseline模型，使用小模型去学习大模型的输出（最小化两个模型的交叉熵），在Softmax之前除以$T$，防止大模型的训练效果比较好时会导致输出近似于One-hot的情况。

     <img src="./image-20200728123112750.png" alt="image-20200728123112750" style="zoom: 50%;" />
     
   - Deep Mutual Learning
   
     - 核心思想：让两个模型同时训练，互相学习对方的logits。如果训练初期两个模型的Capacity都不够强的话，互相学习就会变成“菜鸡互啄”。所以Mutual Learning会设置两个损失函数，一个是Knowledge Distillation Loss Function，即尽可能学习其他神经网络的知识，第二个是正常的Label Loss Function，指导模型学习正常的标签信息。
   
       <img src="./image-20200728123634691.png" alt="image-20200728123634691" style="zoom:50%;" />
   
     - 实现方法：①Step1 - 将实际输出和真实label的交叉熵损失和两个模型输出的KL散度损失加和起来，构成Network 1的损失函数，并据此进行更新；②Step2 - 类似于模型一的损失函数，计算Network 2的损失函数，并据此进行更新。（https://slides.com/arvinliu/kd_mutual）
   
       <img src="./image-20200728123755752.png" alt="image-20200728123755752" style="zoom: 25%;" />          <img src="./image-20200728124708534.png" alt="image-20200728124708534" style="zoom:25%;" />
   
     - 实验结果：如果两个神经网络是相同的架构，那么使用Mutual Learning的方式会比Independent Learning的效果要好；如果两个神经网络的结构不同，一大一小，小模型并不一定会拉低大模型的准确率，有可能会为大模型提供更多的信息。
   
       
   
   - Born Again Neural Networks：该模型类似于Label Refinery，差异在于：①初始Model是KD來的；②
     迭代使用Cross Entropy；③最后Ensemble 所有Student Model。
   
     <img src="./image-20200728125316804.png" alt="image-20200728125316804" style="zoom:33%;" />
   
   - Pure Logits KD中的潜在问题：在logits KD中，可能会出现Student Network的效果不好的情况，这可能是因为Teacher Network的能力过强，Student Network不知道如何学习。
   
     <img src="./image-20200728164612300.png" alt="image-20200728164612300" style="zoom:33%;" />
   
   - 解决Student Network无法学习Teacher Network知识的方法就是，加入Teacher Assistant。Teacher Assistant的参数量介于Teacher和Student之间，作为中间人避免模型差距过大的问题。
   
     <img src="./image-20200728165013745.png" alt="image-20200728165013745" style="zoom:33%;" />
   
     
   
3. <span name="2.3">Feature Distillation</span>

   - 以数字辨识为例，对于数字“0”，Teacher Network可能学习到两个判别特征：①只有一个圈；②没有端点，那么这张图片应该大概率被判定为“0”，然后小概率被判定为“8”。将这样的输入和输出作为知识让Student Network去学习， Student Network可能无法理解为什么会产生这样的输出（图中的黑人问号）。解决方案就是，不但把输入输出告诉Student Network，还把中间的判别标准也告诉Student Network（向上的红色箭头）。

     <img src="./image-20200728170522730.png" alt="image-20200728170522730" style="zoom:33%;" />

   - FitNet

     - 核心思想：先让Student学习如何产生Teacher的中间Feature，之后再使用Baseline KD。

       <img src="./image-20200728170719271.png" alt="image-20200728170719271" style="zoom: 50%;" />

     - 实现方法：每个蓝色方框表示一层Hidden Layer。在FitNet中两者的结构越相近，结果越好。

       ​		Step 1：在Teacher Network中选择需要学习的Feature，在Student Network中选择在第几层学习该Feature。然后将两个隐层的输出进行Fit，让Student Network隐层输出的二范数距离尽可能接近Teacher Network。其中$W_r$的作用是将Student Network的隐层输出的大小与Teacher Network一致，之后才能进行比较。

       <img src="./image-20200728171315400.png" alt="image-20200728171315400" style="zoom:33%;" />

       ​		Step 2：使用Base KD方法，让Student Network学习Teacher Network的输出。

       <img src="./image-20200728171611053.png" alt="image-20200728171611053" style="zoom:33%;" />

   - FitNet中的潜在问题：①模型的Capacity是不同的（Teacher Network过于强大，Student Network过于简单，Student Network进行学习时可能会疯掉）；②Teacher Network蕴含着很多冗余的信息，导致Student Network不知道什么东西是值得学的，因为capacity不足，导致学的一知半解。

     <img src="./image-20200728172403014.png" alt="image-20200728172403014" style="zoom:33%;" />

   - 解决FitNet中潜在问题的方法有：以CNN为例，对Feature Map做Knowledge Compression（可以让Student学习Teacher的Attention Map）

     <img src="./image-20200728172551764.png" alt="image-20200728172551764" style="zoom: 25%;" /><img src="./image-20200728172659807.png" alt="image-20200728172659807" style="zoom: 33%;" />

     

4. <span name="2.4">Relational Distillation</span>

   - Conventional KD的只要目标是以每个sample为单位做知识蒸馏，Relational KD是以sample之间的关系做知识蒸馏。图中蓝色的$t_i$表示Teacher Network的输出，绿色的$s_i$表示Student Network的输出，Conventional KD是让$s_i$接近$t_i$，Relational KD是让$s_i$之间的关系近似于$t_i$之间的关系

     <img src="./image-20200728173516161.png" alt="image-20200728173516161" style="zoom:50%;" />

   - Relational KD要求Student Network学习Teacher Model的Representation，即Logits Distribution / Relationship。

     <img src="./image-20200728173625354.png" alt="image-20200728173625354" style="zoom:50%;" />

   - Relational KD中结构的关系可以描述为Distance-wise KD（L-2 Distance）、Angle-wise KD（余弦相似度）

     <img src="./image-20200728173908024.png" alt="image-20200728173908024" style="zoom:33%;" />

   - Similarity-Preserving KD：借鉴Relational KD的概念，对Feature之间的Relational Information做蒸馏。以MINIST任务为例，模型会学习到两个判别特征：Circle和Vertical Line。对不同图片的两个Feature的数值做余弦相似度计算，就代表着Feature之间的Relational Information。Similarity-Preserving KD就是让Student Network学习Teacher Network的Feature Relational Information，这个信息是经过余弦相似度压缩的，所以不会出现FitNet中学不会的情况。

     <img src="./image-20200728174206064.png" alt="image-20200728174206064" style="zoom: 33%;" /><img src="./image-20200728174642056.png" alt="image-20200728174642056" style="zoom:33%;" />

   - Similarity-Preserving KD的实现方法：蒸馏凉凉sample的Activation相似性

     <img src="./image-20200728174759710.png" alt="image-20200728174759710" style="zoom:33%;" />

     


#### <span name="3">3.Network Purning</span>
1. <span name="3.1">Network Purning Case</span>

   - 以DNN为例，有四个Feature Input（a=4, b=3, c=2）。假设移除了中间层的一个神经元，那么参数数量从$(a+c)*b$变成了$(a+c)*(b-1)$

     <img src="./image-20200728180323621.png" alt="image-20200728180323621" style="zoom:33%;" />

   - 以CNN为例，有四个Feature Map（a=4, b=3, c=2）。Conv（4, 3, 3）指四个channel到三个channel，kernel size为$3\times 3$。假设移除了中间层的一个Feature Map，那么参数数量从$(a+c) * b * k * k $变成了$(a+c) * (b-1) * k * k$

   - Network Purning的核心问题是如何衡量Neuron的重要性（包括Evaluate by Weight、Evaluate by Activation、Evaluate by Gradient）

     

2. <span name="3.2">Evaluate Importance</span>

   - 以Evaluate by Weight - sum of L1 Norm为例，计算每个Feature Map的L1范数之和，根据和的大小判定重要性

     <img src="./image-20200728183220500.png" alt="image-20200728183220500" style="zoom:33%;" />

   - 以Evaluate by Weight - FPGM为例。

     - 计算出Feature Map的L-Norm，根据L-Norm的分布修剪掉处于norm较低的部分。

       <img src="./image-20200728183306943.png" alt="image-20200728183306943" style="zoom:33%;" />FPGM中存

     - Norm的方差过小（无法选取合适的阈值）和所有Norm都不接近0（所有Norm都是有意义的）两个难点。解决办法是将所有Norm转换到几何中心（Geometric Media），修剪掉重复性较强的Feature Map即可。

     <img src="./image-20200728183621647.png" alt="image-20200728183621647" style="zoom:33%;" />

   - Eval by BN's γ  - Network Slimming

     <img src="./image-20200728184032706.png" alt="image-20200728184032706" style="zoom:33%;" />

     <img src="./image-20200728184051301.png" alt="image-20200728184051301" style="zoom:33%;" />

   - Eval by 0s after ReLU - APoZ

     <img src="./image-20200728184150359.png" alt="image-20200728184150359" style="zoom:33%;" />

     

3. <span name="3.3">More About Lottery Ticket Hypothesis</span>

   - 在Lottery Ticket Hypothesis的Paper中，作者使用L1-Norm进行权重修剪。但是有没有一种可能是，在随机初始化时，因为其在的初始化权重比较大，所以最后的权重比较大，从而被选中。但是其蕴含的信息不如一些因为开始被初始化为较小的值而被修剪的权重。实验表明这种考虑是不必要的，Magnitude & Large_final的参数时最好的衡量标准

     <img src="./image-20200728184758920.png" alt="image-20200728184758920" style="zoom:33%;" />

   - 有学者研究什么样的权重可以成为winning tickets。实验表明init sign是比较重要的

     

#### <span name="4">4.Reference</span>

Distilling the Knowledge in a Neural Network (NIPS 2014)

Deep Mutual Learning (CVPR 2018)

Born Again Neural Networks (ICML 2018)

Label Refinery: Improving ImageNet Classification through Label Progression

Improved Knowledge Distillation via Teacher Assistant (AAAI 2020)

FitNets : Hints for Thin Deep Nets (ICLR2015)

Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer (ICLR 2017)

Relational Knowledge Distillation (CVPR 2019)

Similarity-Preserving Knowledge Distillation (ICCV 2019)