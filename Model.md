# Chapter 14 - Network Compression（Part 1）

[Abstract](#Abstract)

[1.Network Purning](#1)

​		[1.1 神经网络修剪的基本原理](#1.1)

​		[1.2 Network Pruning - Practical Issue](#1.2)

[2.Knowledge Distillation（知识蒸馏）](#2)

​		[2.1 Knowledge Distillation基本原理](#2.1)

​		[2.2 训练技巧](#2.2)

[3.Parameter Quantization](#3)

​		[3.1 Parameter Quantization的三种解决方案](#3.1)

​		[3.2 Binary Connect Network](#3.2)

[4.Architecture Design](#3.2)

​	[4.1 隐层的增加与参数的减少](#4.1)

​	[4.2 Depthwise Separable Convolution](#4.2)

​	[4.3 More Related Paper](#4.3)

[5.Dynamic Computation](#5)

​	[5.1 计算资源与计算目标的动态调整](#5.1)



#### <span name="Abstract">Abstract：由于一些智能家居或移动设备的资源有限性，比如有限的存储空间、有限的计算能力等等，这些设备上搭载的神经网络不能够太大太复杂，否则设备可能会过载运转。因此需要研究网络压缩技术降低神经网络的规模。</span>



#### <span name="1">1.Network Purning</span>

1. <span name="1.1">神经网络修剪的基本原理</span>

   - Network Purning可行的原理是神经网络通常是over-parameterized，许多的链接权重和神经元都是冗余的，所以有效地进行修剪并不会有影响网络的功能。，其他学者提出小网络是可以训练的。（https://arxiv.org/abs/1810.05270）

       

2. <span name="1.2">Network Pruning - Practical Issue </span>

   - Weight 

   




#### <span name="2">2.Knowledge Distillation（知识蒸馏）</span>

1. <span name="2.1">Knowledge Distillation基本原理</span>

   - 先训练一个Large的过程。

   - 知识蒸馏在实践过程中的使用价值还有待验证，LHY老师的课题组初步使用知识蒸馏并没有得到很好的结果

     

2. <span name="2.2">训练技巧</span>

   - 在最后的输出层，需要对输

     


#### <span name="3">3.Parameter Quantization</span>
1. <span name="3.1">Parameter Quantization的三种解决方案</span>

   - Parameter Quantizati

     

2. <span name="3.2">Binary Connect Network</span>

   - Binary Connect

   - 

#### <span name="4">4.Architecture Design</span>

1. <span name="4.1">隐层的增加与参数的减少</span>

   - 对于Fully Connected Network，前一层有$N$个神经元，后一层有$M$个神经元，两层之间的权重为$W$，参数量为$N\times M$。通过结构重新设计的方法减少

     

2. <span name="4.2">Depthwise Separable Convolution</span>

   - Standard CNN Review：假

   - Depthwise Separable Convo

     

3. <span name="4.3">More Related Paper</span>

   - Squ

     


#### <span name="5">5.Dynamic Computation</span>

1. <span name="4.1">计算资源与计算目标的动态调整</span>

   - Dynamic Computation指根

     
