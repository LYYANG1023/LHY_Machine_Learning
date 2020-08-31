# Chapter 26 - Deep Reinforence Learning（Part 4 - Asynchronous Advantage Actor-Critic（A3C））

[1.Reinforence Learning Introduction](#1)

​		[1.1 RL的术语与基本思想](#1.1)

​		[1.2 RL的特点](#1.2)

​		[1.3 RL Outline](#1.3)

[2.Policy-based Approach（Learning an Actor）](#2)

​		[2.1 Policy-based Approach三步走](#2.1)

​		[2.2 Step 1：Neural Network  as Actor](#2.2)

​		[2.3 Step 2：Goodness of Actor](#2.3)

​		[2.4 Step 3：Pick the best Actor](#2.4)

[3.Value-based Approach（Learning a Critic）](#3)

​		[3.1 Critic的定义（State Value Function）](#3.1)

​		[3.2 Estimating Critic（State Value Function）](#3.2)

​		[3.3 Critic的定义（State-action Value Function）](#3.3)

[4.Actor-Critic](#4)

​		[4.1 A3C（Asynchronous Advantage Actor-Critic）](#4.1)

[5.Inverse Reinforence Learning](#5)

​		[5.1 Imitation Learning](#5.1)



#### Abstract：A3C是Actor-Critic方法中最知名的一种，其原始文章为：[Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu, “Asynchronous Methods for Deep Reinforcement Learning”, ICML, 2016]



#### <span name="1">1.Reinforence Learning Introduction</span>

1. <span name="1.1">RL的术语与基本思想</span>

   - Policy Gradient Review：Policy Gradient的更新量为$\nabla\tilde{R}_\theta$如下。令$G_t^n=\sum\limits_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n$表示Cumulated Reward。实际上由于互动过程存在一定的随机性，$G_t^n$是一个Random Variable。我们是通过采样的方式计算$G_t^n$的值，所以结果很不稳定的，$G_t^n$就变成一个一个Variance很大的Random Variable。只有在采样的数据足够多的情况下，才能准确估计$G_t^n$，但这在实际训练时是不可能的。所以提出一种想法，训练一个网络去估计$G_t^n$的期望值。

     <img src="/Users/liuyuyang/image-20200826202825240.png" alt="image-20200826202825240" style="zoom: 50%;" />
     
   - Q-Learning Review：
   
     <img src="/Users/liuyuyang/image-20200826203932703.png" alt="image-20200826203932703" style="zoom:50%;" />
     
   - Actor-Critic：使用Q-Learning的方法计算Policy Gradient中累计收益的期望值。$Q^{\pi_\theta}(s_t^n,a_t^n)$的定义恰好就是Cumulated Reward Expect，所以使用$Q^{\pi_\theta}(s_t^n,a_t^n)$代替$G_t^n$。因为在State $s$时，$V^{\pi_\theta}(s_t^n)$是没有引入Action的Cumulated Reward Expect，$Q^{\pi_\theta}(s_t^n,a_t^n)$是引入了指定Action的Cumulated Reward Expect，所以$V^{\pi_\theta}(s_t^n)$其实是$Q^{\pi_\theta}(s_t^n,a_t^n)$的期望值，所以使用$V^{\pi_\theta}(s_t^n)$代替Baseline $b$。$Q^{\pi_\theta}(s_t^n,a_t^n)-V^{\pi_\theta}(s_t^n)$就会变成一个可正可负的值。
   
     <img src="/Users/liuyuyang/image-20200826204751314.png" alt="image-20200826204751314" style="zoom: 50%;" />
     
   - 基于上述公式，就可以进行代码实现，但是需要两个神经网络分别对$Q,V$进行估测，误差也会更大。那么使用一个网络对$Q,V$进行估计，就需要进行一定的变换。实际上在State $s$采取Action $a$的Reward $r_t^n$也是一个Random Variable，是不确定的。例如在玩游戏时，希望使用一个技能杀死敌人，技能放出后究竟是否能杀死敌人，在决定放技能时还是不确定的。因此有$Q^{\pi}(s_t^n,a_t^n)=E[r_t^n+V^{\pi}(s_{t+1}^n)]$，然后去掉期望符号，近似的认为等式仍然成立。则可以得到$Q^{\pi}(s_t^n,a_t^n)-V^{\pi}(s_t^n)=r_t^n+V^{\pi}(s_{t+1}^n)-V^{\pi}(s_t^n)$，此时公式中存在一个Random Variable $r_t^n$，相比于$G_t^n$， $r_t^n$更稳定一些，因为这只是一步Action的收益。
   
     <img src="/Users/liuyuyang/image-20200826205703229.png" alt="image-20200826205703229" style="zoom: 50%;" />
     
   - 因为$r_t^n+V^{\pi}(s_{t+1}^n)-V^{\pi}(s_t^n)$被称为Advantage Function，又是结合了Policy Gradient和Q-Learning，所以这种技术被称为Advantage Actor-Critic。
   
     <img src="/Users/liuyuyang/image-20200826211346264.png" alt="image-20200826211346264" style="zoom:50%;" />
     
   - Advantage Actor-Critic的训练技巧
   
     - Tip 1：Advantage Actor-Critic需要做两件事情，第一件事是输入一个State $s$，输出一个scalar，用来估计$V^{\pi}(s)$；第二件事是使用NN学习一个Actor $\pi(s)$，输入一个State $s$，输出一个Action Distribution。两个网络$\pi(s)$和$V^{\pi}(s)$的前几层共享参数（绿色），先把输入转换成一些High-level的信息，然后在分别处理
     - Exploration的过程仍然是很重要的，因此对$\pi(s)$输出的Action Distribution做出一些限制，要求其信息熵不能太小，即不同的Action被执行的几率尽可能平均一点，有利于进行更多的探索。
   
       
   


#### <span name="2">Asynchronous Advantage Actor-Critic（A3C）</span>

1. <span name="2.1">Policy-based Approach三步走</span>

   - Policy-based Ap

     
     
   
2. <span name="2.2">Step 1：Neural Network  as Actor</span>

   - 使用Neural Network定义F

     
   
3. <span name="2.3">Step 2：Goodness of Actor</span>

   - 已知Actor $\pi_{\theta}(s)$，其中$\theta$为网络

     
   
4. <span name="2.4">Step 3：Pick the best Actor</span>

   - Policy Gradient的具

     


#### <span name="3">3.Value-based Approach（Learning a Critic）</span>
1. <span name="3.1">Critic的定义（State Value Function）</span>

   - Critic的定义是：不

     
   
2. <span name="3.2">Estimating Critic（State Value Function）</span>

   - Monte-Carlo based app

     

3. <span name="3.3">Critic的定义（State-action Value Function）</span>

   - State-action Va

   



#### <span name="4">4.Actor-Critic</span>

1. <span name="4.1">A3C（Asynchronous Advantage Actor-Critic）</span>

   - 之前的Actor都是根据Rew
   
     
   

#### <span name="5">5.Inverse Reinforence Learning</span>

1. <span name="5.1">Imitation Learning</span>

   - Inverse Reinforence L

     
