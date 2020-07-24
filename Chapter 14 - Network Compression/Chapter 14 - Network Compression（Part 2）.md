# Chapter 13 - Attck and Defense（Part 2）

[1.Attacks on Image](#1)

​		[1.1 One Pixel Attack基本思想](#1.1)

​		[1.2 One Pexel Attack的求解](#1.2)

[2.Attacks on Audio](#2)

​		[2.1 Attacks on ASR](#2.1)

​		[2.2 Attacks on ASV](#2.2)

​		[2.3 Hidden Voice Attack](#2.3)



#### <span name="1">1.Attacks on Image</span>

1. <span name="1.1">One Pixel Attack基本思想</span>

   - One Pexel Attack指的是改动图像中的一个像素点，从而扰乱图像识别系统的攻击

   - One Pexel Attack的数学定义：

      - $x$：n-dim的输入，$x=(x_1,x_2,···,x_n)$
      - $f$：表示Image Classifier（model的output过softmax的值）
      - $f_t(x)$：给定input $x$，模型认为$x$属于类别$t$的概率
      - $e(x)$：给原始输入$x$增加的噪声（additive adversarial perturbation）
      - One Pexel Attack on untargeted attack表示为：$e(x)^*=arg \ \min\limits_{e(x)}f_t(x+e(x))，subject\ to \ ||e(x)_0=1||，其中t表示x的原始类别$
      - One Pexel Attack on targeted attack表示为：$e(x)^*=arg \ \min\limits_{e(x)}f_{adv}(x+e(x))，subject\ to \ ||e(x)_0=1||，其中adv表示希望被判别成的类别$

   - Normal Attack v.s.  One Pixel Attack：下图中了两个$16 \times 16$的数字方格就是两种攻击方式对应的噪声。目标函数指的是最大化预判为指定标签的概率，约束指的是图片的改动不能过大。One Pixel Attack的方格中只有一个数值不为0，其约束为噪声的零范数等于1。

      <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724155219691.png" alt="image-20200724155219691" style="zoom:50%;" />

   - Example：https://arxiv.org/pdf/1710.08864.pdf

      ![image-20200724155919452](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724155919452.png)

      

2. <span name="1.2">One Pexel Attack的求解</span>

   - 第一种求解方法是暴力穷举。假设图片为$224 \times 224=50176$个像素点，穷举过程需要很长时间。

   - 实际上我们并不需要所谓的最好的perturbation，即可以满足数学定义中最大化目标概率的$e(x)^*$。在实际应用中并不需要，需要的是能够攻击成功就可以了。比如上图中原本是Cup的图片，识别为Soup Bowl的概率是16.74%。我们并不需要能够让这个概率称为100%的图片，只需要大于16.74%即可。

   - 第二种求解方法是<font color="red">Differential Evolution</font>（By Pablormier - https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=62208829）。

     - Differential Evolution的核心思想是在每次迭代期间，根据当前人口（父母）生成另一组候选解决方案（子项）。 然后将孩子与相应的父母进行比较，如果他们比父母更适合（拥有更高的健身价值），则存活下来。 以这种方式，仅比较父母和他的孩子，就可以同时实现保持多样性和提高健身价值的目标。（https://arxiv.org/pdf/1710.08864.pdf）

       <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724163841648.png" alt="image-20200724163841648" style="zoom:33%;" />

     - Differential Evolution的优点：①很大概率可以找到Global Optima；②与FGSM相比，DE不需要计算梯度，所以对攻击对象model的细节信息不需要知道，需要目标系统的信息会更少。（https://arxiv.org/pdf/1710.08864.pdf）

     - DE在One Pixel Attack上的应用：构建（X，Y，R，G，B）向量作为DE的输入

       



#### <span name="2">2.Attacks on Audio</span>

1. <span name="2.1">Attacks on ASR</span>

   - ASR指的是Auto Speech Recognization，声音可以被看做一张图片，同样给其增加一个噪声，就可以完成攻击过程（https://nicholas.carlini.com/code/audio_adversarial_examples/）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724165417534.png" alt="image-20200724165417534" style="zoom:50%;" />

     

2. <span name="2.2">Attacks on ASV</span>

   - ASV指的是Auto Speaker Verification，其攻击方法类似（https://arxiv.org/pdf/1911.01840.pdf）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724165517807.png" alt="image-20200724165517807" style="zoom:33%;" />

   - 一个典型的案例是Wake Up Words（[http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/Introduction%20(v9).pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/Introduction (v9).pdf)）

     

3. <span name="2.3">Hidden Voice Attack</span>

   - Hidden Voice Attack指的是将一些指令音讯转换成不被人们注意的声音讯号，当期播放时不会引起人们的察觉，但是还能够让一些设备执行该指令。

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724170220176.png" alt="image-20200724170220176" style="zoom:50%;" />

   - 在Speech Recognization时，不能直接吧音讯直接送入模型，而是经过一定的Signal Preprocessing。Hidden Voice Attack攻击的就是Signal Preprocessing过程。（https://arxiv.org/pdf/1904.05734.pdf）

     <img src="/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724170453141.png" alt="image-20200724170453141" style="zoom: 50%;" />

   - Hidden Voice Attack的实现方法主要有Time Domain Inversion（TDI）、Random Phase Generation（RPG）、High Frequency Addition（HFA）、Time Scaling（TS）

     ![image-20200724170746956](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724170746956.png)

     - TDI：利用mFFT多对一的性质，不同的声音可能对应的是同一种mFFT，即对于人类是不同的声音，机器听到的确实相同的

       ![image-20200724170951136](/Users/liuyuyang/GitHubLocalRepositories/LHY_Machine_Learning/Chapter 13 - Attack and Defense/image-20200724170951136.png)

     - RPG：对于FFT过程，返回的是$magnitude=\sqrt{a^2+b^2}$，其中$a+bi$代表一段信号。RPG的想法是修改$a$和$b$的值，却保持$\sqrt{a^2+b^2}$的值不变

     - HFA：在low-pass filter过程会将一些人类声音达不到的高频声音过滤掉，因此可以在音频中加入一些高频信号，让人们听不动，但是经过low-pass filter后却与人类声音差不多

     - TS：利用时域的压缩，加快时间的方式转换音频。但是要保证压缩后音频的sample date仍然有效

       


