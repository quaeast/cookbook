dropout在线性模型中和L2正则项等价

L2正则项和

# Self-Attention

## Attention的计算方法

![](./img/self-attention.png)

$k,q,v$矩阵计算方法：

$$
q^i=W^qa^i\\
k^i=W^ka^i\\
v^i=W^va^i
$$

注意力$\alpha$计算方法：

$$
\alpha_{i,j}=q^i\cdot k^j
$$

$\alpha'$可以有多种计算方式：

$$
\alpha'=softmax(\alpha)\\
\alpha'=RELU(\alpha)\\
$$

**多头注意力**

类似于三维的卷积核，将$W_q,W_k,W_v$矩阵增加一个维度，就是多头注意力机制。

![multi-head_self_attention](./img/multi-head_self_attention.png)

## 从池化的角度理解

对于给定数据：

$$
(x_i,y_i),i=1,2,...,n
$$

平均池化是：

$$
f(x)=\frac{1}{n}\sum_iy_i
$$

Attention池化，就是根据当前数据和所有其他数据根据$x$计算一个权重，对$y$进行加权平均。对应上图其中$x$对应query，$x_j$对应key，$y_i$对应value。：

$$
f(x)=\sum_i\alpha(x,x_i)y_i
$$

在上世纪六十年代就存在了类似于Attention的Nadaraya-Waston核回归。$K()$是一个函数，用来衡量距离：

$$
f(x)=\sum^n_{i=1}\frac{K(x-x_i)}{\sum^n_{j=1}K(x-x_j)}y_i
$$

$K()$可以有很多形式，如果使用高斯核：

$$
K(u)=\frac{1}{\sqrt{2\pi}}exp(-\frac{u^2}{2})
$$

将$K(u)$带入到$f(x)$中可以得到：

$$
f(x)=\sum^n_{i=1}softmax(-\frac{1}{2}(x-x_i)^2)y_i
$$

加入可学习的参数$w$即可得到：

$$
f(x)=\sum^n_{i=1}softmax(-\frac{1}{2}((x-x_i)w)^2)y_i
$$

# 带权重的方法

$$
n = f(l,w)\\
c = g(n,e)\\
c = g(f(l,w),e)\\
c = h(l,w,e)\\
C_{b,3} = E_{b,768}(L_{3,1356}W_{1356,768})^T\\
loss = CE(True_{b,3},softmax(C_{b,3}))\\
求：
\frac{\partial loss}{\partial w} = \frac{\partial loss}{\partial c}\frac{\partial c}{\partial n}\frac{\partial n}{\partial w}\\
$$

$$
softmax(C_{b,3})_{i,j} = p(y=j|C_i) = \frac{e^{C_i^TW_j}}{\sum^K_{k=1}e^{C_i^TW_k}}\\
CE(True, Pred) = -\frac{1}{b}\sum_{i=1...b}\sum_{j=1,2,3}True_{i,j}logPred_{i,j}\\
$$

# 对比学习

$$
a
$$

