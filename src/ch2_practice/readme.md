# CH2 PRACTICE

## 分别使用左右扰动模型计算：$\frac{\partial R^{-1}p}{\partial R}$

也即计算$\frac{\partial R^{-1} p}{\partial \phi}$

### 左扰动

$$
\begin{aligned}
\frac{\partial R^{-1} p}{\partial \phi}&=\lim_{x \rightarrow 0}\frac{(\exp(\phi^\wedge)R)^{-1} p - R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{R^{-1}\exp(\phi^\wedge)^{-1} p - R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{R^{-1}\exp(-\phi^\wedge) p - R^{-1} p}{\phi}\\
一阶泰勒近似&\approx\lim_{x \rightarrow 0}\frac{R^{-1}(I-\phi^\wedge) p - R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{R^{-1}(-\phi^\wedge) p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{R^{-1}(p^\wedge) \phi}{\phi}\\
&=R^{-1}(p^\wedge)
\end{aligned}
$$

### 右扰动

$$
\begin{aligned}
\frac{\partial R^{-1} p}{\partial \phi}&=\lim_{x \rightarrow 0}\frac{(R\exp(\phi^\wedge))^{-1} p - R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{\exp(-\phi^\wedge)R^{-1} p - R^{-1} p}{\phi}\\
一阶泰勒近似&\approx\lim_{x \rightarrow 0}\frac{(I-\phi^\wedge)R^{-1} p - R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{(-\phi^\wedge)R^{-1} p}{\phi}\\
&=\lim_{x \rightarrow 0}\frac{(R^{-1}p)^\wedge \phi}{\phi}\\
&=(R^{-1}p)^\wedge
\end{aligned}
$$

## 分别使用左右扰动模型计算：$\frac{\partial R_1R_2^{-1}}{\partial R_2}$

也即计算$\frac{\partial\ln(R_1R_2^{-1})}{\phi}$

### 左扰动

$$
\begin{aligned}
\frac{\partial\ln(R_1R_2^{-1})}{\phi} &= \lim_{\phi \rightarrow 0}\frac{\ln(R_1(\exp(\phi^\wedge)R_2)^{-1})-\ln(R_1R_2^{-1})}{\phi}\\
&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1R_2^{-1}\exp(-\phi^\wedge))-\ln(R_1R_2^{-1})}{\phi}\\
(BCH一阶近似)&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1R_2^{-1}) + J_r^{-1}(R_1R_2^{-1})(-\phi) - \ln(R_1R_2^{-1})}{\phi}\\
&=\lim_{\phi \rightarrow 0}\frac{-J_r^{-1}(R_1R_2^{-1})\phi}{\phi}\\
&=-J_r^{-1}(R_1R_2^{-1})
\end{aligned}
$$

### 右扰动

$$
\begin{aligned}
\frac{\partial\ln(R_1R_2^{-1})}{\phi} &= \lim_{\phi \rightarrow 0}\frac{\ln(R_1(R_2\exp(\phi^\wedge))^{-1})-\ln(R_1R_2^{-1})}{\phi}\\
&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1\exp(-\phi^\wedge)R_2^{-1})-\ln(R_1R_2^{-1})}{\phi}\\
&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1R_2^TR_2\exp(-\phi^\wedge)R_2^T)-\ln(R_1R_2^{-1})}{\phi}\\
(伴随性质)&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1R_2^T\exp((-R_2\phi)^\wedge))-\ln(R_1R_2^{-1})}{\phi}\\
(BCH一阶近似)&=\lim_{\phi \rightarrow 0}\frac{\ln(R_1R_2^{-1}) + J_r^{-1}(R_1R_2^{-1})(-R_2\phi) - \ln(R_1R_2^{-1})}{\phi}\\
&=-J_r^{-1}(R_1R_2^{-1})R_2
\end{aligned}
$$

## 将实践环节中的运动学修改成带有一定角速度的平抛运动。车辆受固定的Z轴角速度影响，具有一定的初始水平速度，同时受−Z向的重力加速度影响。请修改程序，给出动画演示

[代码](motion.cc)
[动画演示](parabola.mp4)

## 自行寻找相关材料，说明高斯牛顿法和Levenberg-Marquardt在处理非线性迭代时的差异

高斯-牛顿GN法和列文伯格-马夸尔特LM法都是非线性最小二乘问题的迭代优化方法，LM法在一定程度上修正了GN法存在算法不收敛以及收敛到局部最优问题。一般认为LM法比GN法更为鲁棒，对初始参数值的选择更为稳健，更能处理非线性问题，但LM需要更多的计算资源，收敛速度可能比GN更慢，也被称为 阻尼牛顿法（Damped Newton Method） 特别是当问题的维度非常大时。实际问题中，通常选择高斯牛顿法或列文伯格-马夸尔特方法中的一种作为梯度下降策略。当问题性质比较好的时候，用高斯牛顿GN法。如果问题接近病态，则用列文伯格-马夸尔特LM方法。
