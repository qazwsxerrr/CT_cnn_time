# SINGLE-ANGLE RADON SAMPLES

Guangxi University

November 14, 2025

Suppose that  $\operatorname{supp}(\phi) \subseteq [N_1, M_1] \times [N_2, M_2]$ . Additionally,

$$
E = \left[ a _ {1}, b _ {1} \right] \times \left[ a _ {2}, b _ {2} \right] \subseteq \mathbb {R} ^ {2}. \text {C o r r e s p o n d i n g l y , d e f i n e}
$$

$$
V _ {E} (\phi) := \left\{f \in V \left(\phi , \mathbb {Z} ^ {2}\right) \text {s u c h t h a t} \operatorname {s u p p} (f) \subseteq E \right\}. \tag {1.1}
$$

Then there exists uniquely a sequence  $\{c_{\pmb{k}}\}_{\pmb{k} \in E^{+}}$  such that  $f$  can be expressed as

$$
f = \sum_ {\boldsymbol {k} \in E ^ {+}} c _ {\boldsymbol {k}} \phi (\cdot - \boldsymbol {k}) \tag {1.2}
$$

where

$$
E ^ {+} = \left\{\left[ \lceil a _ {1} - M _ {1} \rceil , \lfloor b _ {1} - N _ {1} \rfloor \right] \times \left[ \lceil a _ {2} - M _ {2} \rceil , \lfloor b _ {2} - N _ {2} \rfloor \right] \right\} \cap \mathbb {Z} ^ {2}.
$$

Assume that  $\alpha = \frac{\beta}{\|\beta\|_2}$  where  $\mathbf{0} \neq \boldsymbol{\beta} \in \mathbb{Z}^2$ .

$$
\mathcal {R} _ {\boldsymbol {\alpha}} f = \sum_ {\boldsymbol {k} \in E ^ {+}} c _ {\boldsymbol {k}} \mathcal {R} _ {\boldsymbol {\alpha}} \phi (\cdot - \boldsymbol {\alpha} \cdot \boldsymbol {k}). \tag {2.1}
$$

And the Fourier transform at  $\xi \in \mathbb{R}$  of  $\mathcal{R}_{\alpha}f$  is

$$
\begin{array}{l} \widehat {\mathcal {R} _ {\alpha} f} (\xi) = \widehat {\phi} (\xi \alpha) \sum c _ {\pmb {k}} e ^ {- \mathbf {i} \alpha \cdot \pmb {k} \xi} \\ = \widehat {\phi} \left(\alpha_ {1} \xi , \alpha_ {2} \xi\right) \sum_ {\mathbf {k} \in E ^ {+}} c _ {\mathbf {k}} e ^ {- \mathbf {i} \left(\alpha_ {1} k _ {1} + \alpha_ {2} k _ {2}\right) \xi}. \tag {2.2} \\ \end{array}
$$

# Fourier transform at high frequencies

Define  $\kappa_{\mathrm{max}} = \max \left\{\pmb {\beta}\cdot \pmb {k}:\pmb {k}\in E^{+}\right\}$ ,  $\kappa_{\mathrm{min}} = \min \left\{\pmb {\beta}\cdot \pmb {k}:\pmb {k}\in E^{+}\right\}$  and  $N = \kappa_{\mathrm{max}} - \kappa_{\mathrm{min}} + 1$ . We choose  $N$  sampling points:

$$
\xi_ {j} = \frac {2 \pi j}{N} \| \boldsymbol {\beta} \| _ {2}, \quad j = 0, 1, \dots , N - 1, \tag {3.1}
$$

Then from (2.2), it can be derived that

$$
\widehat {\mathcal {R} _ {\boldsymbol {\alpha}}} f (\xi_ {j}) = \widehat {\phi} (\xi_ {j} \boldsymbol {\alpha}) \sum_ {n = 0} ^ {N - 1} d _ {n} e ^ {- \mathbf {i} (n + \kappa_ {\min }) \frac {2 \pi j}{N}}, \quad j = 0, 1, \dots , N - 1, \quad (3. 2)
$$

where

$$
d _ {n} = \left\{ \begin{array}{l l} c _ {\boldsymbol {k} _ {j}}, & \text {i f} \ell = \boldsymbol {\beta} \cdot \boldsymbol {k} _ {j} \text {f o r a c e r t a i n} j \in \{1, \dots , \# E ^ {+} \}, \\ 0, & \text {o t h e r w i s e .} \end{array} \right. \tag {3.3}
$$

Let  $F(\xi_j) = e^{\mathbf{i}\kappa_{\min} \frac{2\pi j}{N}} \widehat{\mathcal{R}_{\alpha}f}(\xi_j), \quad j = 0,1,\ldots,N-1$ . Then we can rewrite (3.2) as

$$
F (\xi_ {j}) = \widehat {\phi} (\xi_ {j} \boldsymbol {\alpha}) \sum_ {n = 0} ^ {N - 1} d _ {n} e ^ {- \mathrm {i} n \frac {2 \pi j}{N}}, \quad j = 0, 1, \dots , N - 1, \tag {3.4}
$$

Note that  $N$  sampling points:

$$
\xi_ {j} = \frac {2 \pi j}{N} \| \boldsymbol {\beta} \| _ {2}, \quad j = 0, 1, \dots , N - 1.
$$

The system of equations (3.4) can be transformed into a matrix form:

$$
\boldsymbol {F} = \Phi G \boldsymbol {d},
$$

Where  $G$  is the FFT matrix.

$$
\left[ \begin{array}{c} F (\xi_ {0}) \\ F (\xi_ {1}) \\ F (\xi_ {2}) \\ \vdots \\ F (\xi_ {N - 1}) \end{array} \right] = \left[ \begin{array}{c c c c c} \widehat {\phi} (\xi_ {0} \boldsymbol {\alpha}) & 0 & 0 & \dots & 0 \\ 0 & \widehat {\phi} (\xi_ {1} \boldsymbol {\alpha}) & 0 & \dots & 0 \\ 0 & 0 & \widehat {\phi} (\xi_ {2} \boldsymbol {\alpha}) & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & \widehat {\phi} (\xi_ {N - 1} \boldsymbol {\alpha}) \end{array} \right] \times
$$

$$
\left[ \begin{array}{c c c c c} 1 & 1 & 1 & \dots & 1 \\ 1 & e ^ {- \mathbf {i} \frac {2 \pi}{N}} & e ^ {- \mathbf {i} \frac {4 \pi}{N}} & \dots & e ^ {- \mathbf {i} \frac {2 \pi (N - 1)}{N}} \\ 1 & e ^ {- \mathbf {i} \frac {4 \pi}{N}} & e ^ {- \mathbf {i} \frac {8 \pi}{N}} & \dots & e ^ {- \mathbf {i} \frac {4 \pi (N - 1)}{N}} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & e ^ {- \mathbf {i} \frac {2 \pi (N - 1)}{N}} & e ^ {- \mathbf {i} \frac {4 \pi (N - 1)}{N}} & \dots & e ^ {- \mathbf {i} \frac {2 \pi (N - 1) ^ {2}}{N}} \end{array} \right] \left[ \begin{array}{c} d _ {0} \\ d _ {1} \\ d _ {2} \\ \vdots \\ d _ {N - 1} \end{array} \right]
$$

# Example 1

The  $m$ th  $(m\in \mathbb{N})$  cardinal B-spline  $B_{m}$  is defined by

$B_{m} := \overbrace{\chi_{(0,1]}}^{m\text{ copies}} \star \ldots \star \chi_{(0,1]}$ , where  $\chi_{(0,1]}$  and  $\star$  are the characteristic function of  $(0,1]$  and the convolution, respectively. Through the simple calculation we have

$$
\widehat {B _ {m}} (\xi) = e ^ {- \mathrm {i} m \xi / 2} \left[ \frac {\sin (\xi / 2)}{\xi / 2} \right] ^ {m}.
$$

Take  $\phi (x,y) = B_2(x)B_2(y)$ , then we have

$$
\begin{array}{l} \widehat {\phi} (\xi_ {j} \boldsymbol {\alpha}) = \widehat {\phi} (\frac {2 \pi j}{N} \beta) = \widehat {B _ {2}} (\frac {2 \pi j}{N} \beta_ {1}) \widehat {B _ {2}} (\frac {2 \pi j}{N} \beta_ {2}) \\ = e ^ {- \mathrm {i} \frac {2 \pi j}{N} (\beta_ {1} + \beta_ {2})} \left[ \frac {\sin (\pi j \beta_ {1} / N)}{\pi j \beta_ {1} / N} \right] ^ {2} \left[ \frac {\sin (\pi j \beta_ {2} / N)}{\pi j \beta_ {2} / N} \right] ^ {2} \\ \end{array}
$$

# Question 1

The condition number of matrix  $\Phi$  is large.  
In practical problems,  $\mathcal{R}_{\alpha}f$  often contains noise.

# 定理

设  $\mathbf{A}$  是非奇异阵，  $Ax = b\neq 0$  ，且  $A(x + \delta x) = b + \delta b$  ，则

$$
\frac {\left\| \delta \boldsymbol {x} \right\|}{\left\| \boldsymbol {x} \right\|} \leqslant \left\| \boldsymbol {A} ^ {- 1} \right\| \left\| \boldsymbol {A} \right\| \frac {\left\| \delta \boldsymbol {b} \right\|}{\left\| \boldsymbol {b} \right\|}.
$$

上式给出了解的相对误差的上界，常数项  $b$  的相对误差在解中可能放大  $\left\| A^{-1} \right\| \| A \|$  倍.

To reduce the condition number of the matrix  $\Phi$ , we can select the sampling points as follows:

$$
\xi_ {j} = \left(\frac {2 \pi j}{N} + 4 \pi \ell_ {j}\right) \| \boldsymbol {\beta} \| _ {2}, \quad j = 0, 1, \dots , N - 1,
$$

where  $\ell_j \in \mathbb{Z}^+$  for  $j = 0,1,\ldots,N-1$ . Consequently, it is necessary to perform Fourier sampling at high frequencies.

# Question 2

When  $\mathcal{R}_{\alpha}f$  contains noise, the values of the Fourier transform of  $\mathcal{R}_{\alpha}f$  at high frequencies cannot be accurately computed.

![](images/2f03c786540ef8765f53a126961246b36aa00e838e1c77eb0eb7984ffb911071.jpg)

True Values vs Estimated Values (no Noise) at Low Frequencies

![](images/50f018000e31c096b6467e9dc6a123abdc27a74c78f82217470315be7ad37ac8.jpg)

True Values vs Estimated Values (with Noise) at Low Frequencies

![](images/651d976f8af1621ba3fb1c33cd38637064214d690dc1dfda48b44c1b7aa29aae.jpg)  
True Values vs Estimated Values (no Noise) at High Frequencies

![](images/c20919fdcc31c21537fc90d59ff68d79af619f1fc379a55cd54e09e0ccb0e917.jpg)  
True Values vs Estimated Values (with Noise) at High Frequencies