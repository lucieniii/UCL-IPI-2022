## COMP0114 Inverse Problems in Imaging. Literature Study

### 1. Critical Analysis

#### Q1

As of March 2022, this paper has been cited 8463 times according to [Google Scholar](https://scholar.google.com/scholar?as_q=Stable+Signal+Recovery+from+Incomplete+and+Inaccurate+Measurements&as_epq=&as_oq=&as_eq=&as_occt=any&as_sauthors=&as_publication=&as_ylo=&as_yhi=&hl=en&as_sdt=0%2C5). This is a very high number of citations. According to the papers cited [1], [1] has had a profound impact in several areas, including compressive sensing, signal processing, and mathematics.

#### Q2

One of the papers cited [1] is Candès and Wakin's work "[An Introduction To Compressive Sampling](https://ieeexplore.ieee.org/abstract/document/4472240)", which has 7059 citations until March 2022. The paper provides a clear and accessible introduction to the concept of compressive sampling (CS), which claims that one can use far fewer samples to recover signals than traditional methods limited by Shannon's Law, and [1] has provided key mathematical ideas to the CS method. The paper's impact includes popularising the concept of CS and arousing more researchers' attention. Nowadays, compressive sampling has been used in a wide range of real-world applications, including medical imaging, wireless communications, remote sensing, etc.

#### Q3

There are some constraints to reconstructing a signal $f$ from a discrete set of samples.

First, we assume the signal $f \in \R^m $ is a sparse signal which $T_0=\{t:f(t)\neq 0\}$ have small cardinality. $T_0$ is a set of indices where the corresponding value in $f$ is not zero.

Second, all we know is a compressed form of $f$ which are $n$ linear measurement of $f$ :
$$
y_k=\langle f, a_k\rangle \ k=1,\dots , n\ \ \ \text{or}\ \ \ y=Af
$$
.

Matrix $A\in \R^{n\times m}$ is called a "measurement matrix" which obeys a $uniform\ uncertainty \ pricinple$, to be more specific, obeys a $S$-restricted isometry hypothesis. Let $T\subset\{1,\dots, m\}$ be the indices set of $A$, then $A_T$ is a $n \times |T|$ submatrix of $A$ which contains columns corresponding to the indices in $T$. If for all subsets $T$ with $|T|\le S$ and coefficient sequences $(c_j)_{j\in T}$, $\delta_S$ is the smallest value satisfies that 
$$
(1-\delta_S)||c||^2_{l_2}\le||A_Tc||^2_{l_2}\le(1+\delta_S)||c||^2_{l_2}
$$
, then $A$ obeys a $S$-restricted isometry hypothesis.

If $S$ satisfies $\delta_S+\delta_{2S}+\delta_{3S}\lt1$, then we can recover $f$ exactly by solving the problem 
$$
\text{min}\ ||f||_{l_1} \ \ \text{subject to}\ \ Af=y
$$
for any sparse signal $f$ obeying $|T_0|\le S$.

However, if the measurement is noisy ($y=Af+e$), solving the problem above is not possible to recover the signal exactly.

### 2. Numerical Experiments

#### Construct measurement matrix $A$

I use the following steps to construct the matrix $A$:

+ Construct a matrix $B$ which all the elements in it obey normal distribution;
+ Use `scipy.linalg.orth` to construct two orthonormal systems $U$ and $V$ based on $B$ and $B^{\text{T}}$;
+ Since the measurement matrix $A$ should behave like an orthonormal system, I choose a matrix $W$ which have the value $1.8$ on the main diagonal (decided by experiment);
+ Calculate $A = UWV^{\text{T}}$.

Below is the actual code.

```python
B = np.random.randn(N, M)
U = scipy.linalg.orth(B)
V = scipy.linalg.orth(B.T)
W = np.diag(w * np.ones((N,)))
A = U @ W @ V.T 
```

#### Basis Pursuit

According to the thesis, first, create a signal $f_0$ which is $1024 \times 1$ and only have $50$ non-zero elements in it. At the same time, the non-zero elements are all $-1$ or $1$. Then compress $f_0$ with the measurement matrix $A$ without noises to $y$, which is a $300 \times 1$ vector. Below is the plot of $f_0$ and $y$.

|                            $f_0$                             |                             $y$                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/2-1.png" alt="2-1" style="zoom: 67%;" /> | <img src="/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/2-2.png" alt="2-2" style="zoom: 67%;" /> |

Now we try to recover $f_0$ from $y$ and $A$. Since noises are not considered, the recovered signal $f$ is the solution to the convex problem
$$
\text{min}\ ||f||_{l_1} \ \ \text{subject to}\ \ Af=y
$$
.

Below is the plot of $f_0$ and recovered signal $f$. Seems to be an exact recovery. Value of $||f_0-f||_{l_2}$ is only $6.31\times 10^{-10}$.

|                            $f_0$                             |                             $f$                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/2-1.png" alt="2-1" style="zoom: 67%;" /> | <img src="/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/2-3.png" alt="2-3" style="zoom: 67%;" /> |

I use `cvxpy` library to solve the convex problem, below is the code.

```python
def compress(x, A):
    return A @ x

def basis_pursuit(f, A):    
    y = compress(f, A)
    x = cp.Variable(M) # M = 1024
    prob = cp.Problem(cp.Minimize(cp.atoms.norm1(x)), 
                    [A @ x == y])
    prob.solve()
    return y, x.value
```

#### Basis Pursuit De-Noising

In this problem, the vector $y$ is compressed from $f_0$ with noise: $y=Af+e$. $e$ is a $300\times 1$ vector which elements in it obey the Gaussian distribution with $\mu=0$, and the $\sigma$ of it varies. I test the $\sigma$s in the set $\{0.01, 0.02, 0.05, 0.1, 0.2, 0.5\}$. Now the convex problem becomes to
$$
\text{min}\ ||f||_{l_1} \ \ \text{subject to}\ \ ||Af-y||_2\le \epsilon
$$
where $\epsilon^2=\sigma^2(n+\lambda\sqrt{2n})$ according to the thesis. $n$ is the length of $y$, which is $300$ in our case. $\lambda=2$ according to the suggestion in the thesis.

The resulting plots can be seen in the appendix. Below is the resulting table.

|   $\sigma$    |  $0.01$  |  $0.02$  |  $0.05$  |  $0.1$   |  $0.2$   |  $0.5$   |
| :-----------: | :------: | :------: | :------: | :------: | :------: | :------: |
|  $\epsilon$   | $0.1868$ | $0.3736$ | $0.9341$ | $1.8681$ | $3.7363$ | $9.3406$ |
| $||f-f_0||_2$ |  $0.31$  |  $0.60$  |  $1.43$  |  $2.73$  |  $4.82$  |  $6.69$  |

I didn't take an average result, but it still seems close to the result in the thesis.

Below is the code for solving the Basis Pursuit De-Noising problem.

````python
def compress_with_noise(x, A, e):
    return A @ x + e
def basis_pursuit_de_noising(f, A, e, eps):
    # eps: epsilon
    y = compress_with_noise(f, A, e)
    x = cp.Variable(M) # M = 1024
    prob = cp.Problem(cp.Minimize(cp.atoms.norm1(x)), 
                    [cp.atoms.norm(A @ x - y, p=2) <= eps])
    prob.solve()
    return y, x.value
````

#### Change the singular spectrum of $A$

I use $\text{SVD}$ to change the singular spectrum of $A$ with the following steps:

+ $\text{SVD}(A)=UWV^{\text{T}}$;
+ Change the value of all the elements $w_{ij}$ in $W$ by $w_{ij}=e^{\frac{-w_{ij}}{100}}$;
+ Recalculate $A=UWV^{\text{T}}$.

The actual code is as below.

```python
u, s, vh = np.linalg.svd(A, full_matrices=False)
s = np.exp(-s / 100)
As = (u * s) @ vh
```

The recovery result is as the plot below. (With $\sigma=0.02$)

<img src="/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/2-4.png" alt="2-4" style="zoom: 40%;" />

The results seem still stable. It is possible because after changing the singular spectrum of $A$, it still obeys a $restricted\ isometry\ hypothesis$. The values $w_{ii}$ change from $1.8$ to $0.98$, and it is not a big change. So the signal after compressing can still keep enough information to recover, especially for the $Basis\ Pursuit$ problem.

### 3. Appendix: Results of Basis Pursuit De-Noising

![appd](/Users/lucien/Documents/UCL-CGVI/IPI/UCL-IPI-2022/ex3/report/plots/appd.png)
