# StructuredGaussianMixtures.jl

### Expectation Maximization for Gaussian Mixture Models with Low-Rank Plus Diagonal Covariance  

This package implements the Expectation-Maximization (EM) algorithm for fitting Gaussian Mixture Models (GMMs) under the assumption that covariance matrices are **low-rank plus diagonal**. 
This structure is useful as a form of regularization in high-dimensionsal settings. 
It is particularly useful in high-dimensional settings where the number of features is large compared to the number of data points. 
By constraining the covariance matrices in this manner, we achieve a balance between flexibility and computational efficiency while avoiding overfitting.  

We consider a dataset represented as $X \in \mathbb{R}^{n \times m}$ with **$n$** data points and **$m$** features, potentially with $m \gg n$. A Gaussian Mixture Model with $K$ components over elements $x^{(i)} \in \mathbb{R}^m$ is defined as:  
$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)\text{,} 
$$  
with non-negative mixture weights $\pi_k$ that sum to one. Each Gaussian component has mean $\mu_k \in \mathbb{R}^m$ and covariance matrix of the form:  
$$
\Sigma_k = \text{diag}(d_k) + L_k L_k^T
$$ 
where:  
- $d_k \in \mathbb{R}^m$ is the diagonal vector capturing independent feature noise,  
- $L_k \in \mathbb{R}^{m \times k}$ (with rank $k \ll m$) captures the dominant covariance structure via a low-rank factor.  

The EM algorithm for fitting GMMs alternates between an **E**xpectation and a **M**aximization:  

- **E-step:** Compute responsibilities based on current parameters:  
  $$
  \gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
  $$
  Computing responsibilities requires inverting $\Sigma_k$. 
  Since this can be prohibitively expensive in large dimensions, we leverage the structure of the covariance matrices to calculate the responsibilities using an $\mathcal{O}(k^3)$ matrix inversion rather than a full $\mathcal{O}(m^3)$ inversion.
- **M-step:** Update parameters by maximizing expected complete-data log-likelihood. 
Since updating $\Sigma_k$ directly is intractable in high dimensions, an **inner EM loop** is performed using **Factor Analysis**, where $L_k$ and $D_k$ are iteratively estimated to approximate the full covariance structure.  

This low-rank plus diagonal structure is particularly advantageous in scenarios such as **text modeling, gene expression analysis, and compressed sensing**, where **\( m \gg n \)** leads to singular or poorly conditioned full covariance estimates. Our package leverages efficient matrix decompositions and batched computations to ensure scalability, making it well-suited for large-scale, high-dimensional datasets.  
