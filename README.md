# model H

For simplicity, we first look at the continuum equations of motion for $(\phi, \pi_T)$ without dissipation or noise:

$$\begin{aligned}
\dot{\phi} & = - \nabla\_\mu ( \phi \, \pi\_{T,\mu} ) = - \pi\_{T,\mu}  \nabla\_\mu \phi,  \\
\dot{\pi}\_{T,i} & = - \hat{{\cal T}}\_{ij} \nabla_k \( \pi_{T,k} \pi_{T,j} + \nabla_k \phi \nabla_j \phi ) \\ 
 & = - \hat{{\cal T}}\_{ij} ( \nabla_k \left(\pi_{T,k} \pi_{T,j} \right) + \nabla\_j \phi \nabla^2 \phi ) 
\end{aligned}$$

The time derivative of the Hamiltonian is (assumed $\rho = 1$),
$$\dot{H} = \int d^3x  \[ - \dot{\phi} \nabla^2 \phi + \pi\_{T,i} \dot{\pi}\_{T,i} + V'(\phi)  \dot{\phi} \] $$


Substituting

$$\begin{aligned}
    \dot{{\cal H}} = - \int d^3x \\ \left\[ - \nabla^2\phi \\
    \vec{\pi}\_T \cdot \vec{\nabla} \phi
    %
    + \vec{\pi}\_T \cdot \left( \vec{\pi}\_T \cdot \vec{\nabla} \right) \vec{\pi}\_T
    %
    + \nabla^2\phi \\ \vec{\pi}\_T \cdot \vec{\nabla} \phi
    %
    + \vec{\nabla} \cdot \left( V(\phi) \vec{\pi}\_T \right)  \right\]. 
\end{aligned}$$

The first and third terms cancel each other. In the continuum, the
second term can be written as a divergence,

$$\begin{aligned}
    \vec{\pi}\_T \cdot \left( \vec{\pi}\_T \cdot \vec{\nabla} \right) \vec{\pi}\_T = \vec{\nabla} \cdot \left( \vec{\pi}\_T \\ \frac{\vec{\pi}\_T^2}{2} \right).
\end{aligned}$$

However, this is not guaranteed in the discretised version if we use the
divergence type term
∇<sub>*k*</sub>(*π*<sub>*T*, *k*</sub>*π*<sub>*T*, *j*</sub>). Morinishi et al suggests we use a
skew-symmetric form to take care of this:

$$\begin{aligned}
    \nabla_k \left(\pi\_{T,k} \pi\_{T,j} \right) = \frac{1}{2} \\ \nabla_k \left(\pi\_{T,k} \pi\_{T,j} \right) + \frac{1}{2} \\ \pi\_{T,k} \\ \nabla_k \pi\_{T,j}.
\end{aligned}$$

Also, in the discretised version  we expect the first and third terms
to cancel each other if we use the central difference scheme throughout,
i.e, for (∇<sub>*j*</sub>*ϕ*∇<sup>2</sup>*ϕ*) term and also the kinetic term,
(∇*ϕ*)<sup>2</sup>, in the Hamiltonian:

$$\begin{aligned}
    \nabla_j \phi &\to \frac{\phi(\vec{x}+\hat{\nu}\_j) - \phi(\vec{x} - \hat{\nu}\_j)}{2}, \nonumber \\
    %
    \nabla^2\phi &\to \sum\_\nu \frac{1}{4} \\ \left(\phi(\vec{x} + 2 \hat{\nu}) + \phi(\vec{x} - 2 \hat{\nu}) - 2 \\ \phi(\vec{x})  \right)
\end{aligned}$$

We note that the central difference scheme satisfies
∫*d*<sup>3</sup>*x*(∇*ϕ*)<sup>2</sup> →  − ∫*d*<sup>3</sup>*x* *ϕ*∇<sup>2</sup>*ϕ*
in Hamiltonian:

$$\begin{aligned}
    {\cal H} &= \frac{1}{2} \\ \sum_i \\ \left(\frac{\phi\_{i+1} - \phi\_{i-1}}{2}\right)^2 = \frac{1}{2} \\ \frac{1}{4} \sum\_{i} \left( \phi\_{i+1}^2 + \phi\_{i-1}^2 - 2 \\ \phi\_{i+1} \phi\_{i-1} \right), \nonumber \\
    %
    & = \frac{1}{2} \\ \frac{1}{4} \\ \sum_i \left( 2 \phi_i^2 - \phi_i \\ \phi\_{i+2} \\ - \phi_i \phi\_{i-2} \right) = - \frac{1}{2} \\ \sum_i \\ \phi_i \left( \frac{\phi\_{i+2} + \phi\_{i-2} - 2 \phi_i}{4} \right)
\end{aligned}$$

If we discretize the spatial derivatives we get (leaving aside the
projector and using the skew symmetric form of Morinishi et al for
∇<sub>*k*</sub>(*π*<sub>*T*, *k*</sub>*π*<sub>*T*, *i*</sub>)),

$$\begin{aligned}
    \dot{\phi} &= - \pi\_{T,\mu}(x) \\ \nabla^c\_{\mu} \phi \\
    %
    \dot{\pi}\_{T,\mu} &= - \Bigg\[ \frac{1}{2} \nabla^c\_{\nu} \left( \pi\_{T,\nu} \pi\_{T,\mu} \right) + \frac{1}{2} \\ \pi\_{T,\nu} \\ \nabla^c\_{\nu} \pi\_{T, \mu} \nonumber \\
    %
    & + \nabla^c\_\mu \phi \\ \sum\_\nu \\ \frac{ \left(\phi(\vec{x} + 2 \hat{\nu}) + \phi(\vec{x} - 2 \hat{\nu}) - 2 \\ \phi(\vec{x})  \right)}{4}  \Bigg\],
\end{aligned}$$

where
∇<sub>*μ*</sub><sup>*c*</sup>*ψ*(*x⃗*) = (*ψ*(*x⃗*+*μ̂*)−*ψ*(*x⃗*−*μ̂*))/2.
For the projector we use

$$\begin{aligned}
    P\_{\mu\nu} = \delta\_{\mu\nu} - \frac{\tilde{k}\_\mu \\ \tilde{k}\_\nu}{\tilde{k}^2},
\end{aligned}$$

with *k̃*<sub>*μ*</sub> = sin (*k*<sub>*μ*</sub>), and
*k*<sub>*μ*</sub> = (2*π*/*L*)*n̂*<sub>*k*, *μ*</sub>. The time
derivatives  can be solved
using Runge-Kutta methods.

Writing the evolution
equations as,

$$\begin{aligned}
    \dot{\phi}\_\mu = {\cal F}\_\mu (\phi\_\mu),
\end{aligned}$$

the ‘third-order’ RK scheme gives,

$$\begin{aligned}
    \phi^{n+1}\_\mu = \phi^n\_{\mu} + \Delta t \\ \left( \frac{1}{6} \\ k\_{\mu,1} + \frac{1}{6} \\ k\_{\mu,2} + \frac{2}{3} \\ k\_{\mu,3} \right),
\end{aligned}$$

where *k*<sub>*μ*, *i*</sub> are given by

$$\begin{aligned}
    k\_{\mu,1} &= {\cal F}\_\mu (\phi^n\_\mu), \\
    %
    k\_{\mu, 2} &= {\cal F}\_\mu \left( \phi^n\_\mu + \Delta t \\ k\_{\mu,1}  \right) \\
    %
    k\_{\mu,3} &= {\cal F}\_\mu \left( \phi^n\_\mu + \frac{\Delta t}{4} \\ \left( k\_{\mu,1} + k\_{\mu,2} \right)  \right).
\end{aligned}$$
