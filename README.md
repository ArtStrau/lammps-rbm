# lammps-rbm



This repository analyzes **fully overdamped rotational Brownian motion (RBM)** in LAMMPS.



It includes:

* LAMMPS input scripts that generate angle time series with the existing integrator and with an **improved geometric integrator** proposed as a future LAMMPS extension, following our work (see Reference).
* Python/Jupyter scripts to clean raw output and reconstruct probability distributions.
* Comparison of numerical distributions with **analytic results** from the reference.



Reference:

Felix Höfling \& Arthur V. Straube, [*Phys. Rev. Research* **7**, 043034](https://doi.org/10.1103/wzdn-29p4) (2025)  
\[*open access*, doi:[10.1103/wzdn-29p4](https://doi.org/10.1103/wzdn-29p4)]



The initial focus is on **spherical particles**.

