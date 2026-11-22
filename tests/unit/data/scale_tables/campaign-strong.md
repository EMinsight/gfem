| Nodes | Ranks x threads | Solver | Mesh | DOF | DOF per rank | Owned per rank, min-max | DOF per node | `tol` | Repeats | Iterations, ports 1/2 | `gfem.solve` median, s | Range, s | Coarse apply median, s | Efficiency against 1 node | Largest node, GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28 x 4 | `bddc_gmres` | m_void | 2,800,000 | 100 k | 99-101 k | 2.8 M | 1e-08 | 2 | 40/41 | 110 | 100-120 | 28 | base | 50 |
| 2 | 56 x 4 | `bddc_gmres` | m_void | 2,800,000 | 50 k | 49-51 k | 1.4 M | 1e-08 | 1 | 40/41 | 50 | - | 12 | 1.10 | 50 |
| 2 | 56 x 4 | `bddc_gmres` | m_void | 2,800,000 | 50 k | - | 1.4 M | - | 1 | - | TIMEOUT in `gmres` | - | - | - | - |

`gfem.solve` is the slowest rank's whole call: spaces, ports, preconditioner or factorization, and GMRES, with the instrumentation's true residual and one extra apply and matvec inside it. The mesh read and the partition come before it. Efficiency is the median's time at the base times its nodes, over time here times these nodes, same solver and mesh. `tol` is `SolverOptions.tolerance`, GMRES's relative preconditioned residual; it does not apply to `direct`. A failed run is its own row, with Slurm's state and the phase its trace left open; its DOF are the mesh's.
