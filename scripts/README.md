# Example scripts

Key examples are `thermalize.jl` and `tau_measure.jl`. We run these on HPC with
their respective `submit_*` scripts, where submission parameters are taken from
`run_*` templates (`run_h100` is for the H100 GPU). The `tau_measure.jl` script
was used to collect data at various offsets from the critical temperture (as
defined in `src/initialize.jl`).
