using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using CUDA

include("helpers.jl")

const plans = (CUFFT.plan_fft(CuArray{FloatType}(undef,(L,L,L,3)), (1,2,3)),
               CUFFT.plan_ifft!(CuArray{ComplexType}(undef,(L,L,L,3)), (1,2,3)))

function view_tuple(u)
    if size(u, 4) == 3
        return (@view(u[:,:,:,1]),@view(u[:,:,:,2]),@view(u[:,:,:,3]))
    end

    (@view(u[:,:,:,1]),@view(u[:,:,:,2]),@view(u[:,:,:,3]),@view(u[:,:,:,4]))
end

@parallel_indices (ix,iy,iz) function poisson_scaling(πfft)
    k1 = sin(2pi * (ix-1) / L)
    k2 = sin(2pi * (iy-1) / L)
    k3 = sin(2pi * (iz-1) / L)
    k_factor = k1^2 + k2^2 + k3^2

    if k_factor > 1e-11
        k_factor = (k1 * πfft[ix,iy,iz,1] + k2 * πfft[ix,iy,iz,2] + k3 * πfft[ix,iy,iz,3]) / k_factor
        πfft[ix,iy,iz,1] -= k1 * k_factor
        πfft[ix,iy,iz,2] -= k2 * k_factor
        πfft[ix,iy,iz,3] -= k3 * k_factor
    else
        πfft[ix,iy,iz,1] = 0.0
        πfft[ix,iy,iz,2] = 0.0
        πfft[ix,iy,iz,3] = 0.0
    end

    return
end

function project(π, temp)
    temp .= plans[1] * π
    @parallel (1:L, 1:L, 1:L) poisson_scaling(temp)
    plans[2] * temp;
    π .= real.(temp)
end

@parallel function deterministic_elementary_step(
        π1, π2, π3, ϕ,
        dπ1, dπ2, dπ3, dϕ)

    ### phi update
    # π_μ ∇_μ ϕ
    @all(dϕ) = -1.0/ρ * (@all(π1) * @d_xc(ϕ) + @all(π2) * @d_yc(ϕ) + @all(π3) * @d_zc(ϕ))

    ### pi update
    # ∇_μ ϕ ∇²ϕ
    @all(dπ1) = -@d_xc(ϕ) * @d2_xyz(ϕ)
    @all(dπ2) = -@d_yc(ϕ) * @d2_xyz(ϕ)
    @all(dπ3) = -@d_zc(ϕ) * @d2_xyz(ϕ)

    # π_ν ∇_ν π_μ
    @all(dπ1) = @all(dπ1) - 0.5/ρ * (@all(π1) * @d_xc(π1) + @all(π2) * @d_yc(π1) + @all(π3) * @d_zc(π1))
    @all(dπ2) = @all(dπ2) - 0.5/ρ * (@all(π1) * @d_xc(π2) + @all(π2) * @d_yc(π2) + @all(π3) * @d_zc(π2))
    @all(dπ3) = @all(dπ3) - 0.5/ρ * (@all(π1) * @d_xc(π3) + @all(π2) * @d_yc(π3) + @all(π3) * @d_zc(π3))

    # ∇_ν π_μ π_ν
    @all(dπ1) = @all(dπ1) - 0.5/ρ * (@prd_d_xc(π1,π1) + @prd_d_yc(π1,π2) + @prd_d_zc(π1,π3))
    @all(dπ2) = @all(dπ2) - 0.5/ρ * (@prd_d_xc(π2,π1) + @prd_d_yc(π2,π2) + @prd_d_zc(π2,π3))
    @all(dπ3) = @all(dπ3) - 0.5/ρ * (@prd_d_xc(π3,π1) + @prd_d_yc(π3,π2) + @prd_d_zc(π3,π3))

    return
end

function deterministic(state, k1, k2, k3, rk_state, fft_temp)
    project(state.π, fft_temp)
    @parallel deterministic_elementary_step(view_tuple(state.u)..., view_tuple(k1)...)

    rk_state.u .= state.u .+ Δtdet*k1
    project(rk_state.π, fft_temp)
    @parallel deterministic_elementary_step(view_tuple(rk_state.u)..., view_tuple(k2)...)

    rk_state.u .= state.u .+ Δtdet*0.25*(k1 .+ k2)
    project(rk_state.π, fft_temp)
    @parallel deterministic_elementary_step(view_tuple(rk_state.u)..., view_tuple(k3)...)

    state.u .+= Δtdet*(0.5*k1 .+ 0.5*k2 .+ 2.0*k3)/3.0  
    project(state.π, fft_temp)
end

"""
  Elementary stochastic step with the transfer of the momentum density (μ-th component) from the cell x1 to x2 
"""
function pi_step(π, n, m, μ, (i,j,k))
    xyz = ((2i + m)%L+1, j%L+1, k%L+1)
    x1 = (xyz[(3-n)%3+1], xyz[(4-n)%3+1], xyz[(5-n)%3+1])
    x2 = ((x1[1]-(n!=0))%L+1, (x1[2]-(n!=1))%L+1, (x1[3]-(n!=2))%L+1)

    norm = cos(2pi*rand())*sqrt(-2.0*log(rand()))
    q = Rate_pi * norm

    δH = (q * (π[x1..., μ] - π[x2..., μ]) + q^2)/ρ
    P = min(1.0f0, exp(-δH))
    r = rand()

    π[x1..., μ] += q * (r<P)
    π[x2..., μ] -= q * (r<P)
end

function _gpu_pi(π, n, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:3*L^3÷2-1
        μ = l ÷ (L^3÷2) + 1
        i = (l ÷ L^2) % (L ÷ 2)
        j = (l ÷ L) % L
        k = l % L

        pi_step(π, n, m, μ, (i,j,k))
    end
end

"""
  Computing the local change of energy in the cell x 
"""
function ΔH_phi(ϕ, m², x, q)
    ϕold = ϕ[x...]
    ϕt = ϕold + q
    Δϕ = ϕt - ϕold
    Δϕ² = ϕt^2 - ϕold^2

    ∑nn = ϕ[NNp(x[1]), x[2], x[3]] + ϕ[x[1], NNp(x[2]), x[3]] + ϕ[x[1], x[2], NNp(x[3])]
        + ϕ[NNm(x[1]), x[2], x[3]] + ϕ[x[1], NNm(x[2]), x[3]] + ϕ[x[1], x[2], NNm(x[3])]

    return 3Δϕ² - Δϕ * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

function phi_step(ϕ, m², n, m, (i,j,k))
    xyz = ((4i + 2j + m%2)%L+1, (j + k + m÷2)%L+1, k%L+1)
    x1 = (xyz[(3-n)%3+1], xyz[(4-n)%3+1], xyz[(5-n)%3+1])
    x2 = ((x1[1]-(n!=0))%L+1, (x1[2]-(n!=1))%L+1, (x1[3]-(n!=2))%L+1)

    norm = cos(2pi*rand())*sqrt(-2*log(rand()))
    q = Rate_phi * norm

    δH = ΔH_phi(ϕ, m², x1, q) + ΔH_phi(ϕ, m², x2, -q) + q^2
    P = min(1.0f0, exp(-δH))
    r = rand()

    ϕ[x1...] += q * (r<P)
    ϕ[x2...] -= q * (r<P)
end

function _gpu_phi(ϕ, m², n, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷4-1
        i = l ÷ L^2
        j = (l ÷ L) % L
        k = l % L

        phi_step(ϕ, m², n, m, (i,j,k))
    end
end

gpu_phi = @cuda launch=false _gpu_phi(CuArray{FloatType}(undef,(L,L,L)), zero(FloatType), 0, 0)
gpu_pi  = @cuda launch=false _gpu_pi(CuArray{FloatType}(undef,(L,L,L,3)), 0, 0)

const N_phi = L^3÷4
config = launch_configuration(gpu_phi.fun)
const threads_phi = min(N_phi, config.threads)
const blocks_phi = cld(N_phi, threads_phi)

const N_pi = L^3÷2
config = launch_configuration(gpu_pi.fun)
const threads_pi = min(N_pi, config.threads)
const blocks_pi = cld(N_pi, threads_pi)

function dissipative(state, m²)
    # pi update
    for n in 0:2, m in 0:1
        gpu_pi(state.π, n, m; threads=threads_pi, blocks=blocks_pi)
    end

    # phi update
    for n in 0:2, m in 0:3
        gpu_phi(state.ϕ, m², n, m; threads=threads_phi, blocks=blocks_phi)
    end
end

"""
    sum_check(x)

Checks if any x, or any of its entries are infinite or NAN 
"""
function sum_check(x)
    s = sum(x)
    isnan(s) || !isfinite(s)
end

function thermalize(state, arrays, m², N)
    for _ in 1:N
        if sum_check(state.ϕ) 
            break
        end
        dissipative(state, m²)

        deterministic(state, arrays...)
    end
end

function prethermalize(state, fft_temp, m², N)
    for _ in 1:N
        if sum_check(state.ϕ) 
            break
        end
        dissipative(state, m²)
        project(state.π, fft_temp)
    end
end
