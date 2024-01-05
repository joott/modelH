#= 

.88b  d88.  .d88b.  d8888b. d88888b db           db   db 
88'YbdP`88 .8P  Y8. 88  `8D 88'     88           88   88 
88  88  88 88    88 88   88 88ooooo 88           88ooo88 
88  88  88 88    88 88   88 88~~~~~ 88           88~~~88 
88  88  88 `8b  d8' 88  .8D 88.     88booo.      88   88 
YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P      YP   YP 

=# 

cd(@__DIR__)

using Distributions
using Printf
using Random
using CUDA
using JLD2
using CodecZlib

include("initialize.jl")
include("simulation.jl")

struct State
    u::ArrayType
    π::ArrayType
    ϕ::ArrayType
    State(u) = new(u, @view(u[:,:,:,1:3]), @view(u[:,:,:,4]))
end

"""
    Hot start initializes system state such that `sum(component) = 0`
"""
function hotstart(n, n_components)
	u = rand(ξ, n, n, n, n_components)

    for i in 1:4
        u[:,:,:,i] .-= shuffle(u[:,:,:,i])
    end

    State(ArrayType(u))
end

function make_temp_arrays(state)
    (k1, k2, k3) = (similar(state.u), similar(state.u), similar(state.u))
    rk_temp = State(similar(state.u))
    fft_temp = ArrayType{ComplexType}(undef, (L,L,L,3))
    (k1,k2,k3,rk_temp,fft_temp)
end

function kinetic_energy(ϕ)
    0.25 * sum(3 * ϕ.^2 - circshift(ϕ, (1,0,0)) .* circshift(ϕ, (-1,0,0))
                        - circshift(ϕ, (0,1,0)) .* circshift(ϕ, (0,-1,0))
                        - circshift(ϕ, (0,0,1)) .* circshift(ϕ, (0,0,-1)))
end

function energy(state)
    K = kinetic_energy(state.ϕ)
    (π1, π2, π3) = view_tuple(state.π)
    K + sum(0.5 * (π1.^2 + π2.^2 + π3.^2 + 1/2 * m² * state.ϕ.^2 + λ/4 * state.ϕ.^2))
end

function main()
    state = hotstart(L,4)
    fft_temp = ArrayType{ComplexType}(undef, (L,L,L,3))

    for i in 1:L^2
        prethermalize(state, fft_temp, m², L^2)
        @show i
        jldsave("thermalized/thermalized_L_$(L)_id_$(ID).jld2", true; u=Array(state.u), m²=m²)
    end
end

main()
