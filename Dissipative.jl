#= 

.88b  d88.  .d88b.  d8888b. d88888b db           db   db 
88'YbdP`88 .8P  Y8. 88  `8D 88'     88           88   88 
88  88  88 88    88 88   88 88ooooo 88           88ooo88 
88  88  88 88    88 88   88 88~~~~~ 88           88~~~88 
88  88  88 `8b  d8' 88  .8D 88.     88booo.      88   88 
YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P      YP   YP 

=# 


"""
  Elementary stochastic step with the transfer of the momentum density (μ-th component) from the cell x1 to x2 
"""
function pi_step(π, μ, x1, x2)
    norm = cos(2pi*rand())*sqrt(-2.0*log(rand()))
    q = Rate_pi * norm

    @inbounds δH = (q * (π[x1..., μ] - π[x2..., μ]) + q^2)/ρ
    P = min(1.0f0, exp(-δH))
    r = rand()

    @inbounds π[x1..., μ] += q * (r<P)
    @inbounds π[x2..., μ] -= q * (r<P)
    # note that (r<P) returns wither 1 or 0 
end

"""

"""
function pi_sweep(π, n, m, μ, (i,j,k))
    xyz = ((2i + m)%L+1, j%L+1, k%L+1)
    @inbounds x1 = (xyz[(3-n)%3+1], xyz[(4-n)%3+1], xyz[(5-n)%3+1])
    @inbounds x2 = ((x1[1]-(n!=0))%L+1, (x1[2]-(n!=1))%L+1, (x1[3]-(n!=2))%L+1)

    pi_step(π, μ, x1, x2)
end

"""
  Computing the local change of energy in the cell x 
"""
function ΔH_phi(x, ϕ, q, m²)
    @inbounds ϕold = ϕ[x...]
    ϕt = ϕold + q
    Δϕ = ϕt - ϕold
    #Δϕ = q
    Δϕ² = ϕt^2 - ϕold^2

    @inbounds ∑nn = ϕ[NNp(x[1]), x[2], x[3]] + ϕ[x[1], NNp(x[2]), x[3]] + ϕ[x[1], x[2], NNp(x[3])] + ϕ[NNm(x[1]), x[2], x[3]] + ϕ[x[1], NNm(x[2]), x[3]] + ϕ[x[1], x[2], NNm(x[3])]

    return 3Δϕ² - Δϕ * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

"""

"""
function phi_step(m², ϕ, x1, x2)
    norm = cos(2pi*rand())*sqrt(-2*log(rand()))
    q = Rate_phi * norm

    δH = ΔH_phi(x1, ϕ, q, m²) + ΔH_phi(x2, ϕ, -q, m²) + q^2
    P = min(1.0f0, exp(-δH))
    r = rand()

    @inbounds ϕ[x1...] += q * (r<P)
    @inbounds ϕ[x2...] -= q * (r<P)
end

"""

"""
function phi_sweep(m², ϕ, n, m, (i,j,k))
    xyz = ((4i + 2j + m%2)%L+1, (j + k + m÷2)%L+1, k%L+1)
    @inbounds x1 = (xyz[(3-n)%3+1], xyz[(4-n)%3+1], xyz[(5-n)%3+1])
    @inbounds x2 = ((x1[1]-(n!=0))%L+1, (x1[2]-(n!=1))%L+1, (x1[3]-(n!=2))%L+1)

    phi_step(m², ϕ, x1, x2)
end

"""

"""
function dissipative(ϕ, π, m²)
    # pi update
    for n in 0:2, m in 0:1
        Threads.@threads for index in 0:3*L^3÷2-1
            μ = index ÷ (L^3÷2) + 1
            i = (index ÷ L^2) % (L ÷ 2)
            j = (index ÷ L) % L
            k = index % L

            pi_sweep(π, n, m, μ, (i,j,k))
        end
    end

    # phi update
    for n in 0:2, m in 0:3
        Threads.@threads for index in 0:L^3÷4-1
            i = index ÷ L^2
            j = (index ÷ L) % L
            k = index % L

            phi_sweep(m², ϕ, n, m, (i,j,k))
        end
    end
end
