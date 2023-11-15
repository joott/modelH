#= 

.88b  d88.  .d88b.  d8888b. d88888b db           db   db 
88'YbdP`88 .8P  Y8. 88  `8D 88'     88           88   88 
88  88  88 88    88 88   88 88ooooo 88           88ooo88 
88  88  88 88    88 88   88 88~~~~~ 88           88~~~88 
88  88  88 `8b  d8' 88  .8D 88.     88booo.      88   88 
YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P      YP   YP 

=# 


#julia -t 10 --check-bounds=no

cd(@__DIR__)

using Distributions
using Printf
using Random
using FFTW
using Plots

const LEFT = -1
const RIGHT = 1




# """
# Random seed is set by the first argument passed to Julia
# """

ID = parse(Int,ARGS[1])
Random.seed!(ID)



# """
#   Parameters below are
#   1. L is the number of lattice sites in each dimension; it accepts the second argument passed to julia   
#   2. λ is the 4 field coupling
#   3. Γ is the scalar field diffusion rate; in our calculations we set it to 1, assuming that the time is measured in the appropriate units 
#   4. T is the temperature 
#   5. 
#   6. 
#   7. m² = -2.28587 is the critical value of the mass parameter 
#   8. 
#   9. 
#   10. 
# """
const L = parse(Int,ARGS[2])
#L=8
const λ = 4.0e0
const Γ = 1.0e0
const T = 1.0e0
const ρ = 1.0e0

const η = 1.0e0

const m² =  -2.28587
#const m² = -0.0e0
const Δt = 0.04e0/Γ

const Δtdet =  Δt/30

const Rate_phi = Float64(sqrt(2.0*Δt*Γ))
const Rate_pi = Float64(sqrt(2.0*Δt*η))
ξ = Normal(0.0e0, 1.0e0)


#set_zero_subnormals(true)


"""
    Hot start initializes n x n x n x Ncomponents array 
"""
function hotstart(n, Ncomponents)
	rand(ξ, n, n, n, Ncomponents)
end

"""
    Hot start initializes n x n x n array 
"""
function hotstart(n)
	rand(ξ, n, n, n)
end

"""
 Rerurns n+1 defined on a periodic lattice 
"""
function NNp(n)
    n%L+1
end

"""
 Rerurns n-1 defined on a periodic lattice 
"""
function NNm(n)
    (n+L-2)%L+1
end


include("Dissipative.jl")

include("Deterministic.jl")

"""

"""
function prethermalize(ϕ, π, m², N)
    for _ in 1:N
      if sum_check(ϕ) 
               break
      end
      dissipative(ϕ, π, m²)
      project(π)
    end
end


"""

"""
function thermalize(u, m², N) 
    ϕ = @view(u[:,:,:,4])
    Π = @view(u[:,:,:,1:3])

    for _ in 1:N
      if sum_check(ϕ) 
               break
      end
      #dissipative(ϕ, Π, m²)
      
      #project(Π)
      #E1 = energy(ϕ, Π)
      deterministic(u)
      #project(Π)
      #E2 = energy(ϕ, Π)
      #print(E1/E2)
      #print("\n")

    end
    project(Π)
    print(energy(ϕ, Π))
    print("\n")
end

"""

"""
function op(ϕ, L)
	ϕk = fft(ϕ)
	average = ϕk[1,1,1]/L^3
	(real(average),ϕk[:,1,1])
end


function kinetic_energy(ϕ,π)
    sum((3 * ϕ[x,y,z]^2 - ϕ[NNm(x),y,z] * ϕ[NNp(x),y,z] - ϕ[x,NNm(y),z] * ϕ[x,NNp(y),z] - ϕ[x,y,NNm(z)] * ϕ[x,y,NNp(z)])*0.25  for x in 1:L, y in 1:L, z in 1:L)/L^3
end


function energy(ϕ, π)
    K = kinetic_energy(ϕ,π)
    K + sum(0.5 * (π[x,y,z,1]^2 + π[x,y,z,2]^2 + π[x,y,z,3]^2  +  1/2 * m² * ϕ[x,y,z]^2 + λ * ϕ[x,y,z]^4 / 4  ) for x in 1:L, y in 1:L, z in 1:L)/L^3
end


"""
Main function, accepts no arguments; introduced to keep global scope tidier 
"""
function run()

  u = hotstart(L,4)
  ϕ = @view(u[:,:,:,4])
  Π = @view(u[:,:,:,1:3])

  for μ in 1:3
    Π[:,:,:,μ] .= Π[:,:,:,μ] .- shuffle(Π[:,:,:,μ]);
  end
  
  let 
    # sanity tests
    project(Π)
    PiTest = similar(Π)  
    ∇(Π, PiTest,  1)
    display(sum(PiTest))
    PiTest .= Π  
    project(Π)
    project(Π)
    project(Π)
    display(sum(PiTest .- Π  ))
    display( Π[1,1,1,1] )
  end 
  

  ϕ .= ϕ .- shuffle(ϕ)
  
  Π .= 0.0
  ϕ .= 0.0

  [Π[:, y, :, 1] .= 20*sin(2π/L*(y-1)) for y in 1:L]
  
  #[ϕ[i,j,k] = exp(-((i-L÷2)^2+(j-L÷2)^2+(k-L÷2)^2  )) for i in 1:L, j in 1:L, k in 1:L ] 

  maxt = 10^4
  skip = 50 

  for it in 1:20
    open("data/shearwave_$ID"*"_$it.dat","w") do io
      Printf.@printf(io, "%f", it*Δt)
      for i in 1:L
        Printf.@printf(io, " %f", Π[L÷2, i, L÷2, 1])
      end 
      Printf.@printf(io, "\n")
      Printf.flush(io)
    end
    thermalize(u, m², skip)
  end 

end

run()
