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
using JLD2


const LEFT = -1
const RIGHT = 1



# """
# Random seed is set by the first argument passed to Julia
# """

const ID = parse(Int,ARGS[1])
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

const η = 1.0 #parse(Float64,ARGS[3])

const m² =  -2.28587
#const m² = -0.0e0
const Δt = 0.04e0/Γ

const Δtdet =  Δt

const Rate_phi = Float64(sqrt(2.0*Δt*Γ))
const Rate_pi = Float64(sqrt(2.0*Δt*η))
ξ = Normal(0.0e0, 1.0e0)



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
include("Determinstic.jl")

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
function op(ϕ, L)
	ϕk = fft(ϕ)
	average = ϕk[1,1,1]/L^3
	(real(average),ϕk[:,1,1])
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
  
  ϕ .= ϕ .- shuffle(ϕ)
 
  maxt = 10*L^2 

  for it in 1:maxt 

   @time prethermalize(ϕ, Π, m², L^3) 
   
   jldsave("tmpdata/thermalized_$ID"*"_$L"*".jld2"; ϕ=ϕ, Π=Π)
  
  end 

end

run()
