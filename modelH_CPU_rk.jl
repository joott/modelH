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

#Random.seed!(parse(Int,ARGS[1]))



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

const η = 0.1e0

const m² =  -2.28587
#const m² = -0.0e0
const Δt = 0.04e0/Γ

const Δtdet =  Δt

const Rate_phi = Float64(sqrt(2.0*Δt*Γ))
const Rate_pi = Float64(sqrt(2.0*Δt*η))
ξ = Normal(0.0e0, 1.0e0)


#set_zero_subnormals(true)


"""
    sum_check(x)

Checks if any x, or any of its entries are infinite or NAN 
"""
function sum_check(x)
    s = sum(x)
    isnan(s) || !isfinite(s)
end

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


"""
  defines the shifts of arrays in the n-th dimension (component) and the direction direction
  This is useful for derivatives on the lattice 
"""
function shifts(direction, component)
    # quite obscure set of operations to avoid ifs 
    (-direction*(1÷component), -direction*mod(2÷component,2), -direction*(component÷3))
end

function shifts2(direction, component)
    # quite obscure set of operations to avoid ifs 
    (-2*direction*(1÷component), -2*direction*mod(2÷component,2), -2*direction*(component÷3))
end

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
    @inbounds x1 = (xyz[n%3+1], xyz[(n+1)%3+1], xyz[(n+2)%3+1])
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
    @inbounds x1 = (xyz[n%3+1], xyz[(n+1)%3+1], xyz[(n+2)%3+1])
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


"""
Central differene Laplacian 
"""
function Δc(ϕ)
    dϕ = -6*ϕ

    for μ in 1:3
        dϕ .+= circshift(ϕ, shifts2(LEFT, μ)) 
        dϕ .+= circshift(ϕ, shifts2(RIGHT,μ)) 
    end

    dϕ .*= 0.25

    return dϕ
end 

"""
Central difference
"""
function ∇(src, dest, component) 
    dest .= circshift(src, shifts(RIGHT,component)) 
    dest .-= circshift(src, shifts(LEFT,component)) 

    dest .*= 0.5 
    return 1 
end


"""
  Solve Central Poisson equation Δc ϕ' = ϕ, overwrites ϕ
"""
function Poissonc(ϕ)
    Afft = fft(ϕ)
    Threads.@threads for n3 in 1:L 
      for n2 in 1:L, n1 in 1:L
        k2 =  (sin(2*pi/L * (n1-1))^2 + sin(2*pi/L * (n2-1))^2 + sin(2*pi/L * (n3-1))^2)
        if k2 > 1e-11
           @inbounds Afft[n1,n2,n3] = -Afft[n1,n2,n3] /  k2 
        else 
           @inbounds Afft[n1,n2,n3] = 0.0 
        end

      end
    end

    ϕ .= real.(ifft(Afft))
end

"""
  Central Projector 
"""
function project(π)
    # ∇_μ projectμ = 0 

    π_sum = zeros(Float64,(L,L,L))
    temp = zeros(Float64,(L,L,L,3))

    for μ in 1:3 
        ∇(@view(π[:,:,:,μ]), @view(temp[:,:,:,μ]), μ)
        π_sum .-= temp[:,:,:,μ] # note that π_sum is actiually - π_sum
    end 

    Threads.@threads for μ in 1:3 
        ∇(π_sum, @view(temp[:,:,:,μ]), μ)
        temp[:,:,:,μ] .+= Δc(@view(π[:,:,:,μ]))
        Poissonc(@view(temp[:,:,:,μ]))    
    end

    π[:,:,:,1:3] .= temp

    return 1
end


function deterministic_elementary_step(du, u)
    ϕ = @view u[:,:,:,4]
    
    dϕ = @view du[:,:,:,4]


    # temporary arrays 
    dj = similar(ϕ)
    dϕ_ν = similar(ϕ)
    Laplacian = Δc(ϕ)
    ππ = similar(ϕ)

    dj .= 0.0 
    for μ in 1:3
        ∇(ϕ, dϕ, μ) # ∇_μ ϕ
        dj .+= 1.0/ρ*dϕ.*u[:,:,:,μ]         
    end

    du[:,:,:,1:3] .= 0.0
    dϕ .= -dj 

    for ν in 1:3
        ∇(ϕ, dϕ_ν, ν)

        for μ in 1:3
            ππ .= u[:,:,:,ν].*u[:,:,:,μ]
            ∇(ππ, dj, μ) # ∇_μ (π_ν π_μ)
            du[:,:,:,ν] .-= 0.5*dj/ρ  # -1/2ρ ∇_μ π_μ π_ν
            
            ∇(@view(u[:,:,:,ν]), dj ,μ) # ∇_μ π_ν

            du[:,:,:,ν] .-= 0.5*dj.*u[:,:,:,μ]/ρ  # -1/2ρ π_μ ∇_μ π_ν
        end
        
        du[:,:,:,ν] .-= dϕ_ν.*Laplacian 

    end

    return 1 # returns * or ** updates 
end


"""
Wray RK3 scheme
"""
function deterministic(u)
    k1 = similar(u)   
    k2 = similar(u)   
    k3 = similar(u)   
    
    temp = similar(u)   

    project(u)
    deterministic_elementary_step(k1, u)
    
    temp .= u.+Δtdet*k1
    project(temp)
    deterministic_elementary_step(k2, temp)

    temp .= u .+ Δtdet*0.25*(k1 .+ k2)
    project(temp)
    deterministic_elementary_step(k3, temp)

    u .+= Δtdet*(0.5*k1 .+ 0.5*k2 .+ 2.0*k3)/3.0  
    project(u)
end



"""

"""
function prethermalize(ϕ, π, m², N)
    for _ in 1:N
      if sum_check(ϕ) 
               break
      end
      dissipative(ϕ, π, m²)
      project(π)
      dissipative(ϕ, π, m²)
      project(π)
    end
end


"""

"""
function thermalize_en(u, m², N) 
    ϕ = @view(u[:,:,:,4])
    Π = @view(u[:,:,:,1:3])

    for _ in 1:N
      if sum_check(ϕ) 
               break
      end
      dissipative(ϕ, Π, m²)
      
      project(Π)
      E1 = energy(ϕ, Π)
      deterministic(u)
      project(Π)
      E2 = energy(ϕ, Π)
      print(E1/E2)
      print("\n")

    end
    #project(Π)
    #print(energy(ϕ, Π))
    #print("\n")
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
      dissipative(ϕ, Π, m²)
      
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
  
  #Π .= 0.0e0

  ϕ .= ϕ .- shuffle(ϕ)
 
  #smoothen 
  [@time prethermalize(ϕ, Π, m², L^3) for i in 1:L]
  
  #ϕ .= 0.0
  
  #Π .= 0.0e0

  #[ϕ[i,j,k] = exp(-((i-L÷2)^2+(j-L÷2)^2+(k-L÷2)^2  )) for i in 1:L, j in 1:L, k in 1:L ] 

  # energy conservation per deterministic step check 
  [@time thermalize_en(u, m², 1) for i in 1:50]

  [@time thermalize(u, m², L^4) for i in 1:5]
  
  maxt = 10*L^4
  skip = 10 

  open("output_$L.dat","w") do io 
  open("outputpi_$L.dat","w") do iopi 
	  for i in 0:maxt
		  
      if sum_check(ϕ) 
               break
      end

      project(Π)
      (M,ϕk) = op(ϕ, L)
		  Printf.@printf(io, "%i %f", i*skip, M)
		  for kx in 1:L
			  Printf.@printf(io, " %f %f", real(ϕk[kx]), imag(ϕk[kx]))
		  end 
		  
      Printf.@printf(io, " %f \n", energy(ϕ, Π))
      Printf.flush(io)
      
      (M,pik) = op(@view(u[:,:,:,2]), L)
		  Printf.@printf(iopi, "%i %f", i*skip, M)
		  for kx in 1:L
			  Printf.@printf(iopi, " %f %f", real(pik[kx]), imag(pik[kx]))
		  end 

		  Printf.@printf(iopi, "\n")
      Printf.flush(iopi)

		  thermalize(u, m², skip)
      #display(i)
	  end
  end
  end


end

run()
