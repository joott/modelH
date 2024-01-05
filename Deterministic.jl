#= 

.88b  d88.  .d88b.  d8888b. d88888b db           db   db 
88'YbdP`88 .8P  Y8. 88  `8D 88'     88           88   88 
88  88  88 88    88 88   88 88ooooo 88           88ooo88 
88  88  88 88    88 88   88 88~~~~~ 88           88~~~88 
88  88  88 `8b  d8' 88  .8D 88.     88booo.      88   88 
YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P      YP   YP 

=# 

"""
    sum_check(x)

Checks if any x, or any of its entries are infinite or NAN 
"""
function sum_check(x)
    s = sum(x)
    isnan(s) || !isfinite(s)
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

