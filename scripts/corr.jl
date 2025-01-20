cd(@__DIR__)

using DelimitedFiles
using Glob
using JLD2
using CodecZlib

include("bootstrap.jl")

const L = 24

function autocor_loc_2(x, beg, max, n=2)
	C = zeros(Complex{Float64},max+1)
	N = zeros(Int64,max+1)
	Threads.@threads for tau in 0:max
		for i in beg:length(x)-max
			j = i + tau
			@inbounds C[tau+1] = C[tau+1] +  (x[i]*conj(x[j]))^n
			@inbounds N[tau+1] = N[tau+1] + 1
		end
	end
	(collect(0:max),  real.(C) ./ N)
end

jldopen("/home/jkott/perm/modelH/corr_L_$(L).jld2", "w") do savefile

for (i,mass) in enumerate([0.04, 0.119, 0.221])
for (j,eta) in enumerate([0.01, 0.1, 1.0, 10.0])
    tau_id = round(Int, mass*100)
    files = glob("output_H0_phi_L_$(L)_eta_$(eta)_tau_$(tau_id)_id_*.dat", "/home/jkott/perm/modelH/dynamics")
    measurements = fill([], 3)

    for file in files
        data = readdlm(file, ' ')
        for k in 1:3
            phik = data[:,2k+3] .+ data[:,2k+4].*im
            (t, tmp) = autocor_loc_2(phik, 1, L^2, 1)
            push!(measurements[k], real.(tmp))
        end
    end

    for k in 1:3
        (C, C_err) = bootstrap(measurements[k], 100)
        savefile["tau_$(mass)_eta_$(eta)_k_$k/C"] = C
        savefile["tau_$(mass)_eta_$(eta)_k_$k/err"] = C_err
    end
end
end

end
