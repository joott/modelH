cd(@__DIR__)

using JLD2
using CodecZlib
using Printf

include("src/modelH.jl")

function op(phi)
    phik = cpu ? fft(phi) : Array(CUFFT.fft(phi))
    average = phik[1,1,1]/L^3
    (real(average),phik[:,1,1])
end

function main()
    @init_state
    arrays = make_temp_arrays(state)

    thermalize(state, arrays, m², L^3)

    tau_id = round(Int, 100 * parsed_args["mass"])
    eta = round(η,digits=3);
    model = H0 ? (NModC ? "diff" : "H0") : (NModC ? "self" : "H")

    skip = L
    maxt = 5*L^2

    open("/home/jkott/perm/modelH/dynamics/output_$(model)_phi_L_$(L)_eta_$(eta)_tau_$(tau_id)_id_$(seed).dat", "w") do iophi
    open("/home/jkott/perm/modelH/dynamics/output_$(model)_pi_L_$(L)_eta_$(eta)_tau_$(tau_id)_id_$(seed).dat", "w") do iopi
        for i in 0:maxt
            (M,ϕk) = op(state.ϕ)
            Printf.@printf(iophi, "%i %f", i*skip, M)
            for kx in 1:L÷2
                Printf.@printf(iophi, " %.15f %.15f", real(ϕk[kx]), imag(ϕk[kx]))
            end

            Printf.@printf(iophi, "\n")

            (M,pik) = op(@view(state.π[:,:,:,2]))
            Printf.@printf(iopi, "%i", i*skip)
            for kx in 1:L÷2
                Printf.@printf(iopi, " %.15f %.15f", real(pik[kx]), imag(pik[kx]))
            end

            Printf.@printf(iopi, "\n")

            if i%100==0
                Printf.flush(iophi)
                Printf.flush(iopi)
            end

            thermalize(state, arrays, m², skip)
        end
    end
    end
end

main()
