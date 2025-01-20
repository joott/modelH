cd(@__DIR__)

using JLD2
using CodecZlib

include("src/modelH.jl")

function main()
    @init_state
    fft_temp = ArrayType{ComplexType}(undef, (L,L,L,3))
    mass_id = round(parsed_args["mass"], digits=3)

    for i in 1:L
        prethermalize(state, fft_temp, m², L^3)
        @show i
        flush(stdout)
        save_state("/home/jkott/perm/modelH/thermalized/thermalized_L_$(L)_mass_$(mass_id)_id_$(seed).jld2", state, m²)
    end
end

main()
