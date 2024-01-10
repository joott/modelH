cd(@__DIR__)

using JLD2
using CodecZlib

include("src/modelH.jl")

function main()
    state = hotstart(L,4)
    fft_temp = ArrayType{ComplexType}(undef, (L,L,L,3))

    for i in 1:L^2
        prethermalize(state, fft_temp, m², L^2)
        @show i
        jldsave("/home/shared/modelH/thermalized/thermalized_L_$(L)_id_$(ID).jld2", true; u=Array(state.u), m²=m²)
    end
end

main()
