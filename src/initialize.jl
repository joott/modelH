using ArgParse
using Distributions
using Random
using CUDA
using ParallelStencil

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--mass"
            help = "mass as an offset from the critical mass m²c"
            arg_type = Float64
            default = 0.0
        "--dt"
            help = "size of time step"
            arg_type = Float64
            default = 0.04
        "--rng"
            help = "seed for random number generation"
            arg_type = Int
            default = 0
        "--fp64"
            help = "flag to use Float64 type rather than Float32"
            action = :store_true
        "--H0"
            help = "disable ππ term in deterministic step"
            action = :store_true
        "size"
            help = "side length of lattice"
            arg_type = Int
            required = true
        "viscosity"
            help = "the shear viscosity η in lattice units"
            arg_type = Float64
            required = true
    end

    return parse_args(s)
end

#=
 Parameters below are
 1. L is the number of lattice sites in each dimension; it accepts the second argument passed to julia   
 2. λ is the 4 field coupling
 3. Γ is the scalar field diffusion rate; in our calculations we set it to 1, assuming that the time is measured in the appropriate units 
 4. T is the temperature 
 5. m² = -2.28587 is the critical value of the mass parameter 
=#

parsed_args = parse_commandline()

const H0 = parsed_args["H0"]
const FloatType = parsed_args["fp64"] ? Float64 : Float32
const ComplexType = complex(FloatType)
const ArrayType = CuArray

const λ = FloatType(4.0)
const Γ = FloatType(1.0)
const T = FloatType(1.0)
const ρ = FloatType(1.0)

const L = parsed_args["size"]
const η = FloatType(parsed_args["viscosity"])
const m² = FloatType(-2.28587 + parsed_args["mass"])
const Δt = FloatType(parsed_args["dt"]/Γ)

const Δtdet = Δt
const Rate_phi = FloatType(sqrt(2.0*Δt*Γ))
const Rate_pi  = FloatType(sqrt(2.0*Δt*η))
const ξ = Normal(FloatType(0.0), FloatType(1.0))

const ID = parsed_args["rng"]
if ID != 0
    Random.seed!(ID)
    CUDA.seed!(ID)
end

@init_parallel_stencil(CUDA, FloatType, 3);
