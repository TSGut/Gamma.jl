module Gamma

    include("gamma_implementation.jl")
    include("precompile.jl")
    include("loggamma.jl")

    export gamma, loggamma, logabsgamma, logfactorial

end
