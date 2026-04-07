using Gamma
using Test
import SpecialFunctions

@testset "gamma(::$T)" for (T, max, rtol) in ((Float16, 13, 1.0), (Float32, 43, 1.0), (Float64, 170, 7))
    @inferred gamma(one(T))
    v = rand(T, 10000)*max
    for x in v
        @test isapprox(T(SpecialFunctions.gamma(widen(x))), gamma(x), rtol=rtol*eps(T))
        if isinteger(x) && x != 0
            @test_throws DomainError gamma(-x)
        else
            @test isapprox(T(SpecialFunctions.gamma(widen(-x))), gamma(-x), atol=nextfloat(T(0.),2), rtol=rtol*eps(T))
        end
    end
    @test isnan(gamma(T(NaN)))
    @test isinf(gamma(T(Inf)))
    @test_throws DomainError isinf(gamma(-T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ gamma.(x)
@inferred gamma(1)

include("test_loggamma.jl")