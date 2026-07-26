# The generalized exponential-integral machinery in this file is adapted from
# SpecialFunctions.jl, src/expint.jl, version 2.7.2, MIT License.

"""
    IncompleteGammaConvergenceError(algorithm, iterations)

The requested precision was not reached by an incomplete-gamma algorithm.
"""
struct IncompleteGammaConvergenceError <: Exception
    algorithm::Symbol
    iterations::Int
end

function Base.showerror(io::IO, err::IncompleteGammaConvergenceError)
    print(io, err.algorithm, " did not converge after ", err.iterations,
          " iterations at the requested precision")
end

# Public entry points

"""
    gamma(a, z)

Compute the unnormalised upper incomplete gamma function `Γ(a,z)`.
"""
gamma(a::T, z::T) where {T<:Union{Float16,Float32,Float64}} = _gamma(a, z)
gamma(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float16,Float32,Float64}
} = _gamma(a, z)
gamma(a::BigFloat, z::BigFloat) = _gamma(a, z)
gamma(a::Complex{BigFloat}, z::Complex{BigFloat}) = _gamma(a, z)
function gamma(a::Number, z::Number)
    promoted = promote(float(a), float(z))
    return _gamma(promoted...)
end

"""
    gamma_lower(a, z)

Compute the unnormalised lower incomplete gamma function `γ(a,z)`.
"""
gamma_lower(a::T, z::T) where {T<:Union{Float16,Float32,Float64}} =
    _gamma_lower(a, z)
gamma_lower(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float16,Float32,Float64}
} = _gamma_lower(a, z)
gamma_lower(a::BigFloat, z::BigFloat) = _gamma_lower(a, z)
gamma_lower(a::Complex{BigFloat}, z::Complex{BigFloat}) = _gamma_lower(a, z)
function gamma_lower(a::Number, z::Number)
    promoted = promote(float(a), float(z))
    return _gamma_lower(promoted...)
end

"""
    gamma_inc(a, z)

Return the regularised lower and upper incomplete gamma functions `(P, Q)`,
with `P + Q = 1`.
"""
gamma_inc(a::T, z::T) where {T<:Union{Float16,Float32,Float64}} =
    _gamma_inc(a, z)
gamma_inc(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float16,Float32,Float64}
} = _gamma_inc(a, z)
gamma_inc(a::BigFloat, z::BigFloat) = _gamma_inc(a, z)
gamma_inc(a::Complex{BigFloat}, z::Complex{BigFloat}) = _gamma_inc(a, z)
function gamma_inc(a::Number, z::Number)
    promoted = promote(float(a), float(z))
    return _gamma_inc(promoted...)
end

"""
    expint(z)
    expint(ν, z)

Compute the generalized exponential integral
`E_ν(z) = ∫₁^∞ exp(-z*t)/t^ν dt` on its principal branch.
"""
expint(ν::T, z::T) where {T<:Union{Float16,Float32,Float64}} =
    _expint(ν, z, false)
expint(ν::Complex{T}, z::Complex{T}) where {
    T<:Union{Float16,Float32,Float64}
} = _expint(ν, z, false)
expint(ν::BigFloat, z::BigFloat) = _expint(ν, z, false)
expint(ν::Complex{BigFloat}, z::Complex{BigFloat}) = _expint(ν, z, false)
function expint(ν::Number, z::Number)
    promoted = promote(float(ν), float(z))
    return _expint(promoted..., false)
end
expint(z::Number) = expint(one(z), z)

"""
    expintx(z)
    expintx(ν, z)

Compute the exponentially scaled generalized exponential integral
`exp(z) * E_ν(z)`.
"""
expintx(ν::T, z::T) where {T<:Union{Float16,Float32,Float64}} =
    _expint(ν, z, true)
expintx(ν::Complex{T}, z::Complex{T}) where {
    T<:Union{Float16,Float32,Float64}
} = _expint(ν, z, true)
expintx(ν::BigFloat, z::BigFloat) = _expint(ν, z, true)
expintx(ν::Complex{BigFloat}, z::Complex{BigFloat}) = _expint(ν, z, true)
function expintx(ν::Number, z::Number)
    promoted = promote(float(ν), float(z))
    return _expint(promoted..., true)
end
expintx(z::Number) = expintx(one(z), z)

# Generalized exponential integral

# Gamma-free continued fraction for E_ν(z):
# https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/10/0001/
function _En_cf_nogamma(ν::T, z::T;
                        maxiter::Union{Nothing,Int}=nothing,
                        throw_on_failure::Bool=true) where {T<:AbstractFloat}
    R = T
    tol = 8 * eps(one(R))
    cap = if isnothing(maxiter)
        # Precision-scaled bound from the root-exponential estimate
        # exp(-2*sqrt(n)*real(sqrt(z))) for the Legendre fraction.
        target = -log(tol)
        rate = if z >= 0
            rootz = sqrt(z)
            2 * max(
                rootz, sqrt(eps(one(R))) * max(one(R), rootz)
            )
        else
            rootz = sqrt(complex(z))
            2 * max(
                real(rootz),
                sqrt(eps(one(R))) * max(one(R), abs(rootz)),
            )
        end
        asymptotic = (target / rate)^2
        startup = target + abs(ν) + abs(z) + R(32)
        estimate = ceil(max(asymptotic, startup))
        if !isfinite(estimate) || estimate > typemax(Int)
            throw(IncompleteGammaConvergenceError(
                :expint_continued_fraction, typemax(Int)
            ))
        end
        max(16, Int(estimate))
    else
        maxiter
    end
    return _En_cf_nogamma_recurrence(ν, z, tol, cap, throw_on_failure)
end

function _En_cf_nogamma(ν::Complex{T}, z::Complex{T};
                        maxiter::Union{Nothing,Int}=nothing,
                        throw_on_failure::Bool=true) where {T<:AbstractFloat}
    tol = 8 * eps(one(T))
    cap = if isnothing(maxiter)
        target = -log(tol)
        rootz = sqrt(z)
        rate = 2 * max(
            real(rootz),
            sqrt(eps(one(T))) * max(one(T), abs(rootz)),
        )
        asymptotic = (target / rate)^2
        startup = target + abs(ν) + abs(z) + T(32)
        estimate = ceil(max(asymptotic, startup))
        if !isfinite(estimate) || estimate > typemax(Int)
            throw(IncompleteGammaConvergenceError(
                :expint_continued_fraction, typemax(Int)
            ))
        end
        max(16, Int(estimate))
    else
        maxiter
    end
    return _En_cf_nogamma_recurrence(ν, z, tol, cap, throw_on_failure)
end

function _En_cf_nogamma_recurrence(ν::T, z::T, tol, cap,
                                    throw_on_failure) where {T}
    B = float(z + ν)
    Bprev::typeof(B) = z
    A::typeof(B) = one(B)
    Aprev::typeof(B) = one(B)
    previous = Aprev / Bprev
    stable = 0

    for i = 2:cap
        previous_coefficient = i - 1
        A, Aprev = z * A + previous_coefficient * Aprev, A
        B, Bprev = z * B + previous_coefficient * Bprev, B
        coefficient = ν + previous_coefficient
        A, Aprev = A + coefficient * Aprev, A
        B, Bprev = B + coefficient * Bprev, B

        # Rescale the recurrence.
        scale = max(abs(A), abs(Aprev), abs(B), abs(Bprev))
        if isfinite(scale) && !iszero(scale)
            A /= scale
            Aprev /= scale
            B /= scale
            Bprev /= scale
        end

        current = A / B
        denom = max(abs(current), abs(previous))
        if i > 4 && (iszero(denom) ? current == previous :
                    abs(current - previous) <= tol * denom)
            stable += 1
            stable >= 2 && return current, i, true
        else
            stable = 0
        end
        previous = current
    end

    throw_on_failure &&
        throw(IncompleteGammaConvergenceError(:expint_continued_fraction, cap))
    return A / B, cap, false
end

function _series_iteration_cap(parameter, z)
    R = typeof(real(z))
    p = R(precision(R))
    estimate = ceil(
        exp(one(R)) * abs(z) + log(R(2)) * p + abs(parameter) + R(32)
    )
    return Int(min(estimate, R(typemax(Int))))
end

function _En_expand_origin_posint(
    n, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T<:AbstractFloat}
    m = n - 1
    gammaterm = if m < 100
        result = one(z)
        for k = 1:Int(m)
            result *= -z / k
        end
        result
    else
        sign = isodd(Int(m)) ? -one(m) : one(m)
        sign * exp(m * log(abs(z)) - loggamma(m + 1))
    end
    return _En_expand_origin_posint(n, z, gammaterm, maxiter)
end

function _En_expand_origin_posint(
    n, z::Complex{T}; maxiter::Union{Nothing,Int}=nothing
) where {T<:AbstractFloat}
    m = n - 1
    gammaterm = if m < 100
        result = one(z)
        for k = 1:Int(m)
            result *= -z / k
        end
        result
    else
        exp(m * log(-z) - loggamma(m + 1))
    end
    return _En_expand_origin_posint(n, z, gammaterm, maxiter)
end

function _En_expand_origin_posint(n, z::T, gammaterm::T,
                                  maxiter::Union{Nothing,Int}) where {T}
    frac = one(z)
    gammaterm *= digamma(oftype(real(z), n)) - log(z)
    sumterm = n == 1 ? zero(z) : frac / (1 - n)
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? _series_iteration_cap(n, z) : maxiter
    stable = 0

    for k = 1:cap
        frac *= -z / k
        if k != n - 1
            term = frac / (k + 1 - n)
            sumterm += term
            if abs(term) <= tol * max(abs(sumterm), one(R))
                stable += 1
                stable >= 2 && return gammaterm - sumterm
            else
                stable = 0
            end
        end
    end
    throw(IncompleteGammaConvergenceError(:expint_origin_series, cap))
end

function _En_expand_origin_general(
    ν::T, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T}
    gammaterm = gamma(1 - ν) * z^(ν - 1)
    frac = one(z)
    sumterm = frac / (1 - ν)
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? _series_iteration_cap(ν, z) : maxiter
    stable = 0

    for k = 1:cap
        frac *= -z / k
        term = frac / (k + 1 - ν)
        sumterm += term
        if abs(term) <= tol * max(abs(sumterm), one(R))
            stable += 1
            stable >= 2 && return gammaterm - sumterm
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(:expint_origin_series, cap))
end

function _En_taylor(ν::T, start::T, z0::T, delta::T;
                    maxiter::Union{Nothing,Int}=nothing) where {T}
    a = exp(z0) * start
    total = a
    delta_prod_fact = -delta
    R = typeof(real(total))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? max(32, 2 * precision(R)) : maxiter
    stable = 0

    for k = 0:cap
        a = (delta_prod_fact + a * delta * (ν - k - 1) / (k + 1)) / z0
        total += a
        if abs(a) <= tol * max(abs(total), one(R))
            stable += 1
            stable >= 2 && return exp(-z0) * total
        else
            stable = 0
        end
        delta_prod_fact *= -delta / (k + 2)
    end
    throw(IncompleteGammaConvergenceError(:expint_continuation_series, cap))
end

function _En_safeexpmult(z::T, value::T) where {T<:Real}
    ez = exp(z)
    if isinf(ez) || iszero(ez)
        return sign(value) * exp(z + log(abs(value)))
    end
    return ez * value
end

function _En_safeexpmult(
    z::Complex{T}, value::Complex{T}
) where {T<:AbstractFloat}
    ez = exp(z)
    if isinf(ez) || iszero(ez)
        return exp(z + log(value))
    end
    return ez * value
end

function _expint_left_halfplane(
    ν::Complex{T}, z::Complex{T}, expscaled
) where {T<:AbstractFloat}
    reflect = imag(z) < 0
    rez, imz = real(z), abs(imag(z))
    original_z = z
    z = reflect ? conj(z) : z
    ν = reflect ? conj(ν) : ν
    R = typeof(real(z))
    quick = max(64, 2 * precision(R))
    target = -log(8 * eps(one(R)))
    s = R(5) * target / (R(16) * sqrt(R(quick)))
    estimate = 2 * s * sqrt(max(zero(R), s^2 - real(z)))
    imstart = max(imz, estimate)
    z0 = complex(rez, imstart)
    cf, _, converged = _En_cf_nogamma(ν, z0; maxiter=quick,
                                      throw_on_failure=false)
    start = _En_safeexpmult(-z0, cf)

    doublings = 0
    while !converged
        imstart *= 2
        z0 = complex(rez, imstart)
        cf, _, converged = _En_cf_nogamma(ν, z0; maxiter=quick,
                                          throw_on_failure=false)
        start = _En_safeexpmult(-z0, cf)
        doublings += 1
        doublings > 4 * precision(R) + 64 &&
            throw(IncompleteGammaConvergenceError(:expint_left_halfplane, quick))
    end

    if imz > 0 && z0 == z
        result = expscaled ? cf : start
        return reflect ? conj(result) : result
    end

    distance = imstart - imz
    nsteps = max(1, ceil(Int, 2 * distance))
    nsteps > 1_000_000 &&
        throw(IncompleteGammaConvergenceError(:expint_left_halfplane, nsteps))
    delta = (imz - imstart) * im / nsteps
    for _ = 1:nsteps
        start = _En_taylor(ν, start, z0, delta)
        z0 += delta
    end

    result = reflect ? conj(start) : start
    if iszero(imz)
        x = real(z)
        e1 = exp(oftype(x, π) * imag(ν))
        e2 = Complex(cospi(real(ν)), -sinpi(real(ν)))
        lg, sign = loggamma(ν), 1
        branchjump = -2 * sign * e1 * oftype(x, π) * e2 *
                     exp((ν - 1) * log(complex(x)) - lg) * im
        upper_lip = !signbit(imag(original_z))
        branchsign = upper_lip ? 1 : -1
        if isreal(ν)
            result = real(result) - branchsign * imag(branchjump) / 2 * im
        elseif !upper_lip
            result += branchjump
        end
    end
    return expscaled ? _En_safeexpmult(original_z, result) : result
end

function _expint(ν::T, z::T, expscaled::Bool) where {
    T<:Union{Float32,Float64,BigFloat}
}
    if isnan(ν) || isnan(z)
        return oftype(z, NaN) * z
    elseif z < 0
        throw(DomainError(z,
            "expint has a complex value for negative z; pass complex(z)"))
    elseif iszero(z)
        return oftype(z, ν > 1 ? inv(ν - 1) : T(Inf))
    elseif iszero(ν)
        return expscaled ? inv(z) : exp(-z) / z
    end
    return _expint_unsafe(ν, z, expscaled)
end

function _expint(ν::Complex{T}, z::Complex{T},
                 expscaled::Bool) where {
    T<:Union{Float32,Float64,BigFloat}
}
    if isnan(ν) || isnan(z)
        return oftype(z, NaN) * z
    elseif iszero(z)
        return oftype(z, real(ν) > 1 ? inv(ν - 1) : T(Inf))
    elseif iszero(ν)
        return expscaled ? inv(z) : exp(-z) / z
    end
    return _expint_unsafe(ν, z, expscaled)
end

_expint(ν::Float16, z::Float16, expscaled::Bool) =
    Float16(_expint(Float32(ν), Float32(z), expscaled))
_expint(ν::ComplexF16, z::ComplexF16, expscaled::Bool) =
    ComplexF16(_expint(ComplexF32(ν), ComplexF32(z), expscaled))

function _expint_unsafe(ν::T, z::T,
                        expscaled::Bool) where {T<:AbstractFloat}
    if abs2(z) < 9
        result = if isinteger(ν) && ν > 0
            n = ν
            n <= typemax(Int) ? _En_expand_origin_posint(Int(n), z) :
                                _En_expand_origin_posint(n, z)
        else
            _En_expand_origin_general(ν, z)
        end
        return expscaled ? _En_safeexpmult(z, result) : result
    end
    cf, _, _ = _En_cf_nogamma(ν, z)
    return expscaled ? cf : _En_safeexpmult(-z, cf)
end

function _expint_unsafe(ν::Complex{T}, z::Complex{T},
                        expscaled::Bool) where {T<:AbstractFloat}
    if abs2(z) < 9
        result = if isreal(ν) && isinteger(real(ν)) && real(ν) > 0
            n = real(ν)
            n <= typemax(Int) ? _En_expand_origin_posint(Int(n), z) :
                                _En_expand_origin_posint(n, z)
        else
            _En_expand_origin_general(ν, z)
        end
        return expscaled ? _En_safeexpmult(z, result) : result
    elseif real(z) > 0
        cf, _, _ = _En_cf_nogamma(ν, z)
        return expscaled ? cf : _En_safeexpmult(-z, cf)
    end
    return _expint_left_halfplane(ν, z, expscaled)
end

# Incomplete gamma

function _gamma_lower_series(a::T, z::T;
                             maxiter::Union{Nothing,Int}=nothing) where {T}
    if iszero(z)
        real(a) > 0 && return zero(z^a)
        throw(DomainError(z, "the lower incomplete gamma is singular at z = 0"))
    end

    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? _series_iteration_cap(a, z) : maxiter
    term = one(z) / a
    total = term
    stable = 0
    for n = 1:cap
        term *= z / (a + n)
        total += term
        if abs(term) <= tol * abs(total)
            stable += 1
            stable >= 2 &&
                return _En_safeexpmult(a * log(z) - z, total)
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(:lower_incomplete_gamma_series, cap))
end

function _gamma_upper_cf(a::T, z::T) where {T<:AbstractFloat}
    if iszero(z)
        a > 0 && return gamma(a)
        throw(DomainError(z, "the upper incomplete gamma is singular at z = 0"))
    end
    # Γ(a,z) = z^a E_{1-a}(z).
    ν = 1 - a
    scaled, _, _ = _En_cf_nogamma(ν, z)
    return _En_safeexpmult(a * log(z) - z, scaled)
end

function _gamma_upper_cf(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if iszero(z)
        real(a) > 0 && return gamma(a)
        throw(DomainError(z, "the upper incomplete gamma is singular at z = 0"))
    end
    ν = 1 - a
    scaled = if real(z) > 0
        _En_cf_nogamma(ν, z)[1]
    else
        _expint_left_halfplane(ν, z, true)
    end
    return _En_safeexpmult(a * log(z) - z, scaled)
end

function _gamma_lower_series_normalized(
    a::T, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T}
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? _series_iteration_cap(a, z) : maxiter
    term = one(z)
    total = term
    stable = 0
    for n = 1:cap
        term *= z / (a + n)
        total += term
        if abs(term) <= tol * abs(total)
            stable += 1
            if stable >= 2
                exponent = a * log(z) - z
                gamma_argument = a + one(a)
                return _gamma_lower_series_normalized_result(
                    exponent, gamma_argument, total
                )
            end
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(
        :normalized_lower_incomplete_gamma_series, cap
    ))
end

function _gamma_lower_series_normalized_result(
    exponent::T, gamma_argument::T, total::T
) where {T<:AbstractFloat}
    logabs, sign = logabsgamma(gamma_argument)
    return sign * _En_safeexpmult(exponent - logabs, total)
end

function _gamma_lower_series_normalized_result(
    exponent::Complex{T}, gamma_argument::Complex{T}, total::Complex{T}
) where {T<:AbstractFloat}
    if isreal(gamma_argument)
        logabs, sign = logabsgamma(real(gamma_argument))
        return sign * _En_safeexpmult(exponent - logabs, total)
    end
    return _En_safeexpmult(exponent - loggamma(gamma_argument), total)
end

function _gamma_lower_direct(a::T, z::T) where {T<:AbstractFloat}
    if a > 0 && z >= 0
        return z <= a + oftype(a, 0.1)
    end
    absa = abs(a)
    return abs(z) < absa + one(absa)
end

function _gamma_lower_direct(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if isreal(a) && isreal(z) && real(a) > 0 && real(z) >= 0
        return real(z) <= real(a) + oftype(real(a), 0.1)
    end
    absa = abs(a)
    return abs(z) < absa + one(real(absa))
end

function _gamma_upper_unsafe(a::T, z::T) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        lower = _gamma_lower_series(a, z)
        return gamma(a) - lower
    end
    return _gamma_upper_cf(a, z)
end

function _gamma_upper_unsafe(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        lower = _gamma_lower_series(a, z)
        return gamma(a) - lower
    end
    return _gamma_upper_cf(a, z)
end

function _gamma_lower_unsafe(a::T, z::T) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        return _gamma_lower_series(a, z)
    end
    complete = gamma(a)
    upper = _gamma_upper_cf(a, z)
    return complete - upper
end

function _gamma_lower_unsafe(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        return _gamma_lower_series(a, z)
    end
    complete = gamma(a)
    upper = _gamma_upper_cf(a, z)
    return complete - upper
end

_gamma(a::Float16, z::Float16) =
    Float16(_gamma(Float32(a), Float32(z)))
_gamma(a::ComplexF16, z::ComplexF16) =
    ComplexF16(_gamma(ComplexF32(a), ComplexF32(z)))

function _gamma(a::T, z::T) where {T<:Union{Float32,Float64}}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    isfinite(a) && a <= 0 && isinteger(a) && return _gamma_upper_cf(a, z)
    return _gamma_upper_unsafe(a, z)
end

function _gamma(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float32,Float64}
}
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        return _gamma_upper_cf(a, z)
    end
    return _gamma_upper_unsafe(a, z)
end

function _gamma(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
    ))
    isfinite(a) && a <= 0 && isinteger(a) && return _gamma_upper_cf(a, z)
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        value = _gamma_upper_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(value)
        end
    end
end

function _gamma(a::Complex{BigFloat}, z::Complex{BigFloat})
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        return _gamma_upper_cf(a, z)
    end
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        value = _gamma_upper_unsafe(ahi, zhi)
        setprecision(p) do
            Complex{BigFloat}(BigFloat(real(value)), BigFloat(imag(value)))
        end
    end
end

_gamma_lower(a::Float16, z::Float16) =
    Float16(_gamma_lower(Float32(a), Float32(z)))
_gamma_lower(a::ComplexF16, z::ComplexF16) =
    ComplexF16(_gamma_lower(ComplexF32(a), ComplexF32(z)))

function _gamma_lower(a::T, z::T) where {T<:Union{Float32,Float64}}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    isfinite(a) && a <= 0 && isinteger(a) &&
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    return _gamma_lower_unsafe(a, z)
end

function _gamma_lower(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float32,Float64}
}
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    end
    return _gamma_lower_unsafe(a, z)
end

function _gamma_lower(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    isfinite(a) && a <= 0 && isinteger(a) &&
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        value = _gamma_lower_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(value)
        end
    end
end

function _gamma_lower(a::Complex{BigFloat}, z::Complex{BigFloat})
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    end
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        value = _gamma_lower_unsafe(ahi, zhi)
        setprecision(p) do
            Complex{BigFloat}(BigFloat(real(value)), BigFloat(imag(value)))
        end
    end
end

function _gamma_inc_unsafe(a::T, z::T) where {T<:AbstractFloat}
    if iszero(z)
        a > 0 && return zero(z), one(z)
        throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    if _gamma_lower_direct(a, z)
        p = _gamma_lower_series_normalized(a, z)
        return p, one(p) - p
    end
    ν = 1 - a
    scaled, _, _ = _En_cf_nogamma(ν, z)
    exponent = a * log(z) - z
    return _gamma_inc_from_upper(a, scaled, exponent)
end

function _gamma_inc_unsafe(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if iszero(z)
        real(a) > 0 && return zero(z), one(z)
        throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    if _gamma_lower_direct(a, z)
        p = _gamma_lower_series_normalized(a, z)
        return p, one(p) - p
    end
    ν = 1 - a
    scaled = if real(z) > 0
        _En_cf_nogamma(ν, z)[1]
    else
        _expint_left_halfplane(ν, z, true)
    end
    exponent = a * log(z) - z
    return _gamma_inc_from_upper(a, scaled, exponent)
end

function _gamma_inc_from_upper(a::Float64, scaled::Float64,
                               exponent::Float64)
    if 0 < a <= 11.5
        denominator = gamma(a)
        numerator = exp(exponent)
        if isfinite(denominator) && !iszero(denominator) &&
           isfinite(numerator) && !iszero(numerator)
            q = numerator * scaled / denominator
            isfinite(q) && !iszero(q) && return one(q) - q, q
        end
    end
    logabs, sign = logabsgamma(a)
    q = sign * _En_safeexpmult(exponent - logabs, scaled)
    return one(q) - q, q
end

function _gamma_inc_from_upper(a::T, scaled::T,
                               exponent::T) where {T<:AbstractFloat}
    logabs, sign = logabsgamma(a)
    q = sign * _En_safeexpmult(exponent - logabs, scaled)
    return one(q) - q, q
end

function _gamma_inc_from_upper(a::ComplexF64, scaled::ComplexF64,
                               exponent::ComplexF64)
    if isreal(a) && 0 < real(a) <= 11.5
        denominator = gamma(real(a))
        numerator = exp(exponent)
        if isfinite(denominator) && !iszero(denominator) &&
           isfinite(numerator) && !iszero(numerator)
            q = numerator * scaled / denominator
            isfinite(q) && !iszero(q) && return one(q) - q, q
        end
    end
    q = if isreal(a)
        logabs, sign = logabsgamma(real(a))
        sign * _En_safeexpmult(exponent - logabs, scaled)
    else
        _En_safeexpmult(exponent - loggamma(a), scaled)
    end
    return one(q) - q, q
end

function _gamma_inc_from_upper(
    a::Complex{T}, scaled::Complex{T}, exponent::Complex{T}
) where {T<:AbstractFloat}
    q = if isreal(a)
        logabs, sign = logabsgamma(real(a))
        sign * _En_safeexpmult(exponent - logabs, scaled)
    else
        _En_safeexpmult(exponent - loggamma(a), scaled)
    end
    return one(q) - q, q
end

function _gamma_inc(a::Float16, z::Float16)
    p, q = _gamma_inc(Float32(a), Float32(z))
    return Float16(p), Float16(q)
end

function _gamma_inc(a::ComplexF16, z::ComplexF16)
    p, q = _gamma_inc(ComplexF32(a), ComplexF32(z))
    return ComplexF16(p), ComplexF16(q)
end

function _gamma_inc(a::T, z::T) where {T<:Union{Float32,Float64}}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    isfinite(a) && a <= 0 && isinteger(a) &&
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    return _gamma_inc_unsafe(a, z)
end

function _gamma_inc(a::Complex{T}, z::Complex{T}) where {
    T<:Union{Float32,Float64}
}
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    end
    return _gamma_inc_unsafe(a, z)
end

function _gamma_inc(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    isfinite(a) && a <= 0 && isinteger(a) &&
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        lower, upper = _gamma_inc_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(lower), BigFloat(upper)
        end
    end
end

function _gamma_inc(a::Complex{BigFloat}, z::Complex{BigFloat})
    if isreal(a) && isfinite(real(a)) &&
       real(a) <= 0 && isinteger(real(a))
        throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
    end
    p = precision(BigFloat)
    guard = max(32, ndigits(p; base=2) + 16)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        lower, upper = _gamma_inc_unsafe(ahi, zhi)
        setprecision(p) do
            (
                Complex{BigFloat}(
                    BigFloat(real(lower)), BigFloat(imag(lower))
                ),
                Complex{BigFloat}(
                    BigFloat(real(upper)), BigFloat(imag(upper))
                ),
            )
        end
    end
end
