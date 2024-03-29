export phi, findinterval, bisection
#=
The big picture idea here is to optimize the trust region subproblem using a factorization method based
on the optimality conditions:
H d_k + g + δ d_k = 0
H + δ I ≥ 0
δ(r -  ||d_k ||) = 0

That is why we defined the below phi to solve that using bisection logic.
=#

function getHessianLowerTriangularPart(H)
	h_vec = Vector{Float64}()
	for i in 1:size(H)[1]
		for j in 1:i
			push!(h_vec, H[i, j])
		end
	end
	return h_vec
end

function solveTrustRegionSubproblem(f::Float64, g::Vector{Float64}, H, x_k::Vector{Float64}, δ::Float64, ϵ::Float64, r::Float64)
	return optimizeSecondOrderModel(g, H, δ, ϵ, r)
end

#Based on Theorem 4.3 in Numerical Optimization by Wright
function optimizeSecondOrderModel(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    #When δ is 0 and the Hessian is positive semidefinite, we can directly compute the direction
    try
        cholesky(Matrix(H))
        d_k = H \ (-g)
        if norm(d_k, 2) <= r
            return 0.0, d_k
        end
    catch e
		#Do nothing
    end

    try
	    δ, δ_prime = findinterval(g, H, δ, ϵ, r)
        δ_m = bisection(g, H, δ, ϵ, δ_prime, r)
        sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
        d_k = (cholesky(H + δ_m * sparse_identity) \ (-g))
        return δ_m, d_k
    catch e
        if e == ErrorException("Bisection logic failed to find a root for the phi function")
	    	δ, d_k = solveHardCaseLogic(g, H, r)
            return δ, d_k
        elseif e == ErrorException("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0.")
            δ, d_k = solveHardCaseLogic(g, H, r)
	    	return δ, d_k
        else
            throw(e)
        end
    end
end

function phi(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    sparse_identity = SparseMatrixCSC{Float64}(LinearAlgebra.I, size(H)[1], size(H)[2])
    shifted_hessian = H + δ * sparse_identity
    #cholesky factorization only works on positive definite matrices
    try
        cholesky(shifted_hessian)
        computed_norm = norm(shifted_hessian \ g, 2)
	if computed_norm < ϵ * r
        return 1
	elseif computed_norm <= r
        return 0
    else
        return -1
    end
    catch e
        return -1
    end
end

function findinterval(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, r::Float64)
    Φ_δ = phi(g, H, 0.0, ϵ, r)

    if Φ_δ == 0
        δ = 0.0
        δ_prime = 0.0
        return δ, δ_prime
    end

    Φ_δ = phi(g, H, δ, ϵ, r)

    if Φ_δ == 0
        δ_prime = δ
        return δ, δ_prime
    end

    δ_prime = δ == 0.0 ? 1.0 : δ * 2
    Φ_δ_prime = 0.0

    k = 1
    while k < 100
        Φ_δ_prime = phi(g, H, δ_prime, ϵ, r)
        if Φ_δ_prime == 0
            δ = δ_prime
            return δ, δ_prime
        end

        if ((Φ_δ * Φ_δ_prime) < 0)
            break
        end
        if Φ_δ_prime > 0
            δ_prime = δ_prime / 2
        elseif Φ_δ_prime < 0
            δ_prime = δ_prime * 2
        end
        k = k + 1
    end

    #switch so that δ for ϕ_δ >= 0 and δ_prime for ϕ_δ_prime <= 0
    if Φ_δ_prime > 0 && Φ_δ < 0
        δ_temp = δ
        Φ_δ_temp = Φ_δ
        δ = δ_prime
        δ_prime = δ_temp
        Φ_δ = Φ_δ_prime
        Φ_δ_prime = Φ_δ_temp
    end

    if (Φ_δ  * Φ_δ_prime > 0)
        throw(error("Bisection logic failed to find a pair δ and δ_prime such that ϕ(δ) >= 0 and ϕ(δ_prime) <= 0."))
    end
    return δ, δ_prime
end

function bisection(g::Vector{Float64}, H, δ::Float64, ϵ::Float64, δ_prime::Float64, r::Float64)
    # the input of the function is the two end of the interval (δ,δ_prime)
    # our goal here is to find the approximate δ using classic bisection method

    #Bisection logic
    k = 1
    δ_m = (δ + δ_prime) / 2
    Φ_δ_m = phi(g, H, δ_m, ϵ, r)

    while (Φ_δ_m != 0) && k <= 100
        if Φ_δ_m > 0
            δ = δ_m
        else
            δ_prime = δ_m
        end
        δ_m = (δ + δ_prime) / 2
        Φ_δ_m = phi(g, H, δ_m, ϵ, r)
        k = k + 1
    end

    if (Φ_δ_m != 0)
        throw(error("Bisection logic failed to find a root for the phi function"))
    end
    return δ_m
end

#Based on 'THE HARD CASE' section from Numerical Optimization by Nocedal and Wright
function solveHardCaseLogic(g::Vector{Float64}, H, r::Float64)
    minimumEigenValue = eigmin(Matrix(H))
    δ = -minimumEigenValue
    z =  eigvecs(Matrix(H))[:,1]
    Q = eigvecs(Matrix(H))
    eigenvaluesVector = eigvals(Matrix(H))

    D = zeros(size(Q))
    for i in 1:size(D)[1]
        D[i, i] = eigenvaluesVector[i]
    end

    norm_d_k_squared_without_τ_squared = 0.0

    for i in 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            norm_d_k_squared_without_τ_squared = norm_d_k_squared_without_τ_squared + ((Q[:, i]' * g) ^ 2 / (eigenvaluesVector[i] + δ) ^ 2)
        end
    end

    norm_d_k_squared = r ^ 2
    τ = sqrt(norm_d_k_squared - norm_d_k_squared_without_τ_squared)
    d_k = τ .* z

    for i in 1:length(eigenvaluesVector)
        if eigenvaluesVector[i] != minimumEigenValue
            d_k = d_k .- ((Q[:, i]' * g) / (eigenvaluesVector[i] + δ)) * Q[:, i]
        end
    end

    return δ, d_k
end
