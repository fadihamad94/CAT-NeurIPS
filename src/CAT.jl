__precompile__()
#Main algorithm code goes here
module consistently_adaptive_trust_region_method
using NLPModels, LinearAlgebra, DataFrames, SparseArrays
include("./trust_region_subproblem_solver.jl")

export Problem_Data
export phi, findinterval, bisection, restoreFullMatrix, computeSecondOrderModel, optimizeSecondOrderModel, compute_ρ, CAT

mutable struct Problem_Data
    nlp::AbstractNLPModel
    β::Float64
    θ::Float64
    ω::Float64
    r_1::Float64
    MAX_ITERATION::Int64
    gradient_termination_tolerance::Float64
    MAX_TIME::Float64
    γ_2::Float64

    # initialize parameters
    function Problem_Data(nlp::AbstractNLPModel, β::Float64=0.1,
                           θ::Float64=0.1, ω::Float64=8.0, r_1::Float64=1.0,
                           MAX_ITERATION::Int64=10000, gradient_termination_tolerance::Float64=1e-5,
                           MAX_TIME::Float64=30 * 60.0, γ_2::Float64=0.8)
        @assert(β > 0 && β < 1)
        @assert(θ >= 0 && θ < 1)
        @assert(β * θ < 1 - β)
        @assert(ω > 1)
        @assert(r_1 > 0)
        @assert(MAX_ITERATION > 0)
        @assert(MAX_TIME > 0)
        @assert(γ_2 > (1 / ω) && γ_2 <= 1)
        return new(nlp, β, θ, ω, r_1, MAX_ITERATION, gradient_termination_tolerance, MAX_TIME, γ_2)
    end
end

function computeSecondOrderModel(f::Float64, g::Vector{Float64}, H, d_k::Vector{Float64})
    return f + transpose(g) * d_k + 0.5 * transpose(d_k) * H * d_k
end

function compute_ρ(fval_current::Float64, fval_next::Float64, gval_current::Vector{Float64}, gval_next::Vector{Float64}, H, x_k::Vector{Float64}, d_k::Vector{Float64}, θ::Float64)
    second_order_model_value_current_iterate = computeSecondOrderModel(fval_current, gval_current, H, d_k)
    guarantee_factor = θ * 0.5 * norm(gval_next, 2) * norm(d_k, 2)
    ρ = (fval_current - fval_next) / (fval_current - second_order_model_value_current_iterate + guarantee_factor)
    return ρ
end

function CAT(problem::Problem_Data, x::Vector{Float64}, δ::Float64, subproblem_solver_method::String=subproblem_solver_methods.OPTIMIZATION_METHOD_DEFAULT)
    @assert(δ >= 0)
    MAX_ITERATION = problem.MAX_ITERATION
    MAX_TIME = problem.MAX_TIME
    gradient_termination_tolerance = problem.gradient_termination_tolerance
    β = problem.β
    ω = problem.ω
    x_k = x
    δ_k = δ
    r_k = problem.r_1
    γ_2 = problem.γ_2
    nlp = problem.nlp
    θ = problem.θ
    iteration_stats = DataFrame(k = [], deltaval = [], directionval = [], fval = [], gradval = [])
    total_function_evaluation = 0
    total_gradient_evaluation = 0
    total_hessian_evaluation = 0

    k = 1
    try
        gval_current = grad(nlp, x_k)
        fval_current = obj(nlp, x_k)
        total_function_evaluation += 1
        total_gradient_evaluation += 1
        hessian_current = nothing
        compute_hessian = true

        if norm(gval_current, 2) <= gradient_termination_tolerance
            computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
            println("*********************************Iteration Count: ", 1)
            push!(iteration_stats, (1, δ, [], fval_current, norm(gval_current, 2)))
            return x_k, "SUCCESS", iteration_stats, computation_stats, 1
        end
        start_time = time()
        while k <= MAX_ITERATION
            @show "Iteration $k, $(norm(gval_current, 2))"
            if compute_hessian
                hessian_current = restoreFullMatrix(hess(nlp, x_k))
                total_hessian_evaluation += 1
            end
            δ_k, d_k = solveTrustRegionSubproblem(fval_current, gval_current, hessian_current, x_k, δ_k, γ_2, r_k, subproblem_solver_method)
            fval_next = obj(nlp, x_k + d_k)
            total_function_evaluation += 1
            gval_next = grad(nlp, x_k + d_k)
            total_gradient_evaluation += 1
            ρ_k = compute_ρ(fval_current, fval_next, gval_current, gval_next, hessian_current, x_k, d_k, θ)

            if fval_next <= fval_current
                x_k = x_k + d_k
                fval_current = fval_next
                gval_current = gval_next
                compute_hessian = true
            else
                #else x_k+1 = x_k, fval_current, gval_current, hessian_current will not change
                compute_hessian = false
            end
            if ρ_k <= β
                r_k = norm(d_k, 2) / ω
            else
                r_k = ω * norm(d_k, 2)
            end
            push!(iteration_stats, (k, δ_k, d_k, fval_current, norm(gval_current, 2)))
            if norm(gval_next, 2) <= gradient_termination_tolerance
                push!(iteration_stats, (k, δ_k, d_k, fval_next, norm(gval_next, 2)))
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
                println("*********************************Iteration Count: ", k)
                return x_k, "SUCCESS", iteration_stats, computation_stats, k
            end

            if time() - start_time > MAX_TIME
                computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
                return x_k, "MAX_TIME", iteration_stats, computation_stats, k
            end

            k += 1
        end
    catch e
        @warn e
        computation_stats = Dict("total_function_evaluation" => (MAX_ITERATION + 1), "total_gradient_evaluation" => (MAX_ITERATION + 1), "total_hessian_evaluation" => total_hessian_evaluation)
        return x_k, "FAILURE", iteration_stats, computation_stats, (MAX_ITERATION + 1)
    end
    computation_stats = Dict("total_function_evaluation" => total_function_evaluation, "total_gradient_evaluation" => total_gradient_evaluation, "total_hessian_evaluation" => total_hessian_evaluation)
    return x_k, "ITERARION_LIMIT", iteration_stats, computation_stats, (MAX_ITERATION + 1)
end

end # module
