using Test, NLPModels, NLPModelsJuMP, JuMP, LinearAlgebra

include("../src/CAT.jl")
include("./create_tests.jl")
include("../src/trust_region_subproblem_solver.jl")

#Unit test optimize second order model function
function test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [1.0, 1.0]
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test d_k == [-0.0, -0.0]
    @test δ_k == δ
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_zero()
    tol = 1e-3
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k - [0.10666201604464587, 0.13940239507034077], 2) <= tol
    @test norm(δ_k - 64.0, 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    δ = 250.0
    ϵ = 0.8
    r = 0.2
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k - [-0.0032288617186154635, 0.17943860857585672], 2) <= tol
    @test norm(δ_k - 500.0, 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
end

function test_optimize_second_order_model_for_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    ϵ = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test ϵ * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test ϵ * r <= norm(d_k) <= r
    @test norm((x_k + d_k) - [0.5], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - 0.25, 2) <= tol
end

function test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    tol = 1e-3
    problem = test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    ϵ = 0.8
    r = 2.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k) <= r
    @test norm((x_k + d_k) - [1.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - 0, 2) <= tol
end

function test_optimize_second_order_model_for_simple_bivariate_convex_model()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    ϵ = 0.8
    r = 0.5
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test ϵ * r <= norm((H + δ_k * I) \ g, 2) <= r
    @test ϵ * r <= norm(d_k) <= r
    @test norm((x_k + d_k) - [0.3333333333333333, 0.33333333333333337], 2) <= tol
    @test δ_k == 2.0
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - 0.11111111111111106, 2) <= tol
end

function  test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_univariate_convex_model()
    nlp = problem.nlp
    x_k = [0.0]
    δ = 0.0
    ϵ = 0.8
    r = 1.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k) <= r
    @test δ_k == 2.0
    @test norm((x_k + d_k) - [1.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-1), 2) <= tol
end

function  test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    tol = 1e-3
    problem = test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    ϵ = 0.8
    r = 1.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k) <= r
    @test δ_k == 2.0
    @test norm((x_k + d_k) - [1.0, 0.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-1), 2) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_1()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    ϵ = 0.8
    r = 1.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k) <= r
    @test δ_k == 4.0
    @test norm((x_k + d_k) - [0.0, 1.0], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-2), 2) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    tol = 1e-2
    problem = test_create_hard_case_using_bivariate_convex_model_2()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    ϵ = 0.8
    r = 1.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k, 2) - r <= tol
    @test δ_k == 2.0
    @test norm((x_k + d_k) - [0.0025, 1.002496874995117], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-1), 2) <= tol
end

function test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
    tol = 1e-3
    problem = test_create_hard_case_using_bivariate_convex_model_3()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    δ = 0.0
    ϵ = 0.8
    r = 5.0
    g = grad(nlp, x_k)
    H = hess(nlp, x_k)
    δ_k, d_k = consistently_adaptive_trust_region_method.optimizeSecondOrderModel(g, H, δ, ϵ, r)
    @test norm(d_k) <= r
    @test δ_k == 8.0
    @test norm((x_k + d_k) - [-3.5355339059327373, -3.5355339059327373], 2) <= tol
    @test obj(nlp, x_k + d_k) <= obj(nlp, x_k)
    @test norm(obj(nlp, x_k + d_k) - (-100), 2) <= tol
end

function optimize_models()
    test_optimize_second_order_model_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_optimize_second_order_model_phi_zero()
    test_optimize_second_order_model_phi_δ_positive_phi_δ_prime_negative()
    test_optimize_second_order_model_for_simple_univariate_convex_model()
    test_optimize_second_order_model_for_simple_univariate_convex_model_solved_same_as_Newton()
    test_optimize_second_order_model_for_simple_bivariate_convex_model()
    test_optimize_second_order_model_hard_case_using_simple_univariate_convex_model()
    test_optimize_second_order_model_hard_case_using_simple_bivariate_convex_model()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_1()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_2()
    test_optimize_second_order_model_hard_case_using_bivariate_convex_model_3()
end

@testset "TRS_Solver_Tests" begin
    optimize_models()
end
