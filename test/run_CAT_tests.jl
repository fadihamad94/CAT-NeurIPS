using Test, NLPModels, NLPModelsJuMP, JuMP, LinearAlgebra, DataFrames, SparseArrays

include("../src/CAT.jl")
include("./test_TRS_solver.jl")

function test_phi_negative_one()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == -1
end

function test_phi_zero()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = 0.4
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
end

function test_phi_positive_one()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = problem.r_1
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 1
end

function test_find_interval_with_both_phi_zero_starting_from_phi_zero()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    @test δ == δ_prime == 64.0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == 0
end

function test_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    @test δ == δ_prime == 8.0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == 0
end

function test_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 9.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    @test δ == δ_prime == 9.0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == 0
end

function test_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 250.0
    ϵ = 0.8
    r = 0.3
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    @test (δ, δ_prime) == (500.0, 250.0)
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 1
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == -1
end

function test_bisection_with_starting_on_root_δ_zero()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 64.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    δ_m = consistently_adaptive_trust_region_method.bisection(g, H, δ, ϵ, δ_prime, r)
    # @test δ_m == δ == δ_prime == 0
    @test δ_m == δ == δ_prime
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_m, ϵ, r) == 0
end

function test_bisection_with_starting_on_root_δ_not_zero()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [0.0, 0.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 0.0
    ϵ = 0.8
    r = 0.2
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    δ_m = consistently_adaptive_trust_region_method.bisection(g, H, δ, ϵ, δ_prime, r)
    @test δ_m == δ == δ_prime == 8.0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == 0
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_m, ϵ, r) == 0
end

function test_bisection_with_starting_from_negative_one_and_positive_one()
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    g = grad(nlp, x_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    δ = 250.0
    ϵ = 0.8
    r = 0.3
    δ, δ_prime = consistently_adaptive_trust_region_method.findinterval(g, H, δ, ϵ, r)
    δ_m = consistently_adaptive_trust_region_method.bisection(g, H, δ, ϵ, δ_prime, r)
    @test δ_m == 406.25
    @test consistently_adaptive_trust_region_method.phi(g, H, δ, ϵ, r) == 1
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_prime, ϵ, r) == -1
    @test consistently_adaptive_trust_region_method.phi(g, H, δ_m, ϵ, r) == 0
end

function test_restore_full_matrix_two_by_two()
    A = [1330.0 0.0; 480.0 200.0]
    B = consistently_adaptive_trust_region_method.restoreFullMatrix(A)
    @test B == [1330.0 480.0; 480.0 200.0]
end

function test_restore_full_matrix_three_by_three()
    A = [1330.0 0.0 0.0; 480.0 200.0 0.0; 250.0 230.0 180.0]
    B = consistently_adaptive_trust_region_method.restoreFullMatrix(A)
    @test B == [1330.0 480.0 250.0; 480.0 200.0 230.0; 250.0 230.0 180.0]
end

function test_restore_full_matrix_four_by_four()
    A = [1330.0 0.0 0.0 0.0; 480.0 200.0 100.0 0.0; 250.0 230.0 180.0 0.0; 100.0 200.0 300.0 400.0]
    B = consistently_adaptive_trust_region_method.restoreFullMatrix(A)
    @test B == [1330.0 480.0 250.0 100.0; 480.0 200.0 230.0 200.0; 250.0 230.0 180.0 300.0; 100.0 200.0 300.0 400.0]
end

function test_restore_full_matrix_diagonal_matrix()
    A = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]
    B = consistently_adaptive_trust_region_method.restoreFullMatrix(A)
    @test B == A
end

function test_compute_second_order_model_negative_direction()
    problem = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [-1.0, -1.0]
    nlp = problem.nlp
    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    second_order_model_value = consistently_adaptive_trust_region_method.computeSecondOrderModel(function_value, gradient_value, hessian_value, d_k)
    @test second_order_model_value == 104.0
end

function test_compute_second_order_model_zero_direction()
    problem = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [0.0, 0.0]
    nlp = problem.nlp
    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    second_order_model_value = consistently_adaptive_trust_region_method.computeSecondOrderModel(function_value, gradient_value, hessian_value, d_k)
    @test second_order_model_value == 1.0
end

function test_compute_second_order_model_positive_direction()
    problem = test_create_dummy_problem()
    x_k = [0.0, 0.0]
    d_k = [1.0, 1.0]
    nlp = problem.nlp
    function_value = obj(nlp, x_k)
    gradient_value = grad(nlp, x_k)
    hessian_value = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    second_order_model_value = consistently_adaptive_trust_region_method.computeSecondOrderModel(function_value, gradient_value, hessian_value, d_k)
    @test second_order_model_value == 100.0
end

function test_compute_ρ_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = [1.0, 1.0]
    δ = 0.0
    ϵ = 0.2
    r = 0.2
    d_k = [-0.0, -0.0]
    θ = problem.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    ρ = consistently_adaptive_trust_region_method.compute_ρ(fval_current, fval_next, gval_current, gval_next, H, x_k, d_k, θ)
end

function test_compute_ρ_phi_zero()
    tol = 1e-3
    problem = test_create_dummy_problem()
    nlp = problem.nlp
    x_k = nlp.meta.x0
    δ = 0.0
    ϵ = 1.2
    r = 0.2
    d_k = [0.02471910112359557, 0.3806741573033706]
    θ = problem.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    ρ = consistently_adaptive_trust_region_method.compute_ρ(fval_current, fval_next, gval_current, gval_next, H, x_k, d_k, θ)
    @test norm(ρ - 0.980423689675886, 2) <= tol
end

function test_compute_ρ_phi_δ_positive_phi_δ_prime_negative()
    tol = 1e-3
    problem = test_create_dummy_problem2()
    nlp = problem.nlp
    x_k = [0.0, 1.0]
    δ = 250.0
    ϵ = 1.2
    r = 0.2
    d_k = [-0.005830328471736362, 0.34323592199917485]
    θ = problem.θ
    fval_current = obj(nlp, x_k)
    fval_next = obj(nlp, x_k + d_k)
    gval_current = grad(nlp, x_k)
    gval_next = grad(nlp, x_k + d_k)
    H = consistently_adaptive_trust_region_method.restoreFullMatrix(hess(nlp, x_k))
    ρ = consistently_adaptive_trust_region_method.compute_ρ(fval_current, fval_next, gval_current, gval_next, H, x_k, d_k, θ)
    @test norm(ρ - 0.802338318969011, 2) <= tol
end

function unit_tests()
    #Unit test for the ϕ function
    test_phi_negative_one()
    test_phi_zero()
    test_phi_positive_one()

    #Unit test for the find interval function
    test_find_interval_with_both_phi_zero_starting_from_phi_zero()
    test_find_interval_with_both_phi_0_starting_from_phi_negative_one()
    test_find_interval_with_both_phi_0_starting_from_phi_positive_one()
    test_find_interval_with_phi_δ_positive_one_phi_δ_prime_negative_one()

    #Unit test for the bisection function
    test_bisection_with_starting_on_root_δ_zero()
    test_bisection_with_starting_on_root_δ_not_zero()
    test_bisection_with_starting_from_negative_one_and_positive_one()

    #Unit test for restore full matrix function
    # test_restore_full_matrix_two_by_two()
    # test_restore_full_matrix_three_by_three()
    # test_restore_full_matrix_four_by_four()
    # test_restore_full_matrix_diagonal_matrix()

    #Unit test compute second order model function
    test_compute_second_order_model_negative_direction()
    test_compute_second_order_model_zero_direction()
    test_compute_second_order_model_positive_direction()

    #Unit test compute ρ function
    test_compute_ρ_δ_0_H_positive_semidefinite_starting_on_global_minimizer()
    test_compute_ρ_phi_zero()
    test_compute_ρ_phi_δ_positive_phi_δ_prime_negative()
end

function solve_NLP1_starting_at_global_optimum()
    problem = test_create_dummy_problem()
    x = [1.0, 1.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test x == [1.0, 1.0]
    @test obj(problem.nlp, x) == 0.0
    @test status == "SUCCESS"
end

function solveSimpleConvexNLPModel()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x - [0.41666666666666663, 0.5833333333333334], 2) <= 0
    @test norm(obj(problem.nlp, x) - 0, 2) <= tol
    @test status == "SUCCESS"
end

function solveComplexConvexNLPModel()
    tol = 1e-3
    problem = test_create_dummy_problem()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x[1] - 1, 2) <= tol
    @test norm(x[2] - 1, 2) <= tol
    @test norm(obj(problem.nlp, x) - 0, 2) <= tol
    @test status == "SUCCESS"
end

function solveSimpleConvexNLPModelDifferentStartingPoint()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [0.1, 0.1]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x - [0.45499999999999996, 0.545], 2) <= tol
    @test norm(obj(problem.nlp, x) - 0.0, 2) <= tol
    @test status == "SUCCESS"
end

function solveSimpleConvexNLPModelAnotherStartingPoint()
    tol = 1e-3
    problem = test_create_simple_convex_nlp_model()
    x = [20.01, -10.01]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(x - [15.873020752067404, -14.873020752067404], 2) <= tol
    @test norm(obj(problem.nlp, x) - 0.0, 2) <= tol
    @test status == "SUCCESS"
end

function solveComplexConvexNLP1()
    tol = 1e-3
    problem = test_create_complex_convex_nlp1_model()
    problem.MAX_ITERATION = 10
    x = [0.0, 0.0]
    δ = 0.0
    x, status , iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.750000000125, 2) <= tol
    @test norm(x[1] - 0.33332500000000004, 2) <= tol
    @test norm(x[2] - 0.166665, 2) <= tol
    @test status == "SUCCESS"
end

function solveComplexNLPModeL1()
    tol = 1e-3
    problem = test_create_complex_nlp_modeL1()
    x = [0.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - 0.183430792966865, 2) <= tol
    @test norm(x[1] - 0.7221896985843893, 2) <= tol
    @test norm(x[2] - (-0.5819243669997765), 2) <= tol
    @test status == "SUCCESS"
end

function solveNLPSinCosModel1()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    problem.gradient_termination_tolerance = 2e-2
    x = [0.0, 0.0]
    δ = 0.049
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == "SUCCESS"
end

function solveNLPSinCosModel1DifferentStartingPoint()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    x = [10.0, 0.0]
    δ = 0.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == "SUCCESS"
end

function solveNLPSinCosModel1DeltaNotZero()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp1()
    x = [0.0, 0.0]
    δ = 1.0
    x, status = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) + 1, 2) <= tol
    @test status == "SUCCESS"
end

function solveNLPSinCosModel2()
    tol = 1e-3
    problem = test_create_problem_sin_cos_mode_nlp2()
    problem.MAX_ITERATION = 1000
    x = [10.0, 10.0]
    δ = 1.0
    x, status, iteration_stats = consistently_adaptive_trust_region_method.CAT(problem, x, δ)
    @test norm(obj(problem.nlp, x) - (-2), 2) <= tol
    @test norm(x[1] - 10.995653476776056, 2) <= tol
    @test norm(x[2] - 9.424777960768635, 2) <= tol
    @test status == "SUCCESS"
end

function optimize_models()
    # println("--------------------------------------------------------------")
    # println("-----TESTING SOLVING NLP STARTING AT THE GLOBAL MINIMIZER-----")
    # println("--------------------------------------------------------------")
    # println()
    solve_NLP1_starting_at_global_optimum()
    # println()
    # println("---------------------------------------------------------------")
    # println("------------TESTING SOLVING SIMPLE CONVEX NLP MODEL------------")
    # println("---------------------------------------------------------------")
    # println()
    solveSimpleConvexNLPModel()
    # println()
    # println("----------------------------------------------------------------")
    # println("------------TESTING SOLVING COMPLEX CONVEX NLP MODEL------------")
    # println("----------------------------------------------------------------")
    # println()
    solveComplexConvexNLPModel()
    # println()
    # println("-------------------------------------------------------------------")
    # println("-TESTING SOLVING COMPLEX CONVEX NLP MODEL DIFFERENT STARTING POINT-")
    # println("-------------------------------------------------------------------")
    # println()
    solveSimpleConvexNLPModelDifferentStartingPoint()
    # println()
    # println("-------------------------------------------------------------------")
    # println("--TESTING SOLVING COMPLEX CONVEX NLP MODEL ANOTHER STARTING POINT--")
    # println("-------------------------------------------------------------------")
    # println()
    solveSimpleConvexNLPModelAnotherStartingPoint()
    # println()
    # println("----------------------------------------------------------------")
    # println("------------TESTING SOLVING COMPLEX CONVEX NLP MODEL------------")
    # println("----------------------------------------------------------------")
    # println()
    solveComplexConvexNLP1()
    # println()
    # println("---------------------------------------------------------")
    # println("------------TESTING SOLVING COMPLEX NLP MODEL------------")
    # println("---------------------------------------------------------")
    # println()
    solveComplexNLPModeL1()
    # println()
    # println("---------------------------------------------------------")
    # println("------------TESTING SOLVING SIN COS NLP MODEL------------")
    # println("---------------------------------------------------------")
    # println()
    solveNLPSinCosModel1()
    # println("-------------------------------------------------------------")
    # println("-TESTING SOLVING SIN COS NLP MODEL  DIFFERENT STARTING POINT-")
    # println("-------------------------------------------------------------")
    # println()
    solveNLPSinCosModel1DifferentStartingPoint()
    # println()
    # println("----------------------------------------------------------")
    # println("-----TESTING SOLVING SIN COS NLP MODEL DELTA NOT ZERO-----")
    # println("----------------------------------------------------------")
    # println()
    solveNLPSinCosModel1DeltaNotZero()
    # println()
    # println("---------------------------------------------------------")
    # println("------------TESTING SOLVING SIN COS NLP MODEL------------")
    # println("---------------------------------------------------------")
    # println()
    solveNLPSinCosModel2()
end


@testset "basic_CAT_tests" begin
    unit_tests()
end

@testset "optimization_CAT_tests" begin
    optimize_models()
end
