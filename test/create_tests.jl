#Functions to create NLP models

function createDummyNLPModel()
    x0 = [-1.2; 1.0]
    model = Model()
    @variable(model, x[i=1:2], start=x0[i])
    @NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createDummyNLPModel2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1) ^ 2 + x + y + (x ^ 2 - 2 * y ^ 2) ^ 3)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, (x - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSimpleConvexNLPModeL()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexConvexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (x + y - 1) ^ 2 + x + y + (x - 2 * y) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createComplexNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (2 * x + y - 1) ^ 2 + x + y + (x ^ 2 - 2 * y ^ 2) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSinCosNLPModeL1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) * cos(y))
    nlp = MathOptNLPModel(model)
    return nlp
end

function createSinCosNLPModeL2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, sin(x) + cos(y))
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, -x ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - 2 * y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 + 0.01 * x - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 - 10 * x * y + y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end


function createSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, (x - 1) ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleUnivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @NLobjective(model, Min, -x ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem1()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, -x ^ 2 - 2 * y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem2()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 + 0.01 * x - y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function createHardCaseUsingSimpleBivariateConvexProblem3()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, x ^ 2 - 10 * x * y + y ^ 2)
    nlp = MathOptNLPModel(model)
    return nlp
end

function test_create_dummy_problem()
    nlp = createDummyNLPModel()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_dummy_problem2()
    nlp = createDummyNLPModel2()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_convex_nlp_model()
    nlp = createSimpleConvexNLPModeL()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_complex_convex_nlp1_model()
    nlp = createComplexConvexNLPModeL1()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_complex_nlp_modeL1()
    nlp = createComplexNLPModeL1()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_problem_sin_cos_mode_nlp1()
    nlp = createSinCosNLPModeL1()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_problem_sin_cos_mode_nlp2()
    nlp = createSinCosNLPModeL2()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_univariate_convex_model()
    nlp = createSimpleUnivariateConvexProblem()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 0.5, 100, 1e-4)
    return problem
end

function test_create_simple_univariate_convex_model_solved_same_as_Newton()
    nlp = createSimpleUnivariateConvexProblem()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 2.0, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_simple_univariate_convex_model()
    nlp = createHardCaseUsingSimpleUnivariateConvexProblem()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_simple_bivariate_convex_model()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_1()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem1()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_2()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem2()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 1.00, 100, 1e-4)
    return problem
end

function test_create_hard_case_using_bivariate_convex_model_3()
    nlp = createHardCaseUsingSimpleBivariateConvexProblem3()
    problem = consistently_adaptive_trust_region_method.Problem_Data(nlp, 0.25, 0.5, 2.0, 5.00, 100, 1e-4)
    return problem
end
