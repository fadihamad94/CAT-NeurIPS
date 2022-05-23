using CSV, Plots, DataFrames

const FLAT_FACTORIZATION = "FLAT_FACTORIZATION"
const FLAT_THETA_ZERO_FACTORIZATION = "FLAT_THETA_ZERO_FACTORIZATION"
const ARC_G_RULE = "ARC_G_RULE"
const NewtonTrustRegion = "NewtonTrustRegion"

const FLAT_FACTORIZATION_COLOR = :black
const FLAT_THETA_ZERO_FACTORIZATION_COLOR = :red
const ARC_G_RULE_COLOR = :orange
const NewtonTrustRegion_COLOR = :purple

const TOTAL_ITERATIONS = [Int(10 * i) for i in 1:(10000/10)]
const TOTAL_GRADIENTS  = [Int(10 * i) for i in 1:(10000/10)]

function readFile(fileName::String)
    df = DataFrame(CSV.File(fileName))
    return df
end

function filterRows(total_iterations_max::Int64, iterations_vector::Vector{Int64})
    return filter!(x->x < total_iterations_max, iterations_vector)
end

function computeFraction(df::DataFrame, TOTAL::Vector{Int64}, criteria::String)
    total_number_problems = size(df)[1]

    if criteria == "Iterations"
        results_fraction = DataFrame(Iterations=Int[], FLAT_FACTORIZATION=Float64[], FLAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_G_RULE=Float64[], NewtonTrustRegion=Float64[])
        results_total = DataFrame(Iterations=Int[], FLAT_FACTORIZATION=Int[], FLAT_THETA_ZERO_FACTORIZATION=Int[], ARC_G_RULE=Int[], NewtonTrustRegion=Int[])
    else
        results_fraction = DataFrame(Gradients=Int[], FLAT_FACTORIZATION=Float64[], FLAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_G_RULE=Float64[], NewtonTrustRegion=Float64[])
        results_total = DataFrame(Gradients=Int[], FLAT_FACTORIZATION=Int[], FLAT_THETA_ZERO_FACTORIZATION=Int[], ARC_G_RULE=Int[], NewtonTrustRegion=Int[])
    end

    for total in TOTAL
        total_problems_FLAT_FACTORIZATION = length(filterRows(total, df[:, FLAT_FACTORIZATION]))
        total_problems_FLAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, FLAT_THETA_ZERO_FACTORIZATION]))
        total_problems_ARC_G_RULE = length(filterRows(total, df[:, ARC_G_RULE]))
        total_problems_NewtonTrustRegion = length(filterRows(total, df[:, NewtonTrustRegion]))
        push!(results_fraction, (total, total_problems_FLAT_FACTORIZATION / total_number_problems, total_problems_FLAT_THETA_ZERO_FACTORIZATION / total_number_problems, total_problems_ARC_G_RULE / total_number_problems, total_problems_NewtonTrustRegion / total_number_problems))
        push!(results_total, (total, total_problems_FLAT_FACTORIZATION, total_problems_FLAT_THETA_ZERO_FACTORIZATION, total_problems_ARC_G_RULE, total_problems_NewtonTrustRegion))
    end

    return results_fraction
end

function plotFigureComparisonFLAT(df::DataFrame, criteria::String, plot_name::String)
    data = Matrix(df[!, Not(criteria)])
    criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    @show names(df)
    plot(df[!, criteria],
        data,
        label=["Our method default (θ = 0.1)" "Our method (θ = 0.0)"],
        color = [FLAT_FACTORIZATION_COLOR FLAT_THETA_ZERO_FACTORIZATION_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, 10000),
        xaxis=:log10
    )
    dirrectoryName = dirname(Base.source_path())
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresForIterationOld()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_old.png"
    plotFigureOld(results, "Iterations", plot_name)
end

function generateFiguresForIterationAll()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_all.png"
    plotFigureAll(results, "Iterations", plot_name)
end

function generateFiguresForIterationIterative()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> !(x in [FLAT_FACTORIZATION,FLAT_THETA_ZERO_FACTORIZATION,ARC_S_RULE,ARC_S_DELTA_RULE,ARC_FACTORIZATION,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_iterative.png"
    plotFigureIterative(results, "Iterations", plot_name)
end

function generateFiguresForIterationDirect()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> !(x in [STRM,FLAT_THETA_ZERO_FACTORIZATION,FLAT_ITERATIVE,ARC_G_RULE,ARC_S_RULE,ARC_S_DELTA_RULE,ARC_FACTORIZATION,KrylovTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_direct.png"
    plotFigureDirect(results, "Iterations", plot_name)
end

function generateFiguresForGradientOld()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_old.png"
    plotFigureOld(results, "Gradients", plot_name)
end

function generateFiguresForGradientAll()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_all.png"
    plotFigureAll(results, "Gradients", plot_name)
end

function generateFiguresForGradientIterative()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    results = results[:, filter(x -> !(x in [FLAT_FACTORIZATION,FLAT_THETA_ZERO_FACTORIZATION,ARC_S_RULE,ARC_S_DELTA_RULE,ARC_FACTORIZATION,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_iterative.png"
    plotFigureIterative(results, "Gradients", plot_name)
end

function generateFiguresForGradientDirect()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    results = results[:, filter(x -> !(x in [STRM,FLAT_THETA_ZERO_FACTORIZATION,FLAT_ITERATIVE,ARC_G_RULE,ARC_S_RULE,ARC_S_DELTA_RULE,ARC_FACTORIZATION,KrylovTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_direct.png"
    plotFigureDirect(results, "Gradients", plot_name)
end

function generateFiguresComparisonFLAT()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", FLAT_FACTORIZATION,FLAT_THETA_ZERO_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_comparison_FLAT.png"
    plotFigureComparisonFLAT(results, "Iterations", plot_name)
end

function plotFiguresComparisonFinal(df::DataFrame, criteria::String, plot_name::String)
    data = Matrix(df[!, Not(criteria)])
    criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    @show names(df)
    plot(df[!, criteria],
        data,
        label=["Our method" "ARC with g-rule" "Newton trust region"],
        color = [FLAT_FACTORIZATION_COLOR ARC_G_RULE_COLOR NewtonTrustRegion_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, 10000),
        xaxis=:log10
    )
    dirrectoryName = dirname(Base.source_path())
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresIterationsComparisonFinal()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", FLAT_FACTORIZATION,ARC_G_RULE,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_final.png"
    plotFiguresComparisonFinal(results, "Iterations", plot_name)
end

function generateFiguresGradientsComparisonFinal()
    dirrectoryName = dirname(Base.source_path())
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    results = results[:, filter(x -> (x in ["Gradients", FLAT_FACTORIZATION,ARC_G_RULE,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_final.png"
    plotFiguresComparisonFinal(results, "Gradients", plot_name)
end

generateFiguresComparisonFLAT()

generateFiguresIterationsComparisonFinal()

generateFiguresGradientsComparisonFinal()
