using CSV, Plots, DataFrames

const CAT_FACTORIZATION = "CAT_FACTORIZATION"
const CAT_THETA_ZERO_FACTORIZATION = "CAT_THETA_ZERO_FACTORIZATION"
const ARC_G_RULE = "ARC_G_RULE"
const NEWTON_TRUST_REGION = "NEWTON_TRUST_REGION"
const NewtonTrustRegion = "NewtonTrustRegion"

const CAT_FACTORIZATION_COLOR = :black
const CAT_THETA_ZERO_FACTORIZATION_COLOR = :red
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
        results_fraction = DataFrame(Iterations=Int[], CAT_FACTORIZATION=Float64[], CAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_G_RULE=Float64[], NewtonTrustRegion=Float64[])
        results_total = DataFrame(Iterations=Int[], CAT_FACTORIZATION=Int[], CAT_THETA_ZERO_FACTORIZATION=Int[], ARC_G_RULE=Int[], NewtonTrustRegion=Int[])
    else
        results_fraction = DataFrame(Gradients=Int[], CAT_FACTORIZATION=Float64[], CAT_THETA_ZERO_FACTORIZATION=Float64[], ARC_G_RULE=Float64[], NewtonTrustRegion=Float64[])
        results_total = DataFrame(Gradients=Int[], CAT_FACTORIZATION=Int[], CAT_THETA_ZERO_FACTORIZATION=Int[], ARC_G_RULE=Int[], NewtonTrustRegion=Int[])
    end

    for total in TOTAL
        total_problems_CAT_FACTORIZATION = length(filterRows(total, df[:, CAT_FACTORIZATION]))
        total_problems_CAT_THETA_ZERO_FACTORIZATION = length(filterRows(total, df[:, CAT_THETA_ZERO_FACTORIZATION]))
        total_problems_ARC_G_RULE = length(filterRows(total, df[:, ARC_G_RULE]))
        total_problems_NewtonTrustRegion = length(filterRows(total, df[:, NEWTON_TRUST_REGION]))
        push!(results_fraction, (total, total_problems_CAT_FACTORIZATION / total_number_problems, total_problems_CAT_THETA_ZERO_FACTORIZATION / total_number_problems, total_problems_ARC_G_RULE / total_number_problems, total_problems_NewtonTrustRegion / total_number_problems))
        push!(results_total, (total, total_problems_CAT_FACTORIZATION, total_problems_CAT_THETA_ZERO_FACTORIZATION, total_problems_ARC_G_RULE, total_problems_NewtonTrustRegion))
    end

    return results_fraction
end

function plotFigureComparisonCAT(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    data = Matrix(df[!, Not(criteria)])
    criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    plot(df[!, criteria],
        data,
        label=["Our method default (θ = 0.1)" "Our method (θ = 0.0)"],
        color = [CAT_FACTORIZATION_COLOR CAT_THETA_ZERO_FACTORIZATION_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, 10000),
        xaxis=:log10
    )
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresComparisonCAT(dirrectoryName::String)
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_FACTORIZATION,CAT_THETA_ZERO_FACTORIZATION]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_comparison_CAT.png"
    plotFigureComparisonCAT(results, "Iterations", dirrectoryName, plot_name)
end

function plotFiguresComparisonFinal(df::DataFrame, criteria::String, dirrectoryName::String, plot_name::String)
    data = Matrix(df[!, Not(criteria)])
    criteria_keyrword = criteria == "Iterations" ? "iterations" : "gradient evaluations"
    plot(df[!, criteria],
        data,
        label=["Our method" "ARC with g-rule" "Newton trust region"],
        color = [CAT_FACTORIZATION_COLOR ARC_G_RULE_COLOR NewtonTrustRegion_COLOR],
        ylabel="Fraction of problems solved",
        xlabel=string("Total number of ", criteria_keyrword),
        legend=:bottomright,
        xlims=(10, 10000),
        xaxis=:log10
    )
    fullPath = string(dirrectoryName, "/", plot_name)
    png(fullPath)
end

function generateFiguresIterationsComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_iterations.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_ITERATIONS, "Iterations")
    results = results[:, filter(x -> (x in ["Iterations", CAT_FACTORIZATION,ARC_G_RULE,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_iterations_count_final.png"
    plotFiguresComparisonFinal(results, "Iterations", dirrectoryName, plot_name)
end

function generateFiguresGradientsComparisonFinal(dirrectoryName::String)
    fileName = "all_algorithm_results_gradients.csv"
    fullPath = string(dirrectoryName, "/", fileName)
    df = readFile(fullPath)
    results = computeFraction(df, TOTAL_GRADIENTS, "Gradients")
    results = results[:, filter(x -> (x in ["Gradients", CAT_FACTORIZATION,ARC_G_RULE,NewtonTrustRegion]), names(results))]
    plot_name = "fraction_of_problems_solved_versus_total_gradients_count_final.png"
    plotFiguresComparisonFinal(results, "Gradients", dirrectoryName, plot_name)
end

function plotAllFigures(dirrectoryName::String)
    generateFiguresComparisonCAT(dirrectoryName)
    generateFiguresIterationsComparisonFinal(dirrectoryName)
    generateFiguresGradientsComparisonFinal(dirrectoryName)
end
