using DataFrames, GLM

df = DataFrame(
    R = Float64.(results.R_means),
    A = Float64.(results.A_means),
    Action = Float64.(results.actions)
)

model = lm(@formula(Action ~ R + A), df)
display(coeftable(model))
