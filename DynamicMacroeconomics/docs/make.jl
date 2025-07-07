using Documenter
using DynamicMacroeconomics

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "DynamicMacroeconomics Documentation",
    authors = "Charles Knipp",
    # modules = [DynamicMacroeconomics],
)

deploydocs(
    repo = "github.com/charlesknipp/DynamicMacroeconomics.jl.git",
    versions = nothing
)