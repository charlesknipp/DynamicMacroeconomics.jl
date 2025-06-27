using Pkg

# download the PR for type stable AD in SSMProblems
download_ssm_packages() = Pkg.add([
    PackageSpec(name="SSMProblems", rev="ck/priors"),
    PackageSpec(name="GeneralisedFilters", rev="ck/priors")
])

# do it for proper builds of DynamicMacro
Pkg.activate("DynamicMacroeconomics/")
download_ssm_packages()

# also for precompilation in the custom environment
Pkg.activate(".")
Pkg.develop(path="DynamicMacroeconomics/")
download_ssm_packages()
