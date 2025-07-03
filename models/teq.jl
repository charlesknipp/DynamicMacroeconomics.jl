using DynamicMacroeconomics

@block function euler(y, πs, i, ωs)
    y[t] = y[t+1] - (1 / γ) * (i[t] - πs[t-1]) + ωs[t]
end;

@block function nkpc(y, πs, ωd)
    πs[t] = β * πs[t+1] + (1 - θ) * (1 - θ * β) / θ * (γ + φ) * y[t] - ωd[t]
end;

@block function taylor(y, πs, i, ωm)
    i[t] = ϕi * i[t-1] + (1 - ϕi) * (ϕπ * πs[t] + ϕy * y[t]) + ωm[t]
end;

@block function ar_shocks(ωs, ωd, ωm, εs, εd, εm)
    ωs[t] = ρs * ωs[t-1] + σs * εs[t]
    ωd[t] = ρd * ωd[t-1] + σd * εd[t]
    ωm[t] = ρm * ωm[t-1] + σm * εm[t]
end;

## DEMO ####################################################################################

θ1 = (
    β = (1 / 1.05),
    θ  = 0.75,
    φ  = 1.00,
    ϕy = 0.10,
    ϕπ = 1.50,
    ϕi = 0.90,
    γ  = 1.00
);

# start with one time shocks
analytical_steady_state_1(θ) = (y=0, πs=0, i=0);
teq_model_1 = RationalExpectationsModel(
    [euler, nkpc, taylor], [:ωs, :ωd, :ωm], analytical_steady_state_1
);

policy_1 = solve(teq_model_1, θ1, 1; algo=QuadraticIteration());

θ2 = (;
    θ1...,
    ρd = 0.80,
    ρs = 0.90,
    ρm = 0.20,
    σd = 1.60,
    σs = 0.95,
    σm = 0.25
);

# add AR(1) shocks
analytical_steady_state_2(θ) = (y=0, πs=0, i=0, ωs=0, ωd=0, ωm=0);
teq_model_2 = RationalExpectationsModel(
    [euler, nkpc, taylor, ar_shocks], [:εs, :εd, :εm], analytical_steady_state_2
);

policy_2 = solve(teq_model_2, θ2, 1; algo=QuadraticIteration());
