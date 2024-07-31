module DomesticationParameters

using Optimization, OptimizationOptimJL,  ComponentArrays, OptimizationOptimisers, Zygote, OptimizationOptimJL
include("MvGaussHermite.jl")


function dsn(g,gbar,V) 
    1/sqrt(2*3.1416*V)*exp.(-0.5*(g .- gbar).^2 ./V)
end 



function step(s, theta, mu, V)
    V_prime = (1/V + s)^(-1)
    mu_prime = (mu/V .+ s*theta)*V_prime
    
    p = exp(-1/2*((mu/sqrt(V))^2 + s*theta^2 -(mu_prime/sqrt(V_prime))^2))
    p *= sqrt(V_prime/V)
    return mu_prime,V_prime,p
end 


Linfty(g,p) = p.L_infty + p.b_Linfty*g
length_(a,g,p) = Linfty(g,p)*(1-exp(-1*p.k*p.a_r))+p.L_0*exp(-1*p.k*p.a_r)
survival(l,p) = 1/(1+exp(-1*p.s_l*(l - p.l50)))

function expected_survival(a,g,p_opt,p_fixed;m=50) 
    mu = [length_(a,g,p_fixed)]
    Cov = zeros(1,1)
    Cov[1,1] = 1/p_fixed.h2_lr-1
    quad = MvGaussHermite.init_mutable(m,mu,Cov)
    f = l -> survival(l[1],p_opt)
    MvGaussHermite.expected_value(f,quad)
end 


function Vstar(Vr::Float64,s::Float64)
    a = s
    b = (0.5-Vr*s)
    c = -Vr
    return (-b+sqrt(b^2-4*a*c))/(2*a)
end 


function selection(gbar,p_opt,p_fixed;grid = collect(-5:0.01:10), V = 1.0)
    
    # genotype variance after viability selection 
    V0 = Vstar(p_fixed.Vr,p_opt.s-p_opt.s_F)
    gbar,V,S_F = step(p_opt.s_F,p_opt.theta,gbar,V0)
    
    # genptype variance after reproduction
    V = 0.5*V + 0.5
    gbar,V,S_E = step(p_opt.s_E,p_opt.theta,gbar,V)
    s_J = 0.0
    if p_fixed.decreasing_selection
        s_J = p_opt.s_J*log(p_fixed.a_r*p_fixed.b_sa+1)/p_fixed.b_sa
    else
        s_J = p_fixed.a_r*p_opt.s_J    
    end
    gbar,V,S_J = step(s_J,p_opt.theta,gbar,V)
    d = broadcast(g -> dsn(g,gbar,V), grid)
    d .*= 1/sum(d)
    gr = broadcast(g -> expected_survival(p_fixed.a_r,g,p_opt,p_fixed,m=20) , grid) 
    d .*= gr
    S_C = sum(d)
    d .*= 1/S_C
    gbar = sum(grid .* d)
    V = sum(d.*(grid .- gbar).^2)
    return gbar, V, S_F, S_E, S_J, S_C
end 


function rLRS(gbar,p_opt,p_fixed)
    
    # genotype distriubtion variance at birth 
    V0 = Vstar(p_fixed.Vr,p_opt.s)
    V0 = 0.5*V0 + p_fixed.Vr
    
    LRS_w = 1.0
    LRS_c = 1.0
    s1 = p_opt.s * 0.5
    s2 = p_opt.s * 0.25
    s3 = p_opt.s * 0.25
    
    # pre density-dependent selectionn 
    gbar_w,V_w,p_w = step(s1, 0.0, 0.0, V0)
    gbar_c,V_c,p_c = step(s1, 0.0, gbar,V0)
    LRS_w *= p_w
    LRS_c *= p_c
    
    # post density-dependent selectionn 
    gbar_w,V_w,p_w = step(s2, 0.0,gbar_w, V_w)
    gbar_c,V_c,p_c = step(s2, 0.0, gbar_c,V_c)
    LRS_w *= p_w
    LRS_c *= p_c
    
    # fecundity selectionn 
    gbar_w,V_w,p_w = step(s3, 0.0, gbar_w, V_w)
    gbar_c,V_c,p_c = step(s3, 0.0, gbar_c,V_c)
    LRS_w *= p_w
    LRS_c *= p_c
    
    return LRS_c/LRS_w
end 

function loss(p_opt,p_fixed,targets)
    V = Vstar(p_fixed.Vr,p_opt.s)
    gbar, V, S_F, S_E, S_J, S_C = selection(0.0,p_opt,p_fixed)
    L = (log(targets.S_F) - log(S_F))^2
    L += (log(targets.S_E) - log(S_E))^2
    L += (log(targets.S_J) - log(S_J))^2
    L += (log(targets.S_C) - log(S_C))^2
    L += (targets.rLRS - rLRS(gbar, p_opt,p_fixed))^2
    L += 0.05*(p_opt.l50 - targets.l50)^2
    L += 0.05*(p_opt.theta - targets.theta)^2
    L += 0.001*p_opt.s^2
    return L
end 


function solve_parameters(targets, fixed_pars)

    p_opt = (s = 0.05,s_F = 0.01,theta=10.0,s_E=0.01,s_J = 0.01,l50=40.0,s_l=1.5)


    pinit = ComponentArray(p_opt)
    prob = OptimizationProblem((x, p) -> loss(x,p,targets), pinit, fixed_pars)
    
    sol = solve(prob, NelderMead())
    parameters = merge(fixed_pars,sol.u)

    return parameters, sol.u.s
end 


end # module 