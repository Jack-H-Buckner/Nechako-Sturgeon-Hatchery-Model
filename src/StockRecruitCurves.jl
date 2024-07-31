module StockRecruitCurves

using Roots

mutable struct BevetonHolt
    a::Float64
    b::Float64
    R_star::Float64
    k::Float64
    LEP::Float64
end


function (sr::BevetonHolt)(x)
    return sr.a*x/(1+sr.b*x)
end 

"""
    init_BevetonHolt(R_hat,k)

initalizes beverton-holt curve with compensaton ration k and
equilibrium recruitment R_hat, givin 
"""
function init_BevetonHolt(R_star::Float64,k::Float64, LEP::Float64)
    a = k ./ LEP
    b = (k.-1) ./ (R_star * LEP)
    BevetonHolt(a,b,R_star,k,LEP)
end 

function update_BevetonHolt_k!(sr, k::Float64)
    sr.a = k ./ sr.LEP
    sr.b = (k.-1) ./ (sr.R_star * sr.LEP)
    sr.k = k
end 

function update_BevetonHolt_Rstar!(sr, R_star::Float64)
    sr.a = sr.k ./ sr.LEP
    sr.b = (sr.k.-1) ./ (R_star * sr.LEP)
    sr.R_star = R_star
end 

function update_BevetonHolt_LEP!(sr, LEP::Float64)
    sr.a = sr.k ./ LEP
    sr.b = (sr.k.-1) ./ (R_star * LEP)
    sr.LEP = LEP
end 

"""
calcualtes density independent egg survivorship
"""
function density_independent(x,sr)
    x*sr.a
end 

"""
calcualtes density dependent egg survivorship
"""
function density_dependent(x,sr)
    1/(1+sr.b*x/sr.a)
end 

end 