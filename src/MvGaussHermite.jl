module MvGaussHermite

using FastGaussQuadrature 
using LinearAlgebra 

export quadrature, mutableQuadrature, init, init_mutable

struct quadrature
    weights::AbstractVector{Float64}
    nodes::AbstractVector{AbstractVector{Float64}}
    Cov::AbstractMatrix{Float64} # covariance matrix
    dims::Int64 # number of dimensions 
    m::Int64 # order of aproximation 
    n::Int64
end 


mutable struct mutableQuadrature
    weights::AbstractVector{Float64} # quadrature weights 
    standardNodes::AbstractVector{AbstractVector{Float64}}
    nodes::AbstractVector{AbstractVector{Float64}} # beleif state nodes
    Cov::AbstractMatrix{Float64} # covariance matrix beleif state
    dims::Int64 # number of dimensions states
    m::Int64 # order of aproximation 
    n::Int64 # number of nodes
    R::AbstractMatrix{Float64}
end 

# I need to check the signs of the sin terms here
# it works in R2 and R3 but not in R4
function planar_rotation(d,theta)
    R = 1.0*Matrix(I,d,d)
    for i in 1:(d-1)
        R_ = 1.0*Matrix(I,d,d)
        R_[i,i] = cos(theta)
        R_[i,i+1] = -sin(theta)
        R_[i+1,i] = sin(theta)
        R_[i+1,i+1] = cos(theta)
        R .= R*R_
    end 
    return R
end 


"""
    nodes_grid(nodes, weights, dims)

makes a grid of Gauss hermite nodes and weights. I'm not quite sure how to write this
function and for and number of dims and above three MC is probably better any ways 
so I have defined sperate functions for dims in {1,2,3}

"""
function nodes_grid(nodes, weights, dims)
    @assert dims in [1,2,3,4]
    
    if dims == 1
        return nodes, weights
    elseif dims ==2
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^2)
        weights_vec = zeros(n^2)
        acc = 0
        for i in 1:n
            for j in 1:n
                acc += 1
                nodes_vec[acc] .= [nodes[i], nodes[j]]
                weights_vec[acc] = weights[i]*weights[j]
            end
        end 
        return nodes_vec, weights_vec
    elseif dims == 3
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        weights_vec = zeros(n^3)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    acc += 1
                    nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k]]
                    weights_vec[acc] = weights[i]*weights[j]*weights[k]
                end
            end
        end 
        return nodes_vec, weights_vec
    elseif dims == 4
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        weights_vec = zeros(n^4)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    for l in 1:n
                        acc += 1
                        nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k], nodes[l]]
                        weights_vec[acc] = weights[i]*weights[j]*weights[k]*weights[l]
                    end
                end
            end
        end 
        return nodes_vec, weights_vec
    end 
    
end 


function nodes_grid(nodes, dims)
    @assert dims in [1,2,3,4]
    
    if dims == 1
        return nodes
    elseif dims ==2
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^2)
        acc = 0
        for i in 1:n
            for j in 1:n
                acc += 1
                nodes_vec[acc] .= [nodes[i], nodes[j]]
            end
        end 
        return nodes_vec
    elseif dims == 3
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    acc += 1
                    nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k]]
                end
            end
        end 
        return nodes_vec
    elseif dims == 4
        n = length(nodes)
        nodes_vec = broadcast(i -> zeros(dims), 1:n^dims)
        acc = 0
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    for l in 1:n
                        acc += 1
                        nodes_vec[acc] .= [nodes[i], nodes[j], nodes[k], nodes[l]]
                    end
                end
            end
        end 
        return nodes_vec
    end 
    
end 


function init(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> broadcast(v -> v, x), nodes)


    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
   
    R = planar_rotation(dims,pi/4)
 
    # transform and plot 
    nodes = broadcast(x -> (S*rV)*R*x.+mu, nodes)
    
    return quadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 




function init(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64},theta::Float64)
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    nodes, weights = nodes_grid(nodes, weights, dims)
    
    nodes = broadcast(x -> broadcast(v -> v, x), nodes)


    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    
    R = planar_rotation(dims,pi/4)
 
    # transform and plot 
    nodes = broadcast(x -> S*rV*R*x.+mu, nodes)
    nodes = nodes[weights .> theta]
    weights = weights[weights .> theta]
    
    return quadrature(weights, nodes, Cov, dims, m, length(nodes))
    
end 



function init_mutable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64})
    dims = size(Cov)[1]
    nodes, weights = FastGaussQuadrature.gausshermite(m)
    weights = weights .* (2*pi)^(-1/2).*exp.((nodes.^2)./2)
    standardNodes, weights = nodes_grid(nodes, weights, dims)
    if dims > 1
        standardNodes = broadcast(x -> broadcast(v -> v, x), standardNodes)
    elseif dims ==1
        standardNodes = broadcast(x -> [x], standardNodes)
    end 

    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(dims,pi/4)
    
    # transform and plot 
    nodes = broadcast(x -> S*rV*R*x.+mu, standardNodes)
    

    return mutableQuadrature(weights, standardNodes, nodes,  Cov, dims, m, length(nodes),R)
    
end 


function init_mutable(m::Int64,mu::AbstractVector{Float64},Cov::AbstractMatrix{Float64},theta::Float64)
    dims = size(Cov)[1]
    # get weights
    standardNodes, weights = FastGaussQuadrature.gausshermite(m)
    # convert weights from exp(-x^2) to (pi/2)^1/2*exp(-1/2x^2)
    weights = weights .* (2*pi)^(-1/2).*exp.((standardNodes.^2)./2)
    standardNodes, weights = nodes_grid(standardNodes, weights, dims)
    
    if dims > 1
        nstandardNodes = broadcast(x -> broadcast(v -> v, x), standardNodes)
    elseif dims ==1
        standardNodes = broadcast(x -> [x], standardNodes)
    end 

    
    # spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,dims,dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    R = planar_rotation(dims,pi/4)

    # transform and plot 
    standardNodes = standardNodes[weights .> theta]
    weights = weights[weights .> theta]
    nodes = broadcast(x -> S*rV*R*x .+ mu, standardNodes)
    
    
    return mutableQuadrature(weights, standardNodes, nodes, Cov, dims, m, length(nodes),R)
    
end 

"""
Updates the transformation of the nodes with a new covariance matrix 
"""
function update_old!(mutableQuadrature, mu::AbstractVector{Float64}, Cov::AbstractMatrix{Float64})

    #spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,mutableQuadrature.dims,mutableQuadrature.dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    #R = planar_rotation(mutableQuadrature.dims,pi/4)
    R = mutableQuadrature.R
    # transform and plot 
    mutableQuadrature.nodes = broadcast(x -> S*rV*R*x .+ mu, mutableQuadrature.standardNodes)
    mutableQuadrature.Cov = Cov
end 

function g!(v,y,S,rV,R,mu)
    v .= S*rV*R*y .+ mu
end 
f! = (v,y,S,rV,R,mu) -> broadcast(i -> g!(v[i],y[i],S,rV,R,mu), 1:length(y))

    
"""
Updates the transformation of the nodes with a new covariance matrix 
"""
function update!(mutableQuadrature, mu::AbstractVector{Float64}, Cov::AbstractMatrix{Float64})

    #spectral decomposition
    estuff = eigen(Cov)
    rV = sqrt.(1.0*Matrix(I,mutableQuadrature.dims,mutableQuadrature.dims).*estuff.values)
    S = real.(estuff.vectors)

    # rotation matrix
    #R = MvGaussHermite.planar_rotation(mutableQuadrature.dims,pi/4)
    R = mutableQuadrature.R
    # transform and plot 
    #mutableQuadrature.nodes = broadcast(x -> S*rV*R*x .+ mu, mutableQuadrature.standardNodes)
    f!(mutableQuadrature.nodes,mutableQuadrature.standardNodes,S,rV,R,mu)
    mutableQuadrature.Cov = Cov
end 

"""
    expected_value(f::Function, quadrature::quadrature,  mu::AbstractVector{Float64})

computes the expecctation of a gausian random variable with mean mu and covariance quadrature.Cov
with respect to a function f
"""
function expected_value(f::Function, quadrature)#
    return sum(f.(broadcast(x -> x, quadrature.nodes)).*quadrature.weights)
end 



end # module 