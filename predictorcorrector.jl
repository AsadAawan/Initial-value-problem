using SpecialFunctions
using Plots

# Parameters
alpha = 0.982
alpha1 = 0.09
lamda = 1e4
dt = 0.01
rho = 0.1
di = 1
dl = 4e-3
N1 = 2000
dv = 23
k = 2.4e-8

# Time interval
t0 = 0
tf = 300
h = 1 / 50
N = Int(round((tf - t0) / h))

# Initial Conditions
T0 = 4e5
I0 = 0
L0 = 0
V0 = 1e5

# System of ODEs
function f1(t, T, I, L, V)
    return lamda .- k .* T .* V .- dt .* T
end

function f2(t, T, I, L, V)
    return (1 .- rho) .* k .* T .* V .- di .* I .+ alpha1 .* L
end

function f3(t, T, I, L, V)
    return rho .* k .* T .* V .- dl .* L .- alpha1 .* L
end

function f4(t, T, I, L, V)
    return N1 .* di .* I .- dv .* V
end

# Solution vectors
T = zeros(N+1, 1)
I = zeros(N+1, 1)
L = zeros(N+1, 1)
V = zeros(N+1, 1)

# Initial values
T[1] = T0
I[1] = I0
L[1] = L0
V[1] = V0

a = similar(T[:, 1], Float64)
b = similar(T[:, 1], Float64)

# Fractional Adams-Bashforth method
for i in 1:N
    b[i+1] = i^alpha - (i - 1)^alpha
    a[i+1] = (i + 1)^(alpha + 1) - 2 * i^(alpha + 1) + (i - 1)^(alpha + 1)
end

for j in 2:N+1
    idx = 1:j-1

    p1 = T0 .+ h^alpha / gamma(alpha + 1) .* sum(b[j .- idx] .* f1(idx .* h, T[idx], I[idx], L[idx], V[idx]))
    p2 = I0 .+ h^alpha / gamma(alpha + 1) .* sum(b[j .- idx] .* f2(idx .* h, T[idx], I[idx], L[idx], V[idx]))
    p3 = L0 .+ h^alpha / gamma(alpha + 1) .* sum(b[j .- idx] .* f3(idx .* h, T[idx], I[idx], L[idx], V[idx]))
    p4 = V0 .+ h^alpha / gamma(alpha + 1) .* sum(b[j .- idx] .* f4(idx .* h, T[idx], I[idx], L[idx], V[idx]))

    T[j] = T0 .+ h^alpha / gamma(alpha + 2) .* (f1(j .* h, p1, p2, p3, p4) .+ ((j .- 1)^(alpha .+ 1) .- (j .- 1 .- alpha) .* j^alpha) .* f1(0, T0, I0, L0, V0) .+ sum(a[j .- idx] .* f1(idx * h, T[idx], I[idx], L[idx], V[idx])))
    I[j] = I0 .+ h^alpha / gamma(alpha + 2) .* (f2(j .* h, p1, p2, p3, p4) .+ ((j .- 1)^(alpha .+ 1) .- (j .- 1 .- alpha) .* j^alpha) .* f2(0, T0, I0, L0, V0) .+ sum(a[j .- idx] .* f2(idx * h, T[idx], I[idx], L[idx], V[idx])))
    L[j] = L0 .+ h^alpha / gamma(alpha + 2) .* (f3(j .* h, p1, p2, p3, p4) .+ ((j .- 1)^(alpha .+ 1) .- (j .- 1 .- alpha) .* j^alpha) .* f3(0, T0, I0, L0, V0) .+ sum(a[j .- idx] .* f3(idx * h, T[idx], I[idx], L[idx], V[idx])))
    V[j] = V0 .+ h^alpha / gamma(alpha + 2) .* (f4(j .* h, p1, p2, p3, p4) .+ ((j .- 1)^(alpha .+ 1) .- (j .- 1 .- alpha) .* j^alpha) .* f4(0, T0, I0, L0, V0) .+ sum(a[j .- idx] .* f4(idx * h, T[idx], I[idx], L[idx], V[idx])))
end
using Plots

# Plotting the results
t = t0:h:tf

plot(t, T, label = "T")
plot!(t, I, label = "I")
plot!(t, L, label = "L")
plot!(t, V, label = "V")

xlabel!("Time")
ylabel!("Population")
title!("Population Dynamics")