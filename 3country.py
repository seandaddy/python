# %%
import sympy as sp
from sympy import solve, Eq

# %%
# Define symbolic variables using sympy
a_i, a_j, a_k, gamma = sp.symbols("a_i a_j a_k gamma")
p_ii, p_ji, p_ki, p_ij, p_jj, p_kj, p_ik, p_jk, p_kk, w = sp.symbols(
    "p_ii p_ji p_ki p_ij p_jj p_kj p_ik p_jk p_kk w"
)
q_ii, q_ji, q_ki, q_ij, q_jj, q_kj, q_ik, q_jk, q_kk = sp.symbols(
    "q_ii q_ji q_ki q_ij q_jj q_kj q_ik q_jk q_kk"
)

U_i = (
    a_i * q_ii
    + a_i * q_ji
    + a_i * q_ki
    - (
        (q_ii) ** 2 / 2
        + (q_ji) ** 2 / 2
        + (q_ki) ** 2 / 2
        + gamma * q_ii * q_ji
        + gamma * q_ii * q_ki
        + gamma * q_ji * q_ki
    )
)
U_j = (
    a_j * q_ij
    + a_j * q_jj
    + a_j * q_kj
    - (
        (q_ij) ** 2 / 2
        + (q_jj) ** 2 / 2
        + (q_kj) ** 2 / 2
        + gamma * q_ij * q_jj
        + gamma * q_ij * q_kj
        + gamma * q_jj * q_kj
    )
)
U_k = (
    a_k * q_ik
    + a_k * q_jk
    + a_k * q_kk
    - (
        (q_ik) ** 2 / 2
        + (q_jk) ** 2 / 2
        + (q_kk) ** 2 / 2
        + gamma * q_ik * q_jk
        + gamma * q_ik * q_kk
        + gamma * q_jk * q_kk
    )
)

CS_i = U_i - p_ii * q_ii - p_ji * q_ji - p_ki * q_ki
CS_j = U_j - p_ij * q_ij - p_jj * q_jj - p_kj * q_kj
CS_k = U_k - p_ik * q_ik - p_jk * q_jk - p_kk * q_kk

Pi_i = (p_ii - w) * q_ii + (p_ij - w) * q_ij + (p_ik - w) * q_ik
Pi_j = (p_ji - w) * q_ji + (p_jj - w) * q_jj + (p_jk - w) * q_jk
Pi_k = (p_ki - w) * q_ki + (p_kj - w) * q_kj + (p_kk - w) * q_kk

# %%
# prompt: take a first derivative CS_i with respect to q_ii and solve for p_ii

rp_ii = sp.solve(sp.diff(CS_i, q_ii), p_ii)[0]
rp_ji = sp.solve(sp.diff(CS_i, q_ji), p_ji)[0]
rp_ki = sp.solve(sp.diff(CS_i, q_ki), p_ki)[0]

rp_ij = sp.solve(sp.diff(CS_j, q_ij), p_ij)[0]
rp_jj = sp.solve(sp.diff(CS_j, q_jj), p_jj)[0]
rp_kj = sp.solve(sp.diff(CS_j, q_kj), p_kj)[0]

rp_ik = sp.solve(sp.diff(CS_k, q_ik), p_ik)[0]
rp_jk = sp.solve(sp.diff(CS_k, q_jk), p_jk)[0]
rp_kk = sp.solve(sp.diff(CS_k, q_kk), p_kk)[0]

print(rp_ii)
print(rp_ji)
print(rp_ki)

print(rp_ij)
print(rp_jj)
print(rp_kj)

print(rp_ik)
print(rp_jk)
print(rp_kk)

# %%
Pi_i = Pi_i.subs(
    {
        p_ii: rp_ii,
        p_ji: rp_ji,
        p_ki: rp_ki,
        p_ij: rp_ij,
        p_jj: rp_jj,
        p_kj: rp_kj,
        p_ik: rp_ik,
        p_jk: rp_jk,
        p_kk: rp_kk,
    }
)
Pi_j = Pi_j.subs(
    {
        p_ii: rp_ii,
        p_ji: rp_ji,
        p_ki: rp_ki,
        p_ij: rp_ij,
        p_jj: rp_jj,
        p_kj: rp_kj,
        p_ik: rp_ik,
        p_jk: rp_jk,
        p_kk: rp_kk,
    }
)
Pi_k = Pi_k.subs(
    {
        p_ii: rp_ii,
        p_ji: rp_ji,
        p_ki: rp_ki,
        p_ij: rp_ij,
        p_jj: rp_jj,
        p_kj: rp_kj,
        p_ik: rp_ik,
        p_jk: rp_jk,
        p_kk: rp_kk,
    }
)

print(Pi_i)
print(Pi_j)
print(Pi_k)

# %%
rq_ii = sp.solve(sp.diff(Pi_i, q_ii), q_ii)[0]
rq_ij = sp.solve(sp.diff(Pi_i, q_ij), q_ij)[0]
rq_ik = sp.solve(sp.diff(Pi_i, q_ik), q_ik)[0]

rq_ji = sp.solve(sp.diff(Pi_j, q_ji), q_ji)[0]
rq_jj = sp.solve(sp.diff(Pi_j, q_jj), q_jj)[0]
rq_jk = sp.solve(sp.diff(Pi_j, q_jk), q_jk)[0]

rq_ki = sp.solve(sp.diff(Pi_k, q_ki), q_ki)[0]
rq_kj = sp.solve(sp.diff(Pi_k, q_kj), q_kj)[0]
rq_kk = sp.solve(sp.diff(Pi_k, q_kk), q_kk)[0]

print(rq_ii)
print(rq_ij)
print(rq_ik)

print(rq_ji)
print(rq_jj)
print(rq_jk)

print(rq_ki)
print(rq_kj)
print(rq_kk)

# %%
solutions = solve(
    [
        Eq(q_ii, rq_ii),
        Eq(q_ij, rq_ij),
        Eq(q_ik, rq_ik),
        Eq(q_ji, rq_ji),
        Eq(q_jj, rq_jj),
        Eq(q_jk, rq_jk),
        Eq(q_ki, rq_ki),
        Eq(q_kj, rq_kj),
        Eq(q_kk, rq_kk),
    ],
    [q_ii, q_ij, q_ik, q_ji, q_jj, q_jk, q_ki, q_kj, q_kk],
)

# Access individual solutions if found
eq_ii = solutions.get(q_ii, None)  # Use .get() to avoid KeyError if not found
eq_ij = solutions.get(q_ij, None)
eq_ik = solutions.get(q_ik, None)
eq_ji = solutions.get(q_ji, None)
eq_jj = solutions.get(q_jj, None)
eq_jk = solutions.get(q_jk, None)
eq_ki = solutions.get(q_ki, None)
eq_kj = solutions.get(q_kj, None)
eq_kk = solutions.get(q_kk, None)

print(eq_ii)
print(eq_ij)
print(eq_ik)
print(eq_ji)
print(eq_jj)
print(eq_jk)
print(eq_ki)
print(eq_kj)
print(eq_kk)
# %%
Pi_i = Pi_i.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
Pi_j = Pi_j.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
Pi_k = Pi_k.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)

print(Pi_i)
print(Pi_j)
print(Pi_k)
# %%
Pi_i = sp.simplify(Pi_i)
Pi_j = sp.simplify(Pi_j)
Pi_k = sp.simplify(Pi_k)

print(Pi_i)
print(Pi_j)
print(Pi_k)

# %%
rp_ii = rp_ii.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_ji = rp_ji.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_ki = rp_ki.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_ij = rp_ij.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_jj = rp_jj.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_jk = rp_jk.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_ik = rp_ik.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_jk = rp_jk.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)
rp_kk = rp_kk.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
    }
)

rp_ii = sp.simplify(rp_ii)
rp_ji = sp.simplify(rp_ji)
rp_ki = sp.simplify(rp_ki)
rp_ij = sp.simplify(rp_ij)
rp_jj = sp.simplify(rp_jj)
rp_jk = sp.simplify(rp_jk)
rp_ik = sp.simplify(rp_ik)
rp_jk = sp.simplify(rp_jk)
rp_kk = sp.simplify(rp_kk)

print(rp_ii)
print(rp_ji)
print(rp_ki)
print(rp_ij)
print(rp_jj)
print(rp_jk)
print(rp_ik)
print(rp_jk)
print(rp_kk)

# %%
# Vertical Relationship
q_l, q_m, q_n, k = sp.symbols("q_l q_m q_n k")
Q = eq_ii + eq_ij + eq_ik + eq_ji + eq_jj + eq_jk + eq_ki + eq_kj + eq_kk
q_a = q_l + q_m + q_n
rw = sp.solve(Eq(Q, q_a), w)[0]
print(rw)

# %%
Pi_l = (rw - k) * q_l
Pi_m = (rw - k) * q_m
Pi_n = (rw - k) * q_n
rq_l = sp.diff(Pi_l, q_l)
rq_m = sp.diff(Pi_m, q_m)
rq_n = sp.diff(Pi_n, q_n)

print(rq_l)
print(rq_m)
print(rq_n)
# %%
solution = solve([Eq(q_l, rq_l), Eq(q_m, rq_m), Eq(q_n, rq_n)], [q_l, q_m, q_n])

# Access individual solutions if found
eq_l = solution.get(q_l, None)  # Use .get() to avoid KeyError if not found
eq_m = solution.get(q_m, None)
eq_n = solution.get(q_n, None)

print(eq_l)
print(eq_m)
print(eq_n)
# %%
ew = rw.subs({q_l: eq_l, q_m: eq_m, q_n: eq_n})
print(ew)

# %%
eq_a = eq_l + eq_m + eq_n
print(eq_a)

# %%
rPi_l = Pi_l.subs({q_l: eq_l, q_m: eq_m, q_n: eq_n})
rPi_m = Pi_m.subs({q_l: eq_l, q_m: eq_m, q_n: eq_n})
rPi_n = Pi_n.subs({q_l: eq_l, q_m: eq_m, q_n: eq_n})

rPi_l = sp.simplify(rPi_l)
rPi_m = sp.simplify(rPi_m)
rPi_n = sp.simplify(rPi_n)

print(rPi_l)
print(rPi_m)
print(rPi_n)

# %%
eq_ii = eq_ii.subs({w: ew})
eq_ji = eq_ji.subs({w: ew})
eq_ki = eq_ki.subs({w: ew})
eq_ij = eq_ij.subs({w: ew})
eq_jj = eq_jj.subs({w: ew})
eq_kj = eq_kj.subs({w: ew})
eq_ik = eq_ik.subs({w: ew})
eq_jk = eq_jk.subs({w: ew})
eq_kk = eq_kk.subs({w: ew})

print(eq_ii)
print(eq_ji)
print(eq_ki)
print(eq_ij)
print(eq_jj)
print(eq_jk)
print(eq_ik)
print(eq_jk)
print(eq_kk)

# %%
CS_i = CS_i.subs(
    {
        q_ii: eq_ii,
        q_ji: eq_ji,
        q_ki: eq_ki,
        q_ij: eq_ij,
        q_jj: eq_jj,
        q_kj: eq_kj,
        q_ik: eq_ik,
        q_jk: eq_jk,
        q_kk: eq_kk,
        p_ii: rp_ii,
        p_ji: rp_ji,
        p_ki: rp_ki,
        p_ij: rp_ij,
        p_jj: rp_jj,
        p_kj: rp_jk,
        p_ik: rp_ik,
        p_jk: rp_jk,
        p_kk: rp_kk,
        w: ew,
        q_l: eq_l,
        q_m: eq_m,
        q_n: eq_n,
    }
)
print(CS_i)
