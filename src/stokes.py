from jax import grad, jacrev, vmap, jit
import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.linalg import lu_factor, lu_solve
from src.util import *

def trapez(N):
    t = jnp.linspace(0, jnp.pi*2, N+1)[:-1]
    w = jnp.ones_like(t) * jnp.pi*2 / N
    return t, w

def integrate(f, t, w):
    return jnp.sum(f(t)*w)

def curve_integral(f, g, t, w):
    return integrate(vmap(lambda s: f(s)*norm(jacrev(g)(s))), t, w)

def kernel_integral(f, k, g, t, w):
    return vmap(lambda r: curve_integral(lambda s: k(r, s) @ f(s), g, t, w))(t)

def mlp_solution(q, g, t, w, eta):
    slp = slp_solution(q, g, t, w)
    dlp = dlp_solution(q, g, t, w)
    return lambda x: eta * slp(x) + dlp(x)

def mlp_matrix(g, t, w, eta):
    return dlp_matrix(g, t, w) + slp_matrix(g, t, w)*eta

def slp_matrix(g, t, w):
    dg = lambda t: norm(jacrev(g)(t))
    integrand = lambda t, s: stokeslet(g(t) - g(s)) * dg(s)
    integrand_ = lambda t, s: cond(s!=t, integrand, lambda *args: jnp.zeros((2,2)), t, s)
    matrix = jnp.einsum("ijkl, j -> ijkl", vmap(vmap(integrand_, (None, 0)), (0, None))(t, t), w)
    return matrix / (4*jnp.pi)

def slp_solution(q, g, t, w):
    dg = lambda t: norm(jacrev(g)(t))
    integrand = lambda x, s: stokeslet(x - g(s)) * dg(s)
    def solution(x):
        integrand_x = vmap(lambda s: integrand(x, s)/ (4*jnp.pi))(t) 
        return jnp.einsum("ijk, ik -> j", integrand_x, q*w[:, None])
    return solution

def dlp_matrix(g, t, w):
    n = normal(g)
    dg = lambda t: norm(jacrev(g)(t))
    integrand = lambda t, s: jnp.einsum("ijk,k->ij", stresslet(g(t) - g(s)), n(s)) * dg(s)
    integrand_ = lambda t, s: cond(s!=t, integrand, lambda *args: jnp.zeros((2,2)), t, s)
    matrix = jnp.einsum("ijkl, j -> ijkl", vmap(vmap(integrand_, (None, 0)), (0, None))(t, t), w)
    Idmat = jnp.einsum("ij,kl->ijkl", jnp.eye(len(t)), jnp.eye(2))
    return matrix/(4*jnp.pi) - Idmat * 0.5
    
def dlp_solution(q, g, t, w):
    n = normal(g)
    dg = lambda t: norm(jacrev(g)(t))
    integrand = lambda x, s: jnp.einsum("ijk,k->ij", stresslet(x - g(s)), n(s))*dg(s)
    def solution(x):
        integrand_x = vmap(lambda s: integrand(x, s)/(4*jnp.pi))(t)
        return jnp.einsum("ijk, ik -> j", integrand_x, q*w[:, None])
    return solution
    
def stokeslet(r):
    abs_r = norm(r)
    c1 = -jnp.log(abs_r) * jnp.eye(2)
    c2 = jnp.einsum("i,j->ij", r, r)/ abs_r**2
    return c1 + c2
    
def stresslet(r):
    abs_r = norm(r)
    return -4*jnp.einsum("i,j,k->ijk", r, r, r) / abs_r**4

def rotmat(theta):  
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

def norm(r):
    return jnp.linalg.norm(r, axis=0)

def normalize(g):
    return g / norm(g)

def normal(g):
    return lambda t: jnp.einsum("ij,j->i", rotmat(-jnp.pi/2), normalize(jacrev(g)(t)))

def stackmatrix(mat):
    return jnp.vstack([jnp.hstack([mat[:, :, i, j] for j in range(2)]) for i in range(2)])

def stackvector(vec):
    return jnp.vstack([vec[:, i, None] for i in range(2)])

def unstackvector(vec):
    N = len(vec)//2
    return jnp.hstack([vec[:N], vec[N:]])
    
def fourier_eval_flat(t, c):
    K = (len(c)-1)//2
    ks = jnp.arange(-K, K+1)
    E = jnp.exp(1j * ks * t)
    return jnp.sum(E * c.flatten())

def constant_vector_field(uinf, t):
    return vmap(lambda _: uinf)(t)

def force(q, g, t, w):
    return vmap(lambda v: curve_integral(lambda t: 1., g, t, v*w), (1))(q)
