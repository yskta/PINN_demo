"""
Explicit (FTCS) vs. Implicit (Jacobi)
"""
#ref: https://qiita.com/ShotaDeguchi/items/574f5e41ad622990f70c
import numpy as np
import matplotlib.pyplot as plt

def main():
    # conputational domain
    xmin, xmax = 0., 1.               # 1 m
    tmin, tmax = 0., 60. * 60. * 1.   # 1 hour
    n_max = int(1e3)   # max iteration (for implicit)
    c_tol = 1e-6       # convergence tolerance (for implicit)

    # material parameter (copper)
    rho = 8960.   # density (kg / m3)
    lam = 398.    # conductivity (W / m / K)
    cap = 386.    # capacity
    D   = lam / (rho * cap)

    # discretization
    dx = 1e-2
    dt = 1e-1
    dt_star = dx ** 2 / (2. * D)   # diffusion condition
    beta = D * dt / (dx ** 2.)    # coef
    nx = 1 + int((xmax - xmin) / dx)
    nt = 1 + int((tmax - tmin) / dt)
    x = np.linspace(xmin, xmax, nx)

    # unknown vector
    u = np.ones(shape=(nx))

    # initial condition
    u_ic = 20.
    u *= u_ic

    # boundary condition
    u_bc = 100.
    u[0]  = u_bc    # Dirichlet
    u[-1] = u[-2]   # Neumann

    # conputational setting
    mode = 1   # 0 - explicit (FTCS), 1 - implicit (Jacobi)

    if mode == 0:
        print(">>>>> explicit")
        if dt < dt_star:
            print(">>>>> dt: %.3e, dt_star: %.3e" % (dt, dt_star))
            print(">>>>> CFL met")
        else:
            print(">>>>> dt: %.3e, dt_star: %.3e" % (dt, dt_star))
            print(">>>>> CFL NOT met, program terminating now")
            exit()
    elif mode == 1:
        print(">>>>> implicit")
        print(">>>>> dt: %.3e, dt_star: %.3e" % (dt, dt_star))

    # FDM: Finite Difference Method
    if mode == 0:   # explicit
        for n in range(0, nt-1):
            # copy of u
            v = np.copy(u)

            # slow: element-wise operation
            # for i in range(1, nx-1):
            #     u[i] = v[i] + beta * (v[i+1] - 2. * v[i] + v[i-1])

            # fast: slice operation
            u[1:-1] = v[1:-1] + beta * (v[2:] - 2. * v[1:-1] + v[:-2])

            # boundary condition
            u[0] = u_bc
            u[-1] = u[-2]

            # damp-out
            if n % int(nt / 10) == 0:
                print("step: %d / %d" % (n, nt))
                plt.figure(figsize=(8, 4))
                plt.plot(x, u)
                plt.xlim(-.1, 1.1)
                plt.ylim(0, 120)
                plt.title("t: %.1f min (step: %d / %d)" % (n * dt / 60., n, nt))
                plt.xlabel("x")
                plt.ylabel("u")
                plt.grid(alpha=.3)
                plt.savefig("./res/" + str(n) + ".png")
                plt.clf()
                plt.close()

    elif mode == 1:   # implicit
        for n in range(0, nt-1):
            v = np.copy(u)
            for n_ in range(n_max):
                w = np.copy(u)

                # slow
                # for i in range(1, nx-1):
                #     u[i] = 1. / (1. + 2. * beta) \
                #             * (v[i] + beta * (w[i+1] + w[i-1]))

                # fast
                u[1:-1] = 1. / (1. + 2. * beta) \
                        * (v[1:-1] + beta * (w[2:] + w[:-2]))

                # residual
                if n_ % 10 == 0:
                    r_ = np.sqrt(np.sum(u - w) ** 2) / np.sum(w ** 2)
                    if r_ < c_tol:
                        break

            # boundary condition
            u[0]  = u_bc
            u[-1] = u[-2]

            # damp-out
            if n % int(nt / 10) == 0:
                print("step: %d / %d" % (n, nt))
                plt.figure(figsize=(8, 4))
                plt.plot(x, u)
                plt.xlim(-.1, 1.1)
                plt.ylim(0, 120)
                plt.title("t: %.1f min (step: %d / %d)" % (n * dt / 60., n, nt))
                plt.xlabel("x")
                plt.ylabel("u")
                plt.grid(alpha=.3)
                plt.savefig("./res/" + str(n) + ".png")
                plt.clf()
                plt.close()

if __name__ == "__main__":
    main()
