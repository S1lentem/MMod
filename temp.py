
def get_partition_cycle(u, ij, n):
    ci = [0] * n
    cj = [0] * n
    ui = [i for i, _ in u]
    uj = [j for _, j in u]
    for i, j in u:
        ci[i] += 1
        cj[j] += 1
    stop = False
    while not stop:
        stop = True
        for i, j in enumerate(ci):
            if j != 1 or cj[i] != 0:
                continue
            stop = False
            k = ui.index(i)
            cj[uj[k]] -= 1
            ci[i] -= 1
            del ui[k]
            del uj[k]

        for i, j in enumerate(cj):
            if j != 1 or ci[i] != 0:
                continue
            stop = False
            k = uj.index(i)
            ci[ui[k]] -= 1
            cj[i] -= 1
            del ui[k]
            del uj[k]
    return partition(ui, uj, ij)


def partition(ui, uj, ij):
    k = list(zip(ui, uj)).index(ij)
    del ui[k]
    del uj[k]
    u_minus = []
    u_plus = []
    r, l = ij
    u_plus.append(ij)
    while ui:
        if l in ui:
            k = ui.index(l)
            u_plus.append((ui[k], uj[k]))
            l = uj[k]
            del ui[k]
            del uj[k]
        else:
            k = uj.index(l)
            u_minus.append((ui[k], uj[k]))
            l = ui[k]
            del ui[k]
            del uj[k]

    return u_plus, u_minus


def dfs(g, used, v, p, prev_v=-1):
    used[v] = 1
    for i in g[v]:
        if i != prev_v:
            if used[i] == 0:
                p[i] = v
                dfs(g, used, i, p, v)
            elif used[i] == 1:
                return
            used[v] = 2


def get_partition_cycle_from_dfs(u, ij, n):
    g = [[] for _ in range(n)]
    for i, j in u:
        g[i].append(j)
        g[j].append(i)
    used = [0] * n
    p = [None] * n
    dfs(g, used, ij[1], p, ij[0])
    v = ij[0]
    cycle = []
    while v != ij[1]:
        cycle.append((p[v], v))
        v = p[v]
    u_plus = [i for i in cycle if i in u]
    u_plus.append(ij)
    u_minus = [(i[1], i[0]) for i in cycle if i not in u_plus]
    return u_plus, u_minus


def get_potentials(c, ub, n):
    u = [None] * n
    u[0] = 0
    for _ in range(n):
        for i, j in ub:
            if u[j] is not None and u[i] is None:
                u[i] = c[(i, j)] + u[j]
                break
            elif u[i] is not None and u[j] is None:
                u[j] = u[i] - c[(i, j)]
                break
    return u


def solve_min_cost_network_flow(a, c, ub, x):
    ub = ub[:]
    x = x.copy()
    n = len(a)
    un = [(i, j) for i, j in c.keys() if (i, j) not in ub]
    while True:
        potentials = get_potentials(c, ub, n)
        ns = [(i, j) for i, j in un if potentials[i] - potentials[j] - c[(i, j)] > 0]
        if not ns:
            return x
        s = ns[0]  # i0j0
        ub.append(s)
        # u_plus, u_minus = get_partition_cycle(ub, s, n)
        u_plus, u_minus = get_partition_cycle_from_dfs(ub, s, n)
        if not u_minus:
            return None
        ij = min(u_minus, key=lambda p: x[p])
        t = x[ij]
        for i in u_plus:
            x[i] += t
        for i in u_minus:
            x[i] -= t
        ub.remove(ij)
        un.remove(s)
        un.append(ij)

def balance_check(a, x):
    n = len(a)
    for i in range(n):
        s = 0
        for k, l in x.items():
            if k[0] == i:
                s += l
            if k[1] == i:
                s -= l
        if s != a[i]:
            return False
    return True


if __name__ == '__main__':
        a = [1, -4, -5, -6, 5, 9]
    c = {(0, 1): 1, (1, 5): 3, (2, 1): 3,
         (2, 3): 5, (4, 2): 4, (4, 3): 1,
         (5, 0): -2, (5, 2): 3, (5, 4): 4}
    ub = [(0, 1), (2, 1), (2, 3), (4, 3), (5, 2)]

    x = {(0, 1): 1, (1, 5): 0, (2, 1): 3,
         (2, 3): 1, (4, 2): 0, (4, 3): 5,
         (5, 0): 0, (5, 2): 9, (5, 4): 0}
    solve_min_cost_network_flow(a, c, ub, x)
