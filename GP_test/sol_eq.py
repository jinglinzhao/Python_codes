def solve_kep_eqn(l,e):
    """ Solve Keplers equation x - e*sin(x) = l for x"""
    try:
        l[0]
        res = np.zeros(l.shape)
        for i,li in enumerate(l):
            tmp,= fsolve(lambda x: x-e*np.sin(x) - li,li)
            res[i] = tmp
    except IndexError:
        res, = fsolve(lambda x: x - e*np.sin(x)-l,l)

    return res