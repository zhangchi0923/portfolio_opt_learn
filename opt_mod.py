from mosek.fusion import *

def MarkowitzWithRisk(n, mu, wb, GT, gamma, delta, industry, size, industryLimit, sizeLimit):

    with Model('Markowitz with Risk') as M:

        # Define variable
        w = M.variable('x', n, Domain.greaterThan(0.0))
        s = M.variable('s', 1, Domain.unbounded())
        active_weight = Expr.sub(w, wb)

        # Fully invested
        M.constraint('fully invested', Expr.sum(w), Domain.equalsTo(1.0))

        # Buy rule
        M.constraint('buy limit', w, Domain.lessThan(0.015))
        M.constraint('long only', w, Domain.greaterThan(0.0))

        # Compute risk
        M.constraint('variance', Expr.vstack(s, 0.5, Expr.mul(GT, active_weight)), Domain.inRotatedQCone())

        # Tracking error limit
        M.constraint('tracking error', Expr.vstack(delta, Expr.mul(GT, active_weight)), Domain.inQCone())

        # industry limit
        M.constraint('industry limit1', Expr.mul(industry, active_weight), Domain.lessThan(industryLimit))
        M.constraint('industry limit2', Expr.mul(industry, active_weight), Domain.greaterThan(-industryLimit))

        # size limit
        M.constraint('size limit1', Expr.dot(size, active_weight), Domain.lessThan(sizeLimit))
        M.constraint('size limit2', Expr.dot(size, active_weight), Domain.greaterThan(-sizeLimit))

        # objective
        active_ret = Expr.dot(mu, active_weight)
        M.objective('obj', ObjectiveSense.Maximize, Expr.sub(active_ret, Expr.mul(gamma, s)))
        M.solve()

        return w.level()
