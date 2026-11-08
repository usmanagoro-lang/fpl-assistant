"""Match predictor: team attack/defence ratings from xG -> Poisson (Dixon-Coles) scorelines."""
import math
from collections import defaultdict

DECAY = 0.85
RHO = -0.08          # Dixon-Coles low-score correction (more 0-0 / 1-1)
MAXG = 8
SHRINK = 3.0         # pseudo-games pulling ratings toward average


def ratings(fixtures, team_xg, team_ids, before_gw, goals_weight=0.3):
    """Attack/defence multipliers using finished fixtures before `before_gw`.
    Blends xG (70%) with real goals (30%)."""
    rows = []
    for fx in fixtures:
        if not fx.get("finished") or fx.get("event") is None or fx["event"] >= before_gw:
            continue
        h, a = fx["team_h"], fx["team_a"]
        xh, xa = team_xg.get((fx["id"], h), 0.0), team_xg.get((fx["id"], a), 0.0)
        gh, ga = fx.get("team_h_score") or 0, fx.get("team_a_score") or 0
        bh = (1 - goals_weight) * xh + goals_weight * gh
        ba = (1 - goals_weight) * xa + goals_weight * ga
        rows.append((fx["event"], h, a, bh, ba))
    if not rows:
        return {t: 1.0 for t in team_ids}, {t: 1.0 for t in team_ids}, 1.35, 1.15
    home_avg = sum(r[3] for r in rows) / len(rows)
    away_avg = sum(r[4] for r in rows) / len(rows)
    lg = (home_avg + away_avg) / 2 or 1.35
    hadv = max(1.0, min(1.35, (home_avg / away_avg) if away_avg else 1.15)) ** 0.5
    att_n, att_d, def_n, def_d = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for ev, h, a, bh, ba in rows:
        w = DECAY ** max(0, before_gw - 1 - ev)
        # remove home advantage before crediting
        att_n[h] += w * bh / hadv; def_n[h] += w * ba * hadv
        att_n[a] += w * ba * hadv; def_n[a] += w * bh / hadv
        att_d[h] += w; att_d[a] += w; def_d[h] += w; def_d[a] += w
    att = {t: (att_n[t] / lg + SHRINK) / (att_d[t] + SHRINK) for t in team_ids}
    dfn = {t: (def_n[t] / lg + SHRINK) / (def_d[t] + SHRINK) for t in team_ids}
    return att, dfn, lg, hadv


def _pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def scoreline(lh, la):
    m = [[_pois(i, lh) * _pois(j, la) for j in range(MAXG)] for i in range(MAXG)]
    m[0][0] *= 1 - lh * la * RHO
    m[0][1] *= 1 + lh * RHO
    m[1][0] *= 1 + la * RHO
    m[1][1] *= 1 - RHO
    s = sum(map(sum, m))
    return [[x / s for x in row] for row in m]


def predict(h, a, att, dfn, lg, hadv):
    lh = lg * att[h] * dfn[a] * hadv
    la = lg * att[a] * dfn[h] / hadv
    m = scoreline(lh, la)
    ph = sum(m[i][j] for i in range(MAXG) for j in range(MAXG) if i > j)
    pa = sum(m[i][j] for i in range(MAXG) for j in range(MAXG) if i < j)
    pd = 1 - ph - pa
    res = max((("H", ph), ("D", pd), ("A", pa)), key=lambda x: x[1])[0]
    cond = {"H": lambda i, j: i > j, "D": lambda i, j: i == j, "A": lambda i, j: i < j}[res]
    best = max(((i, j) for i in range(MAXG) for j in range(MAXG) if cond(i, j)), key=lambda s: m[s[0]][s[1]])
    o25 = sum(m[i][j] for i in range(MAXG) for j in range(MAXG) if i + j > 2)
    btts = sum(m[i][j] for i in range(1, MAXG) for j in range(1, MAXG))
    top = sorted(((m[i][j], f"{i}-{j}") for i in range(MAXG) for j in range(MAXG)), reverse=True)[:3]
    return dict(xh=round(lh, 2), xa=round(la, 2), ph=round(ph, 3), pd=round(pd, 3), pa=round(pa, 3),
                pick=res, score=f"{best[0]}-{best[1]}", o25=round(o25, 3), btts=round(btts, 3),
                top=[(s, round(p, 3)) for p, s in top], matrix=m)


def outcome(gh, ga):
    return "H" if gh > ga else "A" if gh < ga else "D"


def backtest(fixtures, team_xg, team_ids, from_gw, to_gw):
    """Honest walk-forward test: each GW predicted only with data from before it."""
    n = hit = exact = 0
    brier = 0.0
    per_gw = []
    for g in range(from_gw, to_gw + 1):
        att, dfn, lg, hadv = ratings(fixtures, team_xg, team_ids, g)
        gn = gh_ = 0
        for fx in fixtures:
            if fx.get("event") != g or not fx.get("finished"):
                continue
            p = predict(fx["team_h"], fx["team_a"], att, dfn, lg, hadv)
            o = outcome(fx["team_h_score"], fx["team_a_score"])
            n += 1; gn += 1
            if p["pick"] == o:
                hit += 1; gh_ += 1
            if p["score"] == f"{fx['team_h_score']}-{fx['team_a_score']}":
                exact += 1
            brier += sum((p[k] - (1 if o == r else 0)) ** 2 for k, r in (("ph", "H"), ("pd", "D"), ("pa", "A")))
        if gn:
            per_gw.append((g, gh_, gn))
    if not n:
        return None
    return dict(matches=n, result_pct=round(100 * hit / n, 1), exact_pct=round(100 * exact / n, 1),
                brier=round(brier / n, 3), per_gw=per_gw, from_gw=from_gw, to_gw=to_gw)
