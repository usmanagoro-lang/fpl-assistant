"""FPL Assistant — weekly expected-points model, transfer and captain picks.

Pulls the official FPL API, projects points for the next HORIZON gameweeks,
reads your squad and writes a single HTML page to site/index.html.
"""
import json, math, os, time, datetime as dt, urllib.request, html
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = "https://fantasy.premierleague.com/api/"
TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "4733331"))
HORIZON = 5
WEIGHTS = [1.0, 0.85, 0.72, 0.6, 0.5]          # how much each future GW counts
DIFF_OWNERSHIP = 10.0                           # % owned below which a pick is a "differential"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
DC_THRESH = {2: 10, 3: 12, 4: 12}
PRIOR_XG90 = {1: 0.0, 2: 0.04, 3: 0.12, 4: 0.30}
PRIOR_XA90 = {1: 0.01, 2: 0.06, 3: 0.12, 4: 0.10}
PSEUDO_MIN = 180.0
DECAY = 0.85


def get(path, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(BASE + path, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fpl-assistant",
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"Failed {path}: {e}")
            time.sleep(2 * (i + 1))


def f(x, d=0.0):
    try:
        return float(x) if x is not None else d
    except (TypeError, ValueError):
        return d


def main():
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    events = boot["events"]
    teams = {t["id"]: t for t in boot["teams"]}
    players = {p["id"]: p for p in boot["elements"] if p["element_type"] in POS}

    cur = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    cur_gw = cur["id"] if cur else 0
    next_gw = nxt["id"] if nxt else cur_gw + 1
    gws = [g for g in range(next_gw, next_gw + HORIZON) if g <= len(events)]

    # ---------- player histories (only players who have played) ----------
    ids = [pid for pid, p in players.items() if p.get("minutes", 0) > 0]
    hist = {}

    def load(pid):
        try:
            return pid, get(f"element-summary/{pid}/")["history"]
        except Exception:
            return pid, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, h in ex.map(load, ids):
            hist[pid] = h

    # ---------- team attack / defence ratings from xG ----------
    fx_by_id = {fx["id"]: fx for fx in fixtures}
    team_xg = defaultdict(float)              # (fixture, team) -> xG
    for pid, h in hist.items():
        tid = players[pid]["team"]
        for r in h:
            team_xg[(r["fixture"], tid)] += f(r.get("expected_goals"))
    xgf, xga = defaultdict(list), defaultdict(list)   # team -> [(round, xg)]
    for fx in fixtures:
        if not fx.get("finished") or fx.get("event") is None:
            continue
        h_, a_ = fx["team_h"], fx["team_a"]
        gh, ga = team_xg.get((fx["id"], h_), 0.0), team_xg.get((fx["id"], a_), 0.0)
        xgf[h_].append((fx["event"], gh)); xga[h_].append((fx["event"], ga))
        xgf[a_].append((fx["event"], ga)); xga[a_].append((fx["event"], gh))
    all_xg = [x for v in xgf.values() for _, x in v]
    lg = (sum(all_xg) / len(all_xg)) if all_xg else 1.35
    lg = lg or 1.35

    def wavg(rows, prior, k=3.0):
        num, den = prior * k, k
        for rnd, x in rows:
            w = DECAY ** max(0, cur_gw - rnd)
            num += w * x; den += w
        return num / den
    att = {t: wavg(xgf[t], lg) / lg for t in teams}
    dfn = {t: wavg(xga[t], lg) / lg for t in teams}   # >1 = leaky

    # fixtures per team per GW
    team_fx = defaultdict(lambda: defaultdict(list))  # team -> gw -> [(opp, home)]
    for fx in fixtures:
        if fx.get("event") in gws:
            team_fx[fx["team_h"]][fx["event"]].append((fx["team_a"], True))
            team_fx[fx["team_a"]][fx["event"]].append((fx["team_h"], False))

    # ---------- per-player model ----------
    model = {}
    for pid, p in players.items():
        pos = p["element_type"]
        h = sorted(hist.get(pid, []), key=lambda r: (r["round"], r.get("kickoff_time") or ""))
        wmin = wxg = wxa = wbon = wsav = 0.0
        dc_hits = dc_apps = 0.0
        for r in h:
            w = DECAY ** max(0, cur_gw - r["round"])
            m = f(r.get("minutes"))
            wmin += w * m
            wxg += w * f(r.get("expected_goals"))
            wxa += w * f(r.get("expected_assists"))
            wbon += w * f(r.get("bonus"))
            wsav += w * f(r.get("saves"))
            if m >= 60 and pos in DC_THRESH and "defensive_contribution" in r:
                dc_apps += w
                if f(r.get("defensive_contribution")) >= DC_THRESH[pos]:
                    dc_hits += w
        xg90 = (wxg + PRIOR_XG90[pos] * PSEUDO_MIN / 90) / (wmin + PSEUDO_MIN) * 90
        xa90 = (wxa + PRIOR_XA90[pos] * PSEUDO_MIN / 90) / (wmin + PSEUDO_MIN) * 90
        bon90 = (wbon + 0.2 * PSEUDO_MIN / 90) / (wmin + PSEUDO_MIN) * 90
        sav90 = (wsav / wmin * 90) if wmin > 0 else 2.5
        dc_rate = (dc_hits + 0.2) / (dc_apps + 1.0) if pos in DC_THRESH else 0.0

        # expected minutes: recent 5 team fixtures (zeros count)
        recent = h[-5:]
        if recent:
            ws = [DECAY ** i for i in range(len(recent))][::-1]
            xmin = sum(w * f(r.get("minutes")) for w, r in zip(ws, recent)) / sum(ws)
        else:
            xmin = 0.0
        status = p.get("status", "a")
        chance = p.get("chance_of_playing_next_round")
        c0 = 1.0 if chance is None else chance / 100.0
        if status in ("u", "n") or (status == "i" and chance is None):
            c0 = 0.0

        proj = {}
        for k, gw in enumerate(gws):
            if status == "u":
                avail = 0.0
            elif status == "s":
                avail = c0 if k == 0 else 1.0
            else:
                avail = c0 + (1 - c0) * min(1.0, k / 3.0)
            m = xmin * avail
            p60 = min(1.0, m / 80.0)
            pany = 0.0 if m <= 0 else max(p60, min(1.0, m / 55.0))
            pts = 0.0
            for opp, home in team_fx[p["team"]].get(gw, []):
                hf = 1.08 if home else 0.93
                amult = dfn[opp] * hf
                lam = lg * dfn[p["team"]] * att[opp] * (2 - hf)   # goals my team concedes
                frac = m / 90.0
                e = 2 * p60 + 1 * (pany - p60)
                e += frac * xg90 * amult * GOAL_PTS[pos]
                e += frac * xa90 * amult * 3
                e += frac * bon90 * math.sqrt(amult)
                e += p60 * CS_PTS[pos] * math.exp(-lam)
                if pos in (1, 2):
                    e -= p60 * 0.5 * lam * 0.9
                if pos == 1:
                    e += frac * sav90 * att[opp] * (2 - hf) / 3.0
                if pos in DC_THRESH:
                    e += p60 * 2 * dc_rate
                pts += e
            proj[gw] = round(pts, 2)
        val = sum(WEIGHTS[i] * proj[g] for i, g in enumerate(gws))
        model[pid] = dict(proj=proj, val=round(val, 2), xmin=round(xmin), xg90=round(xg90, 2),
                          xa90=round(xa90, 2))

    # ---------- your squad ----------
    entry = get(f"entry/{TEAM_ID}/")
    ehist = get(f"entry/{TEAM_ID}/history/")
    transfers = get(f"entry/{TEAM_ID}/transfers/")
    pick_gw = entry.get("current_event") or cur_gw
    picks = get(f"entry/{TEAM_ID}/event/{pick_gw}/picks/")
    eh = picks.get("entry_history", {})
    bank = eh.get("bank", entry.get("last_deadline_bank", 0) or 0)

    # free transfers (approximate: 1/GW, bank up to 5, WC/FH keep them)
    chips = {c["event"]: c["name"] for c in ehist.get("chips", [])}
    ft = 1
    for row in sorted(ehist.get("current", []), key=lambda r: r["event"]):
        g = row["event"]
        if g == 1:
            ft = 1
            continue
        if chips.get(g) in ("wildcard", "freehit"):
            pass
        else:
            ft = max(0, ft - row.get("event_transfers", 0))
        ft = min(5, ft + 1)
    if ehist.get("current") and ehist["current"][-1]["event"] < cur_gw:
        ft = min(5, ft + (cur_gw - ehist["current"][-1]["event"]))

    # selling prices
    bought = {}
    for t in sorted(transfers, key=lambda t: t["event"]):
        bought[t["element_in"]] = t["element_in_cost"]

    def purchase_price(pid):
        if pid in bought:
            return bought[pid]
        h = hist.get(pid) or []
        return h[0]["value"] if h else players[pid]["now_cost"]

    squad = []
    for pk in picks["picks"]:
        pid = pk["element"]
        if pid not in players:
            continue
        now = players[pid]["now_cost"]; pp = purchase_price(pid)
        sell = now if now <= pp else pp + (now - pp) // 2
        squad.append(dict(id=pid, sell=sell, pos=pk["position"]))
    squad_ids = {s["id"] for s in squad}
    team_count = defaultdict(int)
    for s in squad:
        team_count[players[s["id"]]["team"]] += 1

    def nm(pid):
        return players[pid]["web_name"]

    def V(pid):
        return model[pid]["val"]

    # ---------- squad scoring: best XI + captain each GW, bench barely counts ----------
    def xi_points(ids, gw):
        by = defaultdict(list)
        for pid in ids:
            by[players[pid]["element_type"]].append(model[pid]["proj"].get(gw, 0))
        for k in by:
            by[k].sort(reverse=True)
        best = 0.0; best_set = None
        for d in range(3, 6):
            for m in range(2, 6):
                fw = 10 - d - m
                if fw < 1 or fw > 3 or d > len(by[2]) or m > len(by[3]) or fw > len(by[4]) or not by[1]:
                    continue
                xi = by[1][:1] + by[2][:d] + by[3][:m] + by[4][:fw]
                tot = sum(xi) + max(xi)          # captain doubles
                if tot > best:
                    best, best_set = tot, xi
        total = sum(model[p]["proj"].get(gw, 0) for p in ids)
        return best + 0.1 * (total - (sum(best_set) if best_set else 0))

    def squad_value(ids):
        return sum(WEIGHTS[k] * xi_points(ids, g) for k, g in enumerate(gws))

    base_val = squad_value(list(squad_ids))
    cands = {k: sorted([i for i, p in players.items() if p["element_type"] == k and i not in squad_ids
                        and p.get("status") not in ("u", "i", "s", "n")], key=lambda i: -V(i))[:45] for k in POS}

    # ---------- transfers ----------
    singles = []
    for s in squad:
        o = s["id"]; po = players[o]
        budget = bank + s["sell"]
        for i in cands[po["element_type"]]:
            pi = players[i]
            if pi["now_cost"] > budget:
                continue
            if pi["team"] != po["team"] and team_count[pi["team"]] >= 3:
                continue
            new_ids = [x for x in squad_ids if x != o] + [i]
            g = squad_value(new_ids) - base_val
            if g > 0.3:
                singles.append(dict(out=o, inn=i, gain=round(g, 2),
                                    left=budget - pi["now_cost"]))
    singles.sort(key=lambda x: -x["gain"])

    def varied(lst, cap=2):
        seen_o, seen_i, out = defaultdict(int), defaultdict(int), []
        for t in lst:
            if seen_o[t["out"]] < cap and seen_i[t["inn"]] < cap:
                out.append(t); seen_o[t["out"]] += 1; seen_i[t["inn"]] += 1
        return out
    diffs = varied([t for t in singles if f(players[t["inn"]]["selected_by_percent"]) < DIFF_OWNERSHIP])
    show_singles = varied(singles)

    doubles = []
    top = singles[:40]
    sell_of = {s["id"]: s["sell"] for s in squad}
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            x, y = top[a], top[b]
            if x["out"] == y["out"] or x["inn"] == y["inn"]:
                continue
            cost = players[x["inn"]]["now_cost"] + players[y["inn"]]["now_cost"]
            if cost > bank + sell_of[x["out"]] + sell_of[y["out"]]:
                continue
            tc = dict(team_count)
            for o in (x["out"], y["out"]):
                tc[players[o]["team"]] -= 1
            ok = True
            for i in (x["inn"], y["inn"]):
                tc[players[i]["team"]] = tc.get(players[i]["team"], 0) + 1
                if tc[players[i]["team"]] > 3:
                    ok = False
            if not ok:
                continue
            hit = 0 if ft >= 2 else 4
            new_ids = [z for z in squad_ids if z not in (x["out"], y["out"])] + [x["inn"], y["inn"]]
            g = squad_value(new_ids) - base_val
            doubles.append(dict(moves=[x, y], gain=round(g, 2), hit=hit, net=round(g - hit, 2)))
    doubles.sort(key=lambda d: -d["net"])
    doubles = doubles[:6]

    # ---------- best XI + captain for next GW ----------
    g0 = gws[0] if gws else None
    nx = lambda pid: model[pid]["proj"].get(g0, 0) if g0 else 0
    by = defaultdict(list)
    for s in squad:
        by[players[s["id"]]["element_type"]].append(s["id"])
    for k in by:
        by[k].sort(key=lambda pid: -nx(pid))
    best, best_xi = -1, None
    for d in range(3, 6):
        for m in range(2, 6):
            fw = 10 - d - m
            if fw < 1 or fw > 3 or d > len(by[2]) or m > len(by[3]) or fw > len(by[4]):
                continue
            xi = by[1][:1] + by[2][:d] + by[3][:m] + by[4][:fw]
            tot = sum(nx(p) for p in xi)
            if tot > best:
                best, best_xi = tot, xi
    best_xi = best_xi or [s["id"] for s in squad[:11]]
    bench = [s["id"] for s in squad if s["id"] not in best_xi]
    bench_gk = [p for p in bench if players[p]["element_type"] == 1]
    bench_out = sorted([p for p in bench if players[p]["element_type"] != 1], key=lambda p: -nx(p))
    caps = sorted(best_xi, key=lambda p: -nx(p))[:4]

    # ---------- headline advice ----------
    best_single = singles[0] if singles else None
    best_double = doubles[0] if doubles else None
    if best_double and best_double["hit"] and best_single and best_double["net"] > best_single["gain"] + 2.0:
        advice = ("Take the hit", f"The double move below gains about {best_double['gain']:.1f} pts over {len(gws)} GWs, "
                  f"so it beats the –4 by {best_double['net']:.1f}.")
    elif best_double and not best_double["hit"] and best_single and best_double["gain"] > best_single["gain"] + 1.0:
        advice = ("Use both free transfers", f"Two moves gain about {best_double['gain']:.1f} pts with no hit.")
    elif best_single and best_single["gain"] >= 1.5:
        advice = ("Make one transfer", f"{nm(best_single['out'])} → {nm(best_single['inn'])} gains about "
                  f"{best_single['gain']:.1f} pts over the next {len(gws)} GWs.")
    else:
        advice = ("Roll the transfer", "No move clearly beats your current squad. Bank it for next week"
                  + (" (you're at the 5 FT cap, so a small upgrade is fine too)." if ft >= 5 else "."))

    # ---------- price watch ----------
    netx = sorted(players.values(), key=lambda p: -(p.get("transfers_in_event", 0) - p.get("transfers_out_event", 0)))
    risers = netx[:10]
    fallers = netx[::-1][:10]

    # ---------- build data for page ----------
    def prow(pid, extra=None):
        p = players[pid]; m = model[pid]
        d = dict(id=pid, name=p["web_name"], team=teams[p["team"]]["short_name"], pos=POS[p["element_type"]],
                 price=p["now_cost"] / 10, own=f(p["selected_by_percent"]), form=f(p.get("form")),
                 pts=p.get("total_points", 0), xmin=m["xmin"], xg90=m["xg90"], xa90=m["xa90"],
                 proj=[m["proj"].get(g, 0) for g in gws], val=m["val"], status=p.get("status", "a"),
                 news=p.get("news", ""), chance=p.get("chance_of_playing_next_round"),
                 fx=[" ".join(teams[o]["short_name"] + ("(H)" if h else "(A)") for o, h in team_fx[p["team"]].get(g, [])) or "—"
                     for g in gws],
                 netx=p.get("transfers_in_event", 0) - p.get("transfers_out_event", 0))
        if extra:
            d.update(extra)
        return d

    def tmove(t):
        return dict(out=prow(t["out"]), inn=prow(t["inn"]), gain=t["gain"], left=t["left"] / 10)

    deadline = nxt["deadline_time"] if nxt else None
    data = dict(
        generated=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        team=dict(name=entry.get("name"), manager=f"{entry.get('player_first_name','')} {entry.get('player_last_name','')}".strip(),
                  points=entry.get("summary_overall_points"), rank=entry.get("summary_overall_rank"),
                  gw_points=entry.get("summary_event_points"), bank=bank / 10, ft=ft,
                  value=eh.get("value", 0) / 10),
        gws=gws, next_gw=next_gw, deadline=deadline, advice=advice,
        squad=[prow(s["id"], dict(sell=s["sell"] / 10, xi=s["id"] in best_xi)) for s in squad],
        lineup=dict(xi=[prow(p) for p in sorted(best_xi, key=lambda p: players[p]["element_type"])],
                    bench=[prow(p) for p in bench_gk + bench_out], captains=[prow(p) for p in caps]),
        singles=[tmove(t) for t in show_singles[:15]],
        diffs=[tmove(t) for t in diffs[:15]],
        doubles=[dict(moves=[tmove(m) for m in d["moves"]], gain=d["gain"], hit=d["hit"], net=d["net"]) for d in doubles],
        top={POS[k]: [prow(pid) for pid in sorted([i for i, p in players.items() if p["element_type"] == k],
                                                    key=lambda i: -V(i))[:20]] for k in POS},
        risers=[prow(p["id"]) for p in risers], fallers=[prow(p["id"]) for p in fallers],
        my_fallers=[prow(p["id"]) for p in fallers if p["id"] in squad_ids],
    )
    os.makedirs("site", exist_ok=True)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
    open("site/index.html", "w", encoding="utf-8").write(out)
    json.dump(data, open("site/data.json", "w"))
    print("OK", data["team"], data["advice"])
    mv = lambda t: f"{t['out']['name']}({t['out']['status']}) -> {t['inn']['name']} {t['inn']['team']} £{t['inn']['price']} {t['inn']['own']}% +{t['gain']}"
    print("SQUAD", [(p["name"], p["pos"], p["status"], round(p["val"], 1)) for p in data["squad"]])
    print("CAPTAINS", [(p["name"], p["proj"][0]) for p in data["lineup"]["captains"]])
    print("SINGLES", [mv(t) for t in data["singles"][:6]])
    print("DIFFS", [mv(t) for t in data["diffs"][:6]])
    print("DOUBLES", [([mv(m) for m in d["moves"]], d["net"]) for d in data["doubles"][:3]])
    print("TOPMID", [(p["name"], p["val"]) for p in data["top"]["MID"][:8]])
    print("TOPFWD", [(p["name"], p["val"]) for p in data["top"]["FWD"][:6]])


if __name__ == "__main__":
    main()
