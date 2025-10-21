import polars as ps
import networkx as nx
import pulp as pl
from itertools import islice
import os
from argparse import ArgumentParser

def load_data(inputs_path):
    dist_df = ps.read_csv(
        os.path.join(inputs_path, "distance_matrix.csv")
    )
    office_df = ps.read_csv(
        os.path.join(inputs_path, "offices.csv")
    )
    reqs_df = ps.read_csv(
        os.path.join(inputs_path, "reqs.csv")
    )

    G = nx.DiGraph()
    for _, row in dist_df.iterrows():
        u, v = row["src"], row["dst"]
        G.add_edge(u, v, price=float(row["price"]))

    transfer_price = office_df.set_index("office_id")["transfer_price"].to_dict()
    transfer_max = office_df.set_index("office_id")["transfer_max"].to_dict()

    reqs = []
    for idx, r in reqs_df.iterrows():
        reqs.append(
            {
                "id": int(idx),
                "s": r["src_office_id"],
                "t": r["dst_office_id"],
                "d": float(r["volume"]),
            }
        )

    return G, reqs, transfer_price, transfer_max


def transit_cost_of_path(path, transfer_price_map):
    return sum(transfer_price_map.get(v, 0.0) for v in path[1:-1])

def k_shortest_simple_paths(G_in, source, target, k=1, weight="weight"):
    try:
        gen = nx.shortest_simple_paths(G_in, source, target, weight=weight)
        return list(islice(gen, k))
    except Exception:
        return []


def extract_duals(model, demand_constr, edgecap_constr, node_in_constr, lp_tl):

    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=lp_tl, warmStart=True, keepFiles=True)
    model.setSolver(solver)
    model.solve()

    status = pl.LpStatus[model.status]
    if status != "Optimal":

        if status == "Infeasible":
            print("Model is infeasible. Writing LP file for debugging.")
            model.write("master_lp.lp")
        raise RuntimeError(f"Master LP solve failed or not optimal. Status: {status}")

    pi = {}
    beta = {}
    gamma = {}

    for k in demand_constr:
        constr = model.constraints[f"demand_{k}"]
        pi[k] = constr.pi if constr.pi is not None else 0.0
    for e in edgecap_constr:
        constr = model.constraints[f"edgecap_{e[0]}_{e[1]}"]
        beta[e] = constr.pi if constr.pi is not None else 0.0
    for v in node_in_constr:
        constr = model.constraints[f"node_in_{v}"]
        gamma[v] = constr.pi if constr.pi is not None else 0.0

    obj_val = pl.value(model.objective) if model.objective is not None else float("inf")
    return pi, beta, gamma, obj_val


def pricing_for_request(G, rk, pi, beta, gamma, transfer_price):
    s, t = rk["s"], rk["t"]
    H = nx.DiGraph()
    for u, v in G.edges():
        be = beta.get((u, v), 0.0)
        node_cost = 0.0
        if v != t:
            node_cost = transfer_price.get(v, 0.0) + gamma.get(v, 0.0)
        w = be + node_cost
        H.add_edge(u, v, weight=w)
    try:
        p = nx.shortest_path(H, s, t, weight="weight")
    except Exception:
        return None, None
    path_edges = list(zip(p[:-1], p[1:]))
    sum_beta = sum(beta.get(e, 0.0) for e in path_edges)
    sum_gamma = sum(gamma.get(v, 0.0) for v in p[1:-1])
    path_transit = sum(transfer_price.get(v, 0.0) for v in p[1:-1])
    pi_k = pi[rk["id"]]
    return p, (path_transit + sum_beta + sum_gamma - pi_k)


def build_initial_master(
    G, reqs, paths_for_req, C, transfer_price, transfer_max, TIME_LIMIT_LP=600
):

    prob = pl.LpProblem("master_lp", pl.LpMinimize)

    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT_LP, warmStart=True)
    prob.setSolver(solver)

    y_vars = {}
    for u, v in G.edges():
        y = pl.LpVariable(f"y_{u}_{v}", lowBound=0, cat="Continuous")
        y_vars[(u, v)] = y

    x_vars = {}
    demand_constr = {}
    edgecap_constr = {}
    node_in_constr = {}

    for r in reqs:
        rid = r["id"]
        s_r = r["s"]
        t_r = r["t"]
        d_r = r["d"]

        demand_expr = pl.LpAffineExpression()

        for pid, path in enumerate(paths_for_req[rid]):
            transit_cost = sum(transfer_price.get(v, 0.0) for v in path[1:-1])
            x = pl.LpVariable(f"x_{rid}_{pid}", lowBound=0, cat="Continuous")
            x_vars[(rid, pid)] = {"var": x, "path": path}

            demand_expr += x

            for e in zip(path[:-1], path[1:]):
                if e not in edgecap_constr:
                    edgecap_constr[e] = pl.LpAffineExpression()
                edgecap_constr[e] += x

            for v in path[1:]:
                if v != s_r and v != t_r:
                    if v not in node_in_constr:
                        node_in_constr[v] = pl.LpAffineExpression()
                    node_in_constr[v] += x

        demand_constr[rid] = prob.addConstraint(
            demand_expr == d_r, name=f"demand_{rid}"
        )

    for (u, v), expr in edgecap_constr.items():
        edgecap_constr[(u, v)] = prob.addConstraint(
            expr <= C * y_vars[(u, v)], name=f"edgecap_{u}_{v}"
        )

    for v, expr in node_in_constr.items():
        cap = transfer_max.get(v, float("inf"))
        node_in_constr[v] = prob.addConstraint(expr <= cap, name=f"node_in_{v}")

    obj = pl.LpAffineExpression()
    for (u, v), y in y_vars.items():
        obj += G[u][v]["price"] * y
    for (rid, pid), rec in x_vars.items():
        x = rec["var"]
        transit_cost = sum(transfer_price.get(v, 0.0) for v in rec["path"][1:-1])
        obj += transit_cost * x
    prob.setObjective(obj)

    return prob, x_vars, y_vars, demand_constr, edgecap_constr, node_in_constr


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory with reqs.csv, distance_matrix.csv, offices.csv",
    )
    parser.add_argument(
        "-s",
        "--solution-save_path",
        type=str,
        required=True,
        help="Path to save solution csv",
    )
    parser.add_argument(
        "-C",
        "--truck-capacity",
        type=int,
        default=90,
        help="Allowed goods number per truck",
    )
    parser.add_argument(
        "-e", "--eps", type=float, default=1e-8, help="eps for converging algorithms"
    )
    parser.add_argument(
        "-k",
        "--k-init-paths",
        type=int,
        default=5,
        help="Use k shortest on master init stage",
    )
    parser.add_argument(
        "-K",
        "--max-cg-iters",
        type=int,
        default=200,
        help="Max column-generation iters",
    )
    parser.add_argument(
        "-M", "--time-limit-lp", type=int, default=300, help="Time limit for LP [s]"
    )
    parser.add_argument(
        "-N", "--time-limit-mip", type=int, default=600, help="Time limit for MIP [s]"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    G, reqs, transfer_price, transfer_max = load_data(args.input_dir)

    paths_for_req = {}

    for r in reqs:
        s, t = r["s"], r["t"]

        H = nx.DiGraph()
        for u, v, dat in G.edges(data=True):
            node_transit = transfer_price.get(v, 0.0) if v != t else 0.0
            w = dat["price"] + node_transit
            H.add_edge(u, v, weight=w, price=dat["price"])
        sel = k_shortest_simple_paths(H, s, t, k=args.k_init_paths, weight="weight")
        if H.has_edge(s, t):
            sel.append([s, t])
        if len(sel) == 0:
            raise ValueError(f"No path found for request {r['id']} from {s} to {t}")
        paths_for_req[r["id"]] = sel

    master, x_vars, y_vars, demand_constr, edgecap_constr, node_in_constr = (
        build_initial_master(
            G, reqs, paths_for_req, args.truck_capacity, transfer_price, transfer_max
        )
    )

    iter_count = 0
    while True:
        iter_count += 1
        print(f"\n=== CG iteration {iter_count} ===")

        pi, beta, gamma, lp_obj = extract_duals(
            master, demand_constr, edgecap_constr, node_in_constr, args.time_limit_lp
        )
        print(f"Master LP objective: {lp_obj:.6f}")

        new_columns = 0
        for r in reqs:
            rid = r["id"]
            p, path_cost_sum = pricing_for_request(
                G, r, pi, beta, gamma, transfer_price
            )
            if p is None:
                continue

            redcost = path_cost_sum - pi[rid]
            print(f"req {rid}: best path {p} reduced cost = {redcost:.6e}")

            if redcost < -args.eps:
                x = pl.LpVariable(
                    f"x_{rid}_cg{iter_count}", lowBound=0, cat="Continuous"
                )
                x_vars[(rid, f"cg{iter_count}")] = {"var": x, "path": p}

                demand_expr = pl.LpAffineExpression()
                for (rid2, pid), rec in x_vars.items():
                    if rid2 == rid:
                        demand_expr += rec["var"]
                master.constraints[f"demand_{rid}"] = demand_expr == r["d"]

                for e in zip(p[:-1], p[1:]):
                    if e not in edgecap_constr:
                        edgecap_constr[e] = pl.LpAffineExpression()
                        edgecap_constr[e] = master.addConstraint(
                            edgecap_constr[e] <= args.truck_capacity * y_vars[e],
                            name=f"edgecap_{e[0]}_{e[1]}",
                        )
                    edgecap_expr = pl.LpAffineExpression()
                    for (rid2, pid), rec in x_vars.items():
                        path = rec["path"]
                        if e in list(zip(path[:-1], path[1:])):
                            edgecap_expr += rec["var"]
                    master.constraints[f"edgecap_{e[0]}_{e[1]}"] = (
                        edgecap_expr <= args.truck_capacity * y_vars[e]
                    )

                for v in p[1:]:
                    if v != r["s"] and v != r["t"]:
                        if v not in node_in_constr:
                            node_in_constr[v] = pl.LpAffineExpression()
                            cap = transfer_max.get(v, float("inf"))
                            node_in_constr[v] = master.addConstraint(
                                node_in_constr[v] <= cap, name=f"node_in_{v}"
                            )
                        node_in_expr = pl.LpAffineExpression()
                        for (rid2, pid), rec in x_vars.items():
                            path = rec["path"]
                            if v in path[1:-1]:
                                node_in_expr += rec["var"]
                        master.constraints[f"node_in_{v}"] = (
                            node_in_expr <= transfer_max.get(v, float("inf"))
                        )

                obj = pl.LpAffineExpression()
                for (u, v), y in y_vars.items():
                    obj += G[u][v]["price"] * y
                for (rid, pid), rec in x_vars.items():
                    x_var = rec["var"]
                    transit_cost = sum(
                        transfer_price.get(v, 0.0) for v in rec["path"][1:-1]
                    )
                    obj += transit_cost * x_var
                master.setObjective(obj)

                new_columns += 1
                print(f"  -> Added column for req {rid}, path {p}")

        if new_columns == 0:
            print("No columns with negative reduced cost found. Stopping CG.")
            break
        if iter_count >= args.max_cg_iters:
            print("Reached max CG iterations. Stopping.")
            break

    solution_ll = []

    master.solve()
    print("Final master LP objective:", pl.value(master.objective))

    print("\nSolving final MILP on generated columns (make y integer)...")
    for e, yv in y_vars.items():
        yv.cat = "Integer"
    solver = pl.PULP_CBC_CMD(msg=1, timeLimit=args.time_limit_mip, warmStart=True)
    master.setSolver(solver)
    master.solve()

    if pl.LpStatus[master.status] in ["Optimal", "Not Solved"]:

        sol_x = {}

        for (rid, pid), rec in x_vars.items():
            var = rec["var"]
            val = pl.value(var) if pl.value(var) is not None else 0.0
            if val > 1e-8:
                sol_x[(rid, pid)] = {"flow": val, "path": rec["path"]}

        for (rid, pid), rec in sol_x.items():
            solution_ll.append(
                [
                    int(rec["path"][0]),
                    int(rec["path"][-1]),
                    float(rec["flow"]),
                    [int(v) for v in rec["path"]],
                ]
            )
    else:
        raise RuntimeError(
            f"MIP didn't return feasible/optimal solution; status: {pl.LpStatus[master.status]}"
        )

    return solution_ll


if __name__ == "__main__":
    main()
