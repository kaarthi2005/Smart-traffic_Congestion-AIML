"""Evaluate static versus dynamically rerouted emergency travel in SUMO.

Run this on a machine with SUMO/TraCI installed. It writes
`data/emergency_evaluation.csv` and exits gracefully when SUMO is unavailable.
"""
import argparse
import csv
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import networkx as nx

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET=os.path.join(BASE_DIR,"simulation","network.net.xml")
NORMAL=os.path.join(BASE_DIR,"simulation","routes.rou.xml")
OUT=os.path.join(BASE_DIR,"data","emergency_evaluation.csv")


def graph_and_routes():
    root=ET.parse(NET).getroot(); g=nx.DiGraph()
    for e in root.findall("edge"):
        eid=e.get("id"); lane=e.find("lane")
        if not eid or eid.startswith(":") or lane is None: continue
        u,v=e.get("from"),e.get("to")
        if u and v: g.add_edge(u,v,edge_id=eid,length=float(lane.get("length",1.0)),speed=float(lane.get("speed",13.89)))
    route=nx.shortest_path(g,"A0","C2",weight="length")
    edges=[g[route[i]][route[i+1]]["edge_id"] for i in range(len(route)-1)]
    return g,route,edges


def write_scenario(folder, emergency_edges, config_name):
    route_file=os.path.join(folder,"emergency_eval.rou.xml")
    route_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="emergency" vClass="emergency" accel="3.0" decel="6.0" sigma="0.2" length="5.0" minGap="2.0" maxSpeed="25.0"/>
    <route id="emergency_eval_route" edges="{" ".join(emergency_edges)}"/>
    <vehicle id="emergency_0" type="emergency" route="emergency_eval_route" depart="5" departLane="best" departSpeed="max"/>
</routes>
"""
    with open(route_file,"w",encoding="utf-8") as f:
        f.write(route_xml)
    cfg=os.path.join(folder,config_name)
    cfg_xml = f"""<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
<input><net-file value="{NET}"/><route-files value="{NORMAL},{route_file}"/></input>
<time><begin value="0"/><end value="300"/></time>
</configuration>
"""
    with open(cfg,"w",encoding="utf-8") as f:
        f.write(cfg_xml)
    return cfg

def run_case(cfg, dynamic=False, interval=5, max_steps=300):
    try: import traci
    except ImportError: return None, "TraCI is not installed"
    binary=shutil.which("sumo")
    if not binary: return None, "SUMO executable is not on PATH"
    traci.start([binary,"-c",cfg])
    start=None; arrival=None; speed_sum=0.0; speed_n=0; distance=0.0; reroutes=0; last=-interval
    try:
        g,_,_=graph_and_routes()
        for _ in range(max_steps):
            traci.simulationStep(); now=traci.simulation.getTime(); ids=traci.vehicle.getIDList()
            if "emergency_0" not in ids: continue
            if start is None: start=now
            speed=traci.vehicle.getSpeed("emergency_0"); speed_sum+=speed; speed_n+=1; distance+=speed
            if dynamic and now-last>=interval:
                current=traci.vehicle.getRoadID("emergency_0")
                if current in {d["edge_id"] for _,_,d in g.edges(data=True)}:
                    idx={d["edge_id"]:(u,v,d) for u,v,d in g.edges(data=True)}
                    update=[]
                    for eid,(u,v,d) in idx.items():
                        try: s=traci.edge.getLastStepMeanSpeed(eid); n=traci.edge.getLastStepVehicleNumber(eid)
                        except Exception: s,n=d["speed"],0
                        s=max(float(s),0.1); d["live_cost"]=(d["length"]/s)*(1+min(2.5,n/20.0))
                    _,target,_=idx[current]
                    try:
                        nodes=nx.shortest_path(g,target,"C2",weight="live_cost"); route=[current]+[g[nodes[i]][nodes[i+1]]["edge_id"] for i in range(len(nodes)-1)]
                        if route != list(traci.vehicle.getRoute("emergency_0")):
                            traci.vehicle.setRoute("emergency_0",route); reroutes+=1
                    except nx.NetworkXNoPath: pass
                last=now
            if traci.vehicle.getRoadID("emergency_0")=="-" or traci.vehicle.getRouteIndex("emergency_0")>=len(traci.vehicle.getRoute("emergency_0"))-1:
                # Continue until the vehicle actually leaves the network.
                pass
            if traci.simulation.getArrivedNumber()>0 and start is not None:
                # The emergency vehicle may not be the only arrival; check ID via departed/arrived lists where supported.
                arrived=traci.simulation.getArrivedIDList()
                if "emergency_0" in arrived: arrival=now; break
    finally: traci.close()
    if start is None: return None, "Emergency vehicle never departed"
    if arrival is None: arrival=now
    return {"travel_time_s":arrival-start,"average_speed_mps":speed_sum/max(speed_n,1),"distance_m":distance,"reroutes":reroutes}, ""


def main():
    p=argparse.ArgumentParser(); p.add_argument("--interval",type=int,default=5); args=p.parse_args()
    g,route,edges=graph_and_routes(); print("Baseline route:"," -> ".join(route))
    with tempfile.TemporaryDirectory(prefix="smart_route_eval_") as td:
        cfg=write_scenario(td,edges,"static.sumocfg"); static,err1=run_case(cfg,False,args.interval)
        cfg=write_scenario(td,edges,"dynamic.sumocfg"); dynamic,err2=run_case(cfg,True,args.interval)
    if static is None or dynamic is None:
        print("Evaluation could not run:",err1 or err2); print("Install SUMO and ensure `sumo` is on PATH, then rerun."); return
    improvement=(static["travel_time_s"]-dynamic["travel_time_s"])/max(static["travel_time_s"],1)*100
    rows=[{"strategy":"static","travel_time_s":static["travel_time_s"],"average_speed_mps":static["average_speed_mps"],"distance_m":static["distance_m"],"reroutes":0,"travel_time_improvement_pct":0.0},{"strategy":"dynamic_emergency","travel_time_s":dynamic["travel_time_s"],"average_speed_mps":dynamic["average_speed_mps"],"distance_m":dynamic["distance_m"],"reroutes":dynamic["reroutes"],"travel_time_improvement_pct":improvement}]
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,"w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(f,fieldnames=rows[0].keys()).writerows(rows)
    print("Saved:",OUT); print(f"Dynamic travel-time improvement: {improvement:.2f}%")

if __name__=="__main__": main()
