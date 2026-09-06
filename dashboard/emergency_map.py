"""Standalone emergency route map for the Smart Traffic project."""
import os, xml.etree.ElementTree as ET
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET=os.path.join(BASE_DIR,"simulation","network.net.xml")
ROUTE=["A0","A1","A2","B2","C2"]

@st.cache_data
def load_network():
    root=ET.parse(NET).getroot(); g=nx.DiGraph(); pos={}
    for j in root.findall("junction"):
        if not j.get("id","").startswith(":"):
            pos[j.get("id")]=(float(j.get("x",0)),float(j.get("y",0)))
    for e in root.findall("edge"):
        eid=e.get("id"); u=e.get("from"); v=e.get("to")
        if eid and not eid.startswith(":") and u and v: g.add_edge(u,v,edge_id=eid)
    return g,pos

st.set_page_config(page_title="Emergency Route Map",page_icon="🚑",layout="wide")
st.title("🚑 Emergency Route Visualization")
st.caption("SUMO junction coordinates + AI route overlay")
g,pos=load_network()
fig,ax=plt.subplots(figsize=(10,7))
nx.draw_networkx_edges(g,pos,ax=ax,alpha=.25,arrows=True)
nx.draw_networkx_nodes(g,pos,ax=ax,node_size=900)
nx.draw_networkx_labels(g,pos,ax=ax,font_weight="bold")
red_edges=[(ROUTE[i],ROUTE[i+1]) for i in range(len(ROUTE)-1)]
nx.draw_networkx_edges(g,pos,ax=ax,edgelist=red_edges,width=5,arrows=True,arrowsize=20)
nx.draw_networkx_nodes(g,pos,ax=ax,nodelist=[ROUTE[0]],node_size=1200)
nx.draw_networkx_nodes(g,pos,ax=ax,nodelist=[ROUTE[-1]],node_size=1200)
ax.set_title("Emergency vehicle route: A0 → C2"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
st.info("Green/start and red/destination markers are represented by the highlighted endpoints. The route overlay can be replaced by live TraCI positions in a future deployment.")
