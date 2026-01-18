"""
LANGGRAPH 1.0+ RUNTIME DEMO — FIXED & VERIFIED

Demonstrates:
- explicit state
- checkpoints
- thread_id
- interrupt / resume
- resume without re-input
- state history (FIXED)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ------------------------
# STATE
# ------------------------
class State(TypedDict):
    count: int


# ------------------------
# NODES
# ------------------------
def increment(state: State):
    print("\n[NODE increment]")
    print("  incoming:", state)

    out = {"count": state["count"] + 1}

    print("  outgoing:", out)
    print("  checkpoint saved")
    return out


def pause(state: State):
    print("\n[NODE pause]")
    print("  incoming:", state)
    print("  execution paused")

    answer = interrupt("Continue? yes / no")

    print("  resumed with:", answer)

    if answer == "yes":
        print("  continuing")
        return {}
    else:
        print("  stopping")
        return Command(goto=END)


# ------------------------
# GRAPH
# ------------------------
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_node("pause", pause)

builder.add_edge(START, "increment")
builder.add_edge("increment", "pause")
builder.add_edge("pause", END)

graph = builder.compile(checkpointer=InMemorySaver())


# ------------------------
# CONFIG
# ------------------------
THREAD_ID = "thread-1"

CONFIG = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}

app = graph.with_config(CONFIG)


# ------------------------
# RUN 1
# ------------------------
print("\n=== RUN 1: START ===")
result = app.invoke({"count": 0})

if "__interrupt__" in result:
    print("\n[MAIN] INTERRUPTED:")
    print(" ", result["__interrupt__"])


# ------------------------
# RUN 2
# ------------------------
print("\n=== RUN 2: RESUME ===")
result = app.invoke(Command(resume="yes"))

print("\n[MAIN] FINAL STATE:")
print(" ", result)


# ------------------------
# RUN 3
# ------------------------
print("\n=== RUN 3: INVOKE AGAIN ===")
result = app.invoke(None)

print("\n[MAIN] STATE UNCHANGED:")
print(" ", result)


# ------------------------
# STATE HISTORY (FIXED)
# ------------------------
print("\n=== CHECKPOINT HISTORY ===")

for i, snap in enumerate(graph.get_state_history(CONFIG)):
    print(f"\nCheckpoint {i}")
    print("  next :", snap.next)
    print("  state:", snap.values)
