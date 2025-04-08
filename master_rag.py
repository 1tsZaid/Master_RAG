from typing import TypedDict
from langgraph.graph import StateGraph, END
import json, re
from rag_chains import router_chain, default_chain, destination_chains


class GraphState(TypedDict):
    input: str
    destination: str
    next_inputs: str
    output: str

class Agent():
    def __init__(self):
        builder = StateGraph(GraphState)
        builder.add_node("router", self.route_question)
        builder.add_node("chain_runner", self.run_chain)
        builder.set_entry_point("router")
        builder.add_edge("router", "chain_runner")
        builder.add_edge("chain_runner", END)
        
        self.graph = builder.compile()

    def route_question(self, state: GraphState) -> GraphState:
        response = router_chain.invoke({"input": state["input"]})

        match = re.search(r"```json(.*?)```", response.content, re.DOTALL)
        router_json = json.loads(match.group(1).strip())

        return router_json

    def run_chain(self, state: GraphState) -> GraphState:
        dest = state["destination"]
        
        if dest  == "DEFAULT":
            chain = default_chain
            response = chain.invoke({"input": state["next_inputs"]})
            output = response.content
        else:
            chain = destination_chains[dest]
            response = chain.invoke({"input": state["next_inputs"]})
            output = response['answer']

        # invoke or run function
        return {"output": output}

agent = Agent()

def request(input: str) -> str:
    result = agent.graph.invoke( {"input": input} )
    final_output = result["output"]
    return final_output