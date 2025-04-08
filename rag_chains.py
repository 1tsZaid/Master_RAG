from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from retriever_setup import get_retriever
from gemini_setup import get_LLM


INDEX_NAME = "master-rag"


retriever = get_retriever(index_name=INDEX_NAME, bm25_params_path="bm25_params.json", alpha=0.8)
llm = get_LLM()

# ####################################################################
# Prompt Templates
# ####################################################################

medical_template = """You are an assistant for diagnosing and treating medical conditions
Answer the question based only on the following context:
{context}
You are allowed to rephrase the answer based on the context.
If the answer isn't contained here, say you don't know.
Question: {input}"""

prompt_infos = [
    {
        "name": "medical", 
        "description": "Good for answering questions about medical conditions", 
        "prompt_template": medical_template
    }
]

# ####################################################################
# Default Chain
# ####################################################################

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = default_prompt | llm

# ####################################################################
# Destination Chains
# ####################################################################

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    destination_chains[name] = chain  

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# ####################################################################
# Routing Chain
# ####################################################################

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising \
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"]
)

router_chain = router_prompt | llm