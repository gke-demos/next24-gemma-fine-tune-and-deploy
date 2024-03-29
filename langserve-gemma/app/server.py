from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceTextGenInference, HuggingFaceEndpoint
from langchain.callbacks import streaming_stdout
import logging
import os

gemma_server = "http://locahost:8080/generate"
dolly_server = "http://locahost:8080/generate"

# get the gemma server URL
if "GEMMA_SERVER" in os.environ:
    gemma_server = os.environ["GEMMA_SERVER"]

# get the dolly server URL
if "DOLLY_SERVER" in os.environ:
    dolly_server = os.environ["DOLLY_SERVER"]
    
logging.info("Using gemma server: " + gemma_server)
logging.info("Using dolly server: " + dolly_server)

"""
callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
llm = HuggingFaceTextGenInference(
    inference_server_url=model_server,
    max_new_tokens=512,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=callbacks,
    streaming=True
)
"""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

gemma = HuggingFaceEndpoint(
    endpoint_url=gemma_server,
    max_new_tokens=256,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

gemma_template = "{prompt}"

gemma_prompt = PromptTemplate(
    template=gemma_template, 
    input_variables= ["prompt"]
)

gemma_chain = gemma_prompt | gemma

dolly = HuggingFaceEndpoint(
    endpoint_url=dolly_server,
    max_new_tokens=256,
    top_k=5,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

dolly_template = "### Instruction\n{prompt}\n\n### Answer\n"


dolly_prompt = PromptTemplate(
    template=dolly_template, 
    input_variables= ["prompt"]
)

dolly_parser = StrOutputParser()

dolly_chain = dolly_prompt | dolly | dolly_parser

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# add path for gemma
add_routes(app, gemma_chain, path="/gemma")

# add path for dolly
add_routes(app, dolly_chain, path="/dolly")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081, timeout_keep_alive=120)
