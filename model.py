import fitz  
import re
def extract_full_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    k=0
    for page in doc:
      full_text += page.get_text() + "\n"
    print(full_text)
    #full_text =  page=re.sub(r"!\[IMAGE: page_\d+_img_\d+\]", "image:path ![IMAGE: page_\\d+_img_\\d+]", page.get_text())
    doc.close()
    return full_text
pdf_path = r"C:\Users\Aswin Raj S\OneDrive\Documents\RAG\ati12052021_9-14-35.pdf" 
text_content = extract_full_text(pdf_path)
#print(text_content) 
import chromadb
from chromadb.utils import embedding_functions
client = chromadb.Client()
collection = client.create_collection(name="document11_QA")

import tiktoken
tokenizer = tiktoken.get_encoding('cl100k_base')  

def token_len(prompt, dev_input={}):
    try:
        logger_list = {"message": ""}
        tokens = tokenizer.encode(prompt, disallowed_special=()) 
        return len(tokens)
    except Exception as e:
        error_info = "Exception while finding the token length: " + str(e)
        return 0
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100, length_function=token_len, separators=[".", "!", "?", ";", "\n"])
chunks = text_splitter.split_text(text_content)

for num,chunk in enumerate(chunks):
  collection.add(documents = chunk, ids = "ChunkId"+ str(num))
results = collection.query(
    query_texts=["who is A. Y. Jackson?"],
    n_results=1
)
context = {"A"+str(num+1): str(doc) for num, doc in enumerate(results["documents"][0])}
userquery = "who is A. Y. Jackson?"
question= "context: " +str([context])

question += "\n\nQuestion:" + userquery
print(question)
## Environment secret keys
# from google.colab import userdata
# sec_key=userdata.get("HF_TOKEN")
# print(sec_key)
HF_TOKEN="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"
from dotenv import load_dotenv
import os
load_dotenv()

# Retrieve the secret key
# sec_key = os.getenv("HF_TOKEN")
sec_key = HF_TOKEN

print(f"Your secret key is: {sec_key}")


from langchain_huggingface import HuggingFaceEndpoint
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
from langchain import PromptTemplate, LLMChain

template = """Question: {prompt}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain=LLMChain(llm=llm,prompt=prompt)
print(llm_chain.invoke(question)["text"])
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_id="gpt2"
model=AutoModelForCausalLM.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)
pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=100)
hf=HuggingFacePipeline(pipeline=pipe)
hf.invoke("who is A. Y. Jackson?")
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=-0.1,  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 100},
)
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
chain=prompt|gpu_llm
question="who is A. Y. Jackson?"
chain.invoke({"question":question})