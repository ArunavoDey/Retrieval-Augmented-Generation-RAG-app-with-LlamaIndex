from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import google.generativeai as palm
from llama_index.core import ServiceContext, StorageContext, load_index_from_storage
import os

from llama_index.llms.palm import PaLM

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("data").load_data()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCOvBiiFMdCbExqUBfAgfc5IJHRtBSmTRc"
palm.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = PaLM()
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=200)
#index = VectorStoreIndex.from_documents(documents, settings=Settings, show_progress=True)
#index.storage_context.persist()
storage_context = StorageContext.from_defaults(persist_dir = './storage')
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is Relative  Performance Prediction")
print(response)
display(Markdown(f"<b>{response}</b>"))

