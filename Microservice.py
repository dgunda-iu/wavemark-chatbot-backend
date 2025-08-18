import sys

try:
    import psqlite3
    sys.modules["sqlite3"] = psqlite3
except ImportError:
    pass


from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
import os
import base64
import certifi
import time
from bs4 import BeautifulSoup
from langchain_openai import AzureChatOpenAI
from PIL import Image
from io import BytesIO
from IPython.display import Image as IPImage, display
#from unstructured.staging.base import elements_from_base64_gzipped_json
#from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
#from unstructured.partition.pdf import partition_pdf
import glob
from tqdm import tqdm
import pickle
import uuid
from langchain_chroma import Chroma
#from langchain.vectorstores import Chroma
#from langchain.vectorstores.azuresearch import AzureSearch
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
# from langchain.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os






# Import your graph, config, etc. from your notebook/module
# Example: from wavemark_graph import graph, config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wavemark-chatbot-frontend-ui-app-c9cdewg3cjg6hthf.eastus2-01.azurewebsites.net/"],  # Or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load the chunks and summaries
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("text_summaries.pkl", "rb") as f:
    text_summaries = pickle.load(f)
with open("table_summaries.pkl", "rb") as f:
    table_summaries = pickle.load(f)
with open("image_summaries.pkl", "rb") as f:
    image_summaries = pickle.load(f)


# Filter chunks based on their type
text_chunks = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
table_chunks = [chunk for chunk in chunks if "Table" in str(type(chunk))]
images_b64 = [e.metadata.image_base64 for chunk in chunks if chunk.metadata.orig_elements for e in chunk.metadata.orig_elements if 'Image' in str(type(e))]#'type' in e.to_dict() and e.to_dict()['type'] == "Image"]



# Define the API endpoint for chat
load_dotenv()
load_dotenv(dotenv_path=".env", encoding="utf-8")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")



#Define the vector store and document store

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_deployment="text-embedding-3-large",
    dimensions=1024,
    azure_endpoint="https://wavemarktextembeddingmodel.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15"    
)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    search_kwargs={"k": 5},  # Adjust the filter as needed
) 



#Load the vector store with embeddings

doc_ids= [str(uuid.uuid4()) for _ in text_chunks]
summary_text = [
    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
 
retriever.vectorstore.add_documents(summary_text)
retriever.docstore.mset(list(zip(doc_ids, text_chunks)))

#Add tables
table_ids = [str(uuid.uuid4()) for _ in table_chunks]
summary_tables = [
    Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, table_chunks)))


img_ids = [str(uuid.uuid4()) for _ in images_b64]
summary_img = [
    Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
]
retriever.vectorstore.add_documents(summary_img)
retriever.docstore.mset(list(zip(img_ids, images_b64)))




#RAG pipline

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode

model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2",              # matches your deployment name
    api_version="2025-01-01-preview",              # must match your Azure deployment API version
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY, # Pass the API key here
    temperature=0.5                                 # or your preferred settings
)


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | model             #ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | model     # ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)



#Graph configuration


# Tool to retrieve information
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


#LLM configuration
# Initialize the LLM with Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2",
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,  # Pass the API key here
    temperature=0.5
)



# Define the graph with two steps: query_or_respond and generate


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(tools=[retrieve], tool_choice="any" )          # List of tool functions or tool objectst
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "Try to give the user a helpful answer based on the retrieved context. "
        "Try to provide detailed steps or instructions if applicable."
        "You are an conversatinal assistant for question-answering tasks. "
        "Make sure to start with a short summary of the answer, and then provide detailed steps or instructions if applicable. "
        "Provide teh page numbers if applicable"
        "At the end try to ask follow up questions to the user to keep the conversation more helpful"
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know, please refer to the documentation or ask a support team member through Cherwell service management for help. "
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}




#Graph build
# 1. Create a new graph builder instance
graph_builder = StateGraph(MessagesState)


# 2. Add your nodes
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)


# 3. Set entry point and edges again
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


# 4. Compile the new graph
graph = graph_builder.compile()


#Define the memory saver for checkpointing
# This will save the state of the graph at each step

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}


#route to handle chat messages
class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """Endpoint to handle chat messages."""
    # Initialize the state with the user message
    message = request.message
    state = MessagesState(messages=[{"role": "user", "content": message}])
    
    #Without streaming
    # Run the graph and get the final message
    for step in graph.stream(
    {"messages": [{"role": "user", "content": message}]},
    stream_mode="values",
    config=config,  # Pass the configuration to the graph
):
        pass    
    final_msg = step["messages"][-1]
    return {"response": final_msg.content}

    # Run the graph and stream the response
    # def event_stream():
    #     for step in graph.stream(state, config=config, stream_mode="values"):
    #         msg = step["messages"][-1]
    #         # Only stream AI messages
    #         if getattr(msg, "type", None) == "ai":
    #             yield msg.content

    # return StreamingResponse(event_stream(), media_type="text/plain")
 