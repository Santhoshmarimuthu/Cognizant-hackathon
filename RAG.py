!pip install langchain langchain-community langchain-chroma sentence-transformers chromadb gradio datasets beautifulsoup4 PyPDF2 langchain-google-genai google-generativeai scikit-learn numpy -q

import os, warnings, socket
import gradio as gr
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")

API_KEY = "AIzaSyDFwFzgEId-kzeHhAG0-K2KESwbhqwmjdQ"
VECTORSTORE_DIR = "./vectorstore"

# --- Helpers ---
def find_free_port(start=7861, end=7900):
    for port in range(start, end + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except:
            continue
    return 7861

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# --- Query Validator ---
class QueryValidator:
    keywords = [
        'tumor','brain','cancer','glioblastoma','glioma','astrocytoma',
        'meningioma','pituitary','schwannoma','metastasis',
        'treatment','symptom','diagnosis','therapy',
        'surgery','radiation','chemo','prognosis','biopsy'
    ]
    block = ['hello','hi','bye','thanks','lol']

    @staticmethod
    def is_valid(query):
        q = query.lower()
        if any(b in q for b in QueryValidator.block):
            return False, "Non-medical query"
        if any(k in q for k in QueryValidator.keywords):
            return True, "Valid medical query"
        return False, "Not enough medical context"

# --- Medical Data Loader ---
def load_medical_data():
    try:
        ds = load_dataset("MedRAG/textbooks", split="train[:120]")
        col = "content" if "content" in ds.column_names else ds.column_names[0]
        return [x[col] for x in ds if len(x[col]) > 200]
    except:
        return ["Brain tumor treatment involves surgery, radiation, chemotherapy, targeted therapy, and immunotherapy..."]

# --- Retrieval System ---
class Retrieval:
    def __init__(self, data):
        self.emb = get_embeddings()
        self.validator = QueryValidator()
        self.vectorstore = self._init_vectorstore(data)

    def _init_vectorstore(self, data):
        if os.path.exists(os.path.join(VECTORSTORE_DIR, "chroma.sqlite3")):
            return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=self.emb)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
        docs = [chunk for d in data for chunk in splitter.split_text(d)]
        vs = Chroma.from_texts(docs, self.emb, persist_directory=VECTORSTORE_DIR)
        vs.persist()
        return vs

    def retrieve(self, query, k=4):
        ok, msg = self.validator.is_valid(query)
        if not ok:
            return None, msg
        return self.vectorstore.similarity_search(query, k=k), msg

# --- AI Processor ---
class AIProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=API_KEY)

    def answer(self, query, docs):
        ctx = "\n\n".join([d.page_content for d in docs]) if docs else ""
        prompt = (
            f"You are a medical expert. Use the context to answer.\n"
            f"Context:\n{ctx}\nQuestion: {query}\n"
            f"Answer in 3-5 detailed sentences. Add a medical disclaimer."
        )
        try:
            r = self.llm.invoke(prompt)
            return r.content if hasattr(r, "content") else str(r)
        except Exception as e:
            return f"Error: {e}"

# --- Main Medical System ---
class MedicalSystem:
    def __init__(self):
        data = load_medical_data()
        self.retrieval = Retrieval(data)
        self.ai = AIProcessor()

    def consult(self, query):
        result = self.retrieval.retrieve(query)
        if result[0] is None:
            return f"❌ {result[1]}"
        docs,_ = result
        return self.ai.answer(query, docs)

# --- Gradio UI ---
def launch_ui():
    sys = MedicalSystem()

    with gr.Blocks(css="""
        .gradio-container { max-width:900px; margin:auto; background-color:white; padding:20px; border-radius:10px; }
        .ask-btn { background-color:#1E90FF !important; color:white !important; }
    """) as demo:

        # Chatbot
        chatbot = gr.Chatbot(height=520)
        msg = gr.Textbox(placeholder="Example: What are the latest treatments for glioblastoma?")
        submit = gr.Button("Ask",variant="primary", elem_classes="ask-btn")
        clear = gr.Button("Clear Chat")

        def respond(msg,history):
            if not msg.strip(): return "⚠️ Please enter a valid question."
            return sys.consult(msg)

        def user_input(message,history): return "",history+[[message,None]]
        def bot(history):
            if history[-1][1] is None: history[-1][1]=respond(history[-1][0],history[:-1])
            return history

        msg.submit(user_input,[msg,chatbot],[msg,chatbot]).then(bot,chatbot,chatbot)
        submit.click(user_input,[msg,chatbot],[msg,chatbot]).then(bot,chatbot,chatbot)
        clear.click(lambda: None,None,chatbot)

    return demo

demo = launch_ui()
port = find_free_port()
demo.launch(share=True, server_port=port, server_name="0.0.0.0")
