import utils as Utils
import os as OS
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class NewsChat:
    store = {}
    session_id = ''
    db = None
    retriever = None

    def __init__(self, article_id: str):
        # Try to read GEMINI_API_KEY using utils helper (env, .env, export-style)
        api_key = Utils.load_gemini_key()
        if not api_key:
            raise EnvironmentError("Please set GEMINI_API_KEY environment variable or add it to a .env file")

        genai.configure(api_key=api_key)

        self.session_id = article_id
        self.db = Chroma(persist_directory=Utils.DB_FOLDER, collection_name='collection_1')
        self.retriever = self.db.as_retriever()

        self.qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def ask(self, question: str) -> str:
        # Compute a query embedding with Gemini and use Chroma's vector search to avoid dimension mismatch
        try:
            q_res = genai.embed_content(model="models/text-embedding-004", content=question)
            q_emb = None
            if isinstance(q_res, dict):
                q_emb = q_res.get("embedding") or (q_res.get("data", [{}])[0].get("embedding"))
            else:
                q_emb = getattr(q_res, "embedding", None)
                if q_emb is None and getattr(q_res, "data", None):
                    q_emb = getattr(q_res, "data")[0].get("embedding") if isinstance(getattr(q_res, "data")[0], dict) else getattr(q_res, "data")[0].embedding

            if q_emb is None:
                raise RuntimeError("Query embedding not returned")

            docs = self.db.similarity_search_by_vector(q_emb, k=3)
        except Exception:
            # fallback to retriever if available
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(question)
            elif hasattr(self.retriever, '_get_relevant_documents'):
                try:
                    docs = self.retriever._get_relevant_documents(question, run_manager=None)
                except TypeError:
                    docs = self.retriever._get_relevant_documents(question)
            else:
                docs = []

        context = "\n\n".join([getattr(d, 'page_content', str(d)) for d in docs[:3]])
        prompt = self.qa_system_prompt.format(context=context)

        full_prompt = prompt + "\n\n" + question

        model = genai.GenerativeModel('models/gemini-2.5-flash')
        chat = model.start_chat()
        response = chat.send_message(full_prompt)

        # extract text result
        answer = getattr(response, 'text', None)
        if not answer and hasattr(response, 'candidates') and response.candidates:
            # fallback to candidate content
            cand = response.candidates[0].content
            # content parts may be in .parts
            try:
                answer = cand.parts[0].text
            except Exception:
                answer = str(cand)

        return answer or ""

