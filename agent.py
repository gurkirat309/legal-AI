import utils as Utils
import os as OS
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class NewsChat:
    store = {}
    session_id = ''
    db = None
    retriever = None

    def __init__(self, article_id: str):
        # Try to read GEMINI_API_KEY using utils helper
        api_key = Utils.load_gemini_key()
        if not api_key:
            raise EnvironmentError("Please set GEMINI_API_KEY environment variable or add it to a .env file")

        # Set environment variable for LangChain components
        OS.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

        self.session_id = article_id
        
        # Use available embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        self.db = Chroma(
            persist_directory=Utils.DB_FOLDER, 
            collection_name='legal_collection',
            embedding_function=embeddings
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        self.qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

    def ask(self, question: str) -> str:
        try:
            # 1. Retrieve relevant documents (using modern invoke method)
            docs = self.retriever.invoke(question)
            
            # 2. Build context string
            context = "\n\n".join([d.page_content for d in docs])
            
            # 3. Create prompt
            prompt = self.qa_system_prompt.format(context=context)
            
            # 4. Use Gemini for generation
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            response = llm.invoke(f"{prompt}\n\nQuestion: {question}")
            
            return response.content
            
        except Exception as e:
            print(f"Error in NewsChat.ask: {e}")
            return f"I encountered an error while searching for the answer: {str(e)}"

