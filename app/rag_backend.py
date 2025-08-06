# app/rag_backend.py

import os
import subprocess
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM as Ollama 
from langchain.chains import RetrievalQA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyttsx3
import re

from PIL import Image
import pytesseract
from langchain.docstore.document import Document 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Add these to your existing imports at the top of app/rag_backend.py
import sqlite3
import json
from datetime import datetime

import ollama
import whisper
import shlex
import torch 
# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
DB_DIR = os.path.join(PROJECT_ROOT, 'db')

def init_database():
    """
    CORRECTED: Initializes BOTH the journal and chat history tables with full schemas.
    """
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # --- FIXED: Full, correct schema for the journal table ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        content TEXT NOT NULL,
        emotion_json TEXT
    )
    """)
    
    # --- Chat History Table (this part was already correct) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        setting_name TEXT PRIMARY KEY,
        setting_value TEXT NOT NULL
    )
    """)
    
    conn.commit()
    conn.close()
    print("Database (Journal & Chat History) initialized successfully.")
# --- NEW: Function to save a chat message ---
def save_chat_message(role, content):
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)
    """, (timestamp, role, content))
    conn.commit()
    conn.close()

# --- NEW: Function to load all chat history ---
def load_chat_history():
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
    history = cursor.fetchall()
    conn.close()
    return history

def process_documents():
    """
    Loads documents from the 'docs' directory, splits them into chunks,
    creates embeddings, and stores them in a persistent Chroma vector database.
    """
    print("Starting to process documents...")
    all_docs = []
    
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())

    if not all_docs:
        print("No new documents to process.")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    print("Creating embeddings and storing in ChromaDB...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function, 
        persist_directory=DB_DIR
    )
    
    print(f"Successfully processed and stored {len(all_docs)} documents.")
    return len(all_docs)

def get_qa_chain():
    """
    Initializes and returns a RetrievalQA chain, pointing to your specific Ollama model.
    """
    print("Loading vector database and initializing QA chain...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embedding_function
    )

    # --- MODIFICATION FOR YOUR SETUP ---
    # We are using the exact model name you provided.
    llm = Ollama(model="hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL", temperature=0.7)
    # ------------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    print("QA chain is ready.")
    return qa_chain

def analyze_emotion(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary of sentiment scores (pos, neg, neu, compound).
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

# This function will save a new journal entry to the database
def save_journal_entry(content):
    """
    Analyzes the emotion of the content and saves the journal entry to the database.
    """
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Analyze the emotion of the content
    emotion_scores = analyze_emotion(content)
    
    # Convert the emotion dictionary to a JSON string for storage
    emotion_json = json.dumps(emotion_scores)
    
    # Insert the new entry into the database
    cursor.execute("""
    INSERT INTO journal_entries (timestamp, content, emotion_json)
    VALUES (?, ?, ?)
    """, (timestamp, content, emotion_json))
    
    conn.commit()
    conn.close()
    print(f"Saved new journal entry at {timestamp}")

def get_journal_context(query):
    """
    Checks if a query is about feelings or journals.
    If so, fetches recent journal entries to provide as context.
    """
    # Simple keyword check, can be made more sophisticated later
    emotional_keywords = ["feel", "feeling", "felt", "journal", "mood", "stressed", "happy", "sad"]
    if any(keyword in query.lower() for keyword in emotional_keywords):
        print("Emotional query detected. Fetching journal context.")
        db_path = os.path.join(PROJECT_ROOT, 'memory.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch the 10 most recent journal entries
        cursor.execute("SELECT timestamp, content, emotion_json FROM journal_entries ORDER BY timestamp DESC LIMIT 10")
        entries = cursor.fetchall()
        conn.close()
        
        if not entries:
            return None
            
        # Format the entries into a string for the LLM prompt
        context_string = "Here are some of my recent journal entries:\n"
        for entry in entries:
            timestamp, content, emotion_json = entry
            emotions = json.loads(emotion_json)
            # Pick the dominant emotion to show the LLM
            dominant_emotion = max(emotions, key=lambda k: emotions[k] if k != 'compound' else -1)
            context_string += f"- On {timestamp} (feeling {dominant_emotion}): \"{content[:150]}...\"\n"
        
        return context_string
    
    return None

def save_profile_setting(setting_name, setting_value):
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Use INSERT OR REPLACE to handle both new and existing settings
    cursor.execute("INSERT OR REPLACE INTO user_profile (setting_name, setting_value) VALUES (?, ?)", (setting_name, setting_value))
    conn.commit()
    conn.close()

def load_user_profile():
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT setting_name, setting_value FROM user_profile")
    profile_data = dict(cursor.fetchall()) # Fetch all as a dictionary
    conn.close()
    return profile_data

# --- NEW: Function for Processing Uploaded Files ---
def process_uploaded_files(file_paths):
    """Processes a list of PDF and image files, extracts text, and adds it to the vector store."""
    all_docs = []
    for file_path in file_paths:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
            print(f"Loaded PDF: {os.path.basename(file_path)}")
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            try:
                text = pytesseract.image_to_string(Image.open(file_path))
                if text.strip():
                    # Create a LangChain Document object from the OCR text
                    doc = Document(page_content=text, metadata={"source": os.path.basename(file_path)})
                    all_docs.append(doc)
                    print(f"Processed Image (OCR): {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Could not process image {file_path}: {e}")

    if not all_docs:
        return 0

    # Use the same chunking and embedding process as before
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    vectorstore.add_documents(documents=splits) # Use add_documents to append to existing store
    
    print(f"Successfully added {len(all_docs)} new documents to memory.")
    return len(all_docs)

def extract_text_from_files(file_paths):
    """
    Extracts text from a list of PDF and image files and returns it as a single string.
    Does NOT save to the vector store.
    """
    full_text = ""
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                # Extract text from all pages and join them
                page_texts = [page.page_content for page in loader.load()]
                full_text += f"--- Content from {file_name} ---\n" + "\n".join(page_texts) + "\n\n"
                print(f"Extracted text from PDF: {file_name}")
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                text = pytesseract.image_to_string(Image.open(file_path))
                if text.strip():
                    full_text += f"--- Content from {file_name} (image) ---\n" + text + "\n\n"
                    print(f"Extracted text from Image (OCR): {file_name}")
        except Exception as e:
            print(f"Could not extract text from {file_path}: {e}")
            full_text += f"--- Could not read content from {file_name} ---\n\n"
            
    return full_text

def process_multimodal_file(prompt_text, file_path):
    """
    Handles ALL multimodal requests (image and audio) by executing the
    ollama command in a clean WSL login shell and piping the prompt and
    file path into its standard input. This is the proven, reliable method.
    """
    print(f"Processing multimodal file via WSL login shell: {file_path}")
    try:
        # Convert the Windows path to a WSL path
        wsl_file_path = file_path.replace('\\', '/')
        if ':' in wsl_file_path:
            drive_letter = wsl_file_path[0].lower()
            wsl_file_path = f"/mnt/{drive_letter}{wsl_file_path[2:]}"
        
        print(f"Converted to WSL path: {wsl_file_path}")
        
        # The command to start the interactive session
        command = [
            "wsl",
            "-d", "Ubuntu",
            "bash",
            "-l",
            "-c",
            "/usr/local/bin/ollama run hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL"
        ]

        # The input string to be piped. The --image flag works for audio too.
        input_string = f"'{prompt_text}' --image '{wsl_file_path}'"

        print(f"Executing command: {' '.join(command)}")
        print(f"Piping to stdin: {input_string}")

        result = subprocess.run(
            command,
            input=input_string,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            encoding='utf-8'
        )
        
        response_text = result.stdout.strip()
        print("WSL login shell command with stdin pipe successful.")
        return response_text

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama in WSL login shell: {e}")
        print(f"Stderr from WSL: {e.stderr}")
        return f"Sorry, I encountered an error trying to process the file: {e.stderr}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Sorry, an unexpected error occurred."

whisper_model = None

def load_whisper_model():
    """Loads the Whisper model into memory, specifying the model root directory."""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model for the first time...")
        model_name = "base"

        # --- THIS IS THE FINAL, CRITICAL FIX ---
        # Get the user's home directory and construct the path to the cache.
        # This makes the code work on any machine without hardcoding the username.
        model_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        print(f"Explicitly loading Whisper model from: {model_root}")

        # Tell Whisper exactly where to find the 'base.pt' file.
        whisper_model = whisper.load_model(model_name, download_root=model_root)
        
        print(f"Whisper model '{model_name}' loaded successfully.")
    return whisper_model

def initialize_whisper():
    """
    A dedicated function to be called at startup to pre-load the model.
    Its only purpose is to warm up the model.
    """
    load_whisper_model()


def transcribe_audio_with_whisper(audio_path):
    """
    Transcribes the given audio file using the pre-loaded Whisper model.
    Returns the transcribed text.
    """
    try:
        print(f"Checking file at: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found at path: {audio_path}")

        model = whisper.load_model("base")  # or "small", "medium", etc.
        result = model.transcribe(audio_path, fp16=False)
        transcribed_text = result['text']
        print(f"Whisper transcription successful: '{transcribed_text}'")
        return transcribed_text
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""
    
def generate_welcome_greeting():
    """
    Fetches very recent memories (last 12 hours) and uses the LLM to generate
    a cheerful, context-aware welcome greeting every time the app starts.
    """
    print("Generating welcome greeting...")
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Fetch VERY Recent Memories (last 12 hours) ---
    cursor.execute("""
    SELECT content, emotion_json FROM journal_entries 
    WHERE timestamp >= datetime('now', '-12 hours') 
    ORDER BY timestamp DESC LIMIT 3
    """)
    recent_journals = cursor.fetchall()

    cursor.execute("""
    SELECT role, content FROM chat_history 
    WHERE timestamp >= datetime('now', '-12 hours') 
    ORDER BY timestamp DESC LIMIT 5
    """)
    recent_chats = cursor.fetchall()
    
    conn.close()

    # --- Determine Time-Aware Greeting ---
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        time_greeting = "Good morning"
    elif 12 <= current_hour < 18:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    user_profile = load_user_profile()
    user_name = user_profile.get("user_name", "there")

    if not recent_journals and not recent_chats:
        # If no recent memories, give a simple, warm welcome.
        return f"{time_greeting}, {user_name}! Great to see you. What's on your mind?"

    # --- Prepare the context for the LLM ---
    memory_context = "Here are some of my user's most recent activities to inspire a welcome greeting:\n"
    
    if recent_journals:
        memory_context += "\n--- Recent Journal Entries ---\n"
        for content, emotion_json in recent_journals:
            emotions = json.loads(emotion_json)
            dominant_emotion = max(emotions, key=lambda k: emotions[k] if k != 'compound' else -1)
            memory_context += f"- (User felt {dominant_emotion}): \"{content[:100]}...\"\n"

    if recent_chats:
        memory_context += "\n--- Recent Chat Topics ---\n"
        for role, content in reversed(recent_chats):
            if role.lower() == 'you':
                memory_context += f"- User mentioned: \"{content[:100]}...\"\n"

    # --- The new, improved "Welcome" prompt ---
    briefing_prompt = (
        f"You are Nova, a personal companion. Your user, {user_name}, has just opened the app. "
        f"It's currently the {time_greeting.split(' ')[1].lower()}. "
        "Based *only* on the very recent memories provided below, write a short (1-2 sentences), cheerful, and welcoming greeting. "
        "If you see a recent journal entry, gently acknowledge the feeling. If you see a recent chat, you can reference the topic. "
        "Your goal is to make the user feel seen and cheered up. "
        f"Start the message with a friendly greeting (like 'Hey {user_name}!' or '{time_greeting}!'). "
        "End with an encouraging, open-ended question. Use an emoji. ✨\n\n"
        f"--- Recent Memories ---\n{memory_context}\n\n"
        "Nova's Welcome Greeting:"
    )

    try:
        client = ollama.Client()
        response = client.chat(
            model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL',
            messages=[{'role': 'user', 'content': briefing_prompt}]
        )
        greeting = response['message']['content']
        print(f"Generated greeting: {greeting}")
        return greeting
    except Exception as e:
        print(f"Error generating greeting with LLM: {e}")
        return f"{time_greeting}, {user_name}! So glad to see you again. 😊"

def general_knowledge_chat(user_text, history):
    """
    Handles general conversation by talking directly to the model,
    bypassing the RAG chain.
    """
    print("Handling general knowledge chat...")
    client = ollama.Client()
    journal_context = get_recent_journal_summary_for_prompt()
    
    # We still provide the persona and history for context
    persona_prompt = (
        "You are Nova, a friendly, empathetic, and helpful personal AI companion. "
        "Answer the user's question directly and concisely."
        f"{journal_context}" # Inject the context here
    )
    
    messages = [{'role': 'system', 'content': persona_prompt}]
    
    # Add recent history to the messages
    for msg in history:
        # Ollama expects 'user' and 'assistant' roles
        role = 'user' if msg['role'].lower() == 'you' else 'assistant'
        messages.append({'role': role, 'content': msg['content']})
    
    # Add the new user question
    messages.append({'role': 'user', 'content': user_text})

    try:
        response = client.chat(
            model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL', # Your model
            messages=messages
        )
        return {"result": response['message']['content']}
    except Exception as e:
        print(f"Error in general knowledge chat: {e}")
        return {"result": "Sorry, I had trouble thinking about that."}
    
def get_recent_journal_summary_for_prompt():
    """
    Fetches the 2 most recent journal entries from the last 24 hours
    and formats them as a string for use in an LLM prompt.
    """
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    context_string = ""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Fetch entries from the last 24 hours, limit 2
            cursor.execute("""
            SELECT content, emotion_json FROM journal_entries 
            WHERE timestamp >= datetime('now', '-24 hours') 
            ORDER BY timestamp DESC LIMIT 2
            """)
            recent_journals = cursor.fetchall()

        if recent_journals:
            # Start the context block for the prompt
            context_string += "\n\n--- CONTEXT FROM USER'S RECENT JOURNAL ---\n"
            context_string += "Your user has been feeling a certain way recently. Be mindful of this context in your response.\n"
            for content, emotion_json in recent_journals:
                emotions = json.loads(emotion_json)
                # Find the dominant emotion (excluding 'compound')
                dominant_emotion = max((k for k in emotions if k != 'compound'), key=emotions.get, default='neutral')
                context_string += f"- User recently felt {dominant_emotion} and wrote: \"{content[:150]}...\"\n"
            context_string += "------------------------------------------\n"

    except Exception as e:
        print(f"Could not fetch journal context: {e}")
        # Return empty string on failure
        return ""
        
    return context_string
    
def speak_text(text):
    """
    Uses the system's native TTS engine to speak the given text aloud.
    This runs in a blocking way, so it should be called from a thread.
    """
    try:
        print("Initializing TTS engine...")
        engine = pyttsx3.init()
        
        # Optional: Adjust voice properties if you like
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[1].id) # Index 1 is often a female voice on Windows
        # engine.setProperty('rate', 190) # Speed of speech

        engine.say(text)
        engine.runAndWait()
        print("TTS finished.")
    except Exception as e:
        print(f"Error in TTS engine: {e}")

def search_files_on_desktop(query: str) -> str:
    """
    Searches for files on the user's Desktop containing the query in their name.
    Returns a formatted string of found file paths.
    'query' should be a simple search term, e.g., 'report' or 'q3_presentation'.
    """
    print(f"TOOL: Searching for files with query: {query}")
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    found_files = []
    try:
        for root, _, files in os.walk(desktop_path):
            for file in files:
                if query.lower() in file.lower():
                    found_files.append(os.path.join(root, file))
        
        if not found_files:
            return "No files found on the Desktop matching that query."
        
        # Return a nicely formatted list
        return "Found the following files:\n" + "\n".join(f"- {f}" for f in found_files[:10]) # Limit to 10 results
    except Exception as e:
        return f"An error occurred while searching for files: {e}"

def open_file_or_app(path: str) -> str:
    """
    Opens a file at the given absolute path, or opens a common application by name.
    For apps, just provide the name, e.g., 'spotify', 'notepad', 'chrome'.
    For files, provide the full path, e.g., 'C:/Users/User/Desktop/report.pdf'.
    """
    print(f"TOOL: Attempting to open: {path}")
    try:
        # Sanitize common app names
        app_name = path.lower().strip()
        if app_name in ['spotify', 'notepad', 'chrome', 'vscode']:
            os.startfile(app_name)
            return f"Successfully opened {app_name}."
        
        # Handle file paths
        # Convert forward slashes to backslashes for os.startfile
        path_to_open = os.path.normpath(path)
        if os.path.exists(path_to_open):
            os.startfile(path_to_open)
            return f"Successfully opened the file at {path}."
        else:
            return f"Error: The file path '{path}' does not exist."
    except Exception as e:
        return f"An error occurred while trying to open '{path}': {e}"
    
def run_agent_with_tools(user_text: str, history: list) -> dict:
    """
    An agent that can decide to search for files, open files/apps,
    or just chat.
    """
    print("AGENT: Deciding which tool to use...")
    
    # We provide the LLM with a list of "tools" it can use.
    tools_description = """
    You have access to the following tools:
    1. search_files(query: str): Use this to find files on the user's desktop.
    2. open_file_or_app(path: str): Use this to open a file with its full path or a common application by name (spotify, chrome, notepad).
    3. chat_response(response: str): Use this if no tool is needed and you just want to talk to the user.
    """
    
    # This is the agent's "thought process" prompt.
    agent_prompt = f"""
    {tools_description}

    Review the user's latest request and the conversation history. Decide which tool to use.
    Respond ONLY with the function call for the tool you choose. For example:
    - If the user says "find my presentation", respond with: search_files(query="presentation")
    - If the user says "open spotify", respond with: open_file_or_app(path="spotify")
    - If the user says "Hi, how are you?", respond with: chat_response(response="I'm doing well, thanks for asking!")

    Conversation History:
    {history[-3:]} 

    User's Latest Request: "{user_text}"

    Your Decision (one function call only):
    """

    # --- Step 1: Ask the LLM to choose a tool ---
    client = ollama.Client()
    response = client.chat(
        model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL',
        messages=[{'role': 'user', 'content': agent_prompt}],
        options={'temperature': 0.0} # We want deterministic output
    )
    decision = response['message']['content'].strip().replace("`", "") 
    print(f"AGENT: LLM decided to call: {decision}")

    # --- Step 2: Execute the chosen tool ---
    try:
        # Regex to match function_name(parameter="value")
        # It captures the function name, the parameter key, and the parameter value
        match = re.match(r'(\w+)\s*\(\s*\w+\s*=\s*["\'](.*?)["\']\s*\)', decision)

        if not match:
            # If the regex doesn't match, the format is wrong. Fall back to chat.
            print(f"AGENT: Decision format '{decision}' was invalid. Falling back to general chat.")
            return general_knowledge_chat(user_text, history)

        function_name = match.group(1)
        value = match.group(2)

        if function_name == "search_files":
            result = search_files_on_desktop(value)
            return {"result": result, "source": "tool_search_files"}

        # --- THIS IS THE KEY CHANGE ---
        elif function_name == "open_file_or_app":
            # Execute the action immediately instead of asking for confirmation.
            print(f"AGENT: Executing tool 'open_file_or_app' with path: {value}")
            result = open_file_or_app(value) 
            # The agent's final response IS the result of the tool.
            return {"result": result, "source": "tool_open_file"}

        elif function_name == "chat_response":
            return {"result": value, "source": "tool_chat"}

        else:
            print(f"AGENT: Unknown function '{function_name}'. Falling back to general chat.")
            return general_knowledge_chat(user_text, history)

    except Exception as e:
        print(f"AGENT: Error executing tool decision '{decision}': {e}")
        return {"result": "I had a little trouble with that request. Could you try rephrasing it?"}

def clear_chat_history_from_db():
    """
    Deletes all records from the chat_history table in the database.
    """
    db_path = os.path.join(PROJECT_ROOT, 'memory.db')
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
        print("Database: Chat history cleared successfully.")
        return True
    except Exception as e:
        print(f"Database: Error clearing chat history: {e}")
        return False

def unified_chat_and_journal_handler(user_text: str, history: list) -> dict:
    """
    Handles all non-tool, non-RAG conversations. It decides whether to
    pull in journal context based on the user's question.
    """
    print("UNIFIED HANDLER: Processing request.")
    
    # --- The Smart Part: Decide if we need journal context ---
    journal_context = ""
    journal_keywords = ['journal', 'feeling', 'felt', 'mood', 'reflect', 'stress', 'anxious', 'happy', 'sad', 'sarah']
    if any(kw in user_text.lower() for kw in journal_keywords):
        print("UNIFIED HANDLER: Journal keywords detected. Fetching journal context.")
        # Fetch a good number of recent entries for context
        db_path = os.path.join(PROJECT_ROOT, 'memory.db')
        journal_context += "\n\n--- CONTEXT FROM USER'S PRIVATE JOURNAL ---\n"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM journal_entries ORDER BY timestamp DESC LIMIT 20")
            entries = cursor.fetchall()
            for entry in entries:
                journal_context += f"- \"{entry[0]}\"\n"
        journal_context += "------------------------------------------\n"

    # Create the prompt
    final_prompt = (
        "You are Nova, an empathetic AI companion. Your persona is to be helpful and insightful. "
        "You are operating in a special mode where you HAVE BEEN GIVEN access to a user's private journal entries "
        "to help them reflect. Do not mention that you are a language model or that you don't have access to files. "
        "Your task is to answer the user's question based *only* on the provided context from their journal and our conversation history. "
        "Synthesize patterns and insights directly from the text provided to you.\n"
        f"{journal_context}\n\n"  # This will be empty if no keywords were found
        "--- CONVERSATION HISTORY ---\n"
        f"{history}\n\n"
        f"--- USER'S QUESTION ---\n{user_text}\n\n"
        "Nova's Insightful Response:"
    )

    try:
        client = ollama.Client()
        response = client.chat(
            model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL',
            messages=[{'role': 'user', 'content': final_prompt}]
        )
        return {"result": response['message']['content']}
    except Exception as e:
        print(f"Error in unified handler: {e}")
        return {"result": "I had trouble processing that. Please try again."}