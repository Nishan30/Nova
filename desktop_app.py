from datetime import datetime
import struct
import sys
import threading
from pynput import keyboard
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QWidget, QLineEdit, QTextEdit, QPushButton,
                               QTabWidget, QLabel, QFileDialog, QMessageBox,QHBoxLayout)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer
import qtawesome as qta
import os 
from app.rag_backend import ( clear_chat_history_from_db, extract_text_from_files, general_knowledge_chat, generate_welcome_greeting, get_recent_journal_summary_for_prompt, initialize_whisper, open_file_or_app, process_multimodal_file, get_qa_chain, get_journal_context, init_database,
                             load_user_profile, run_agent_with_tools, save_journal_entry, save_profile_setting, process_uploaded_files,
                             save_chat_message, load_chat_history, speak_text, transcribe_audio_with_whisper, unified_chat_and_journal_handler)

import speech_recognition as sr
import pvporcupine
import pyaudio 
from dotenv import load_dotenv
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
init_database()

# --- Worker for loading the chain at startup ---
class ChainLoaderWorker(QObject):
    finished = Signal(object)
    def run(self):
        print("ChainLoaderWorker: Starting to load main QA chain...")
        qa_chain = get_qa_chain()
        self.finished.emit(qa_chain)
        print("ChainLoaderWorker: Main QA chain loaded and emitted.")

class MultimodalWorker(QObject):
    finished = Signal(object) # This will emit a string
    def __init__(self, user_text, image_path):
        super().__init__()
        self.user_text = user_text
        self.image_path = image_path
    def run(self):
        # Call our new backend function
        response_text = process_multimodal_file(self.user_text, self.image_path)
        self.finished.emit({"result": response_text}) # Emit in the same dict format as the old

# --- Worker for handling each chat query ---
class BackendWorker(QObject):
    finished = Signal(object)

    def __init__(self, user_text, file_context, history, qa_chain,journal_context):
        super().__init__()
        self.user_text = user_text
        self.file_context = file_context
        self.history = history
        self.qa_chain = qa_chain
        self.journal_context = journal_context

    def run(self):
        """
        CORRECTED: Now correctly assembles the prompt with file context,
        conversation history, and emotional context from the user's journal.
        """
        print("BackendWorker: Running with pre-loaded chain and journal context.")
        user_profile = load_user_profile()
        user_name = user_profile.get("user_name", "there")

        # --- UPDATED, COMPREHENSIVE PERSONA PROMPT ---
        persona_prompt = (
            f"You are Nova, a friendly, empathetic, and supportive personal AI companion for {user_name}. "
            "Your persona is encouraging and you can use emojis. Your primary goal is to answer questions "
            "based on the documents and conversation history provided. Be mindful of the user's recent "
            "journal entries to understand their emotional state and adjust your tone accordingly."
            f"{self.journal_context}" # This is the injected journal context string
        )

        # --- ROBUST PROMPT ASSEMBLY ---
        final_prompt_parts = [persona_prompt]
        
        # Add conversational history
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        if self.history:
            final_prompt_parts.append(f"\n\n--- Recent Conversation ---\n{formatted_history}")
        
        # Add the context from the attached file, if it exists
        if self.file_context:
            final_prompt_parts.append(f"\n\n--- Attached File Context ---\n{self.file_context}")

        # Add the user's current question
        final_prompt_parts.append(f"\n\n--- My New Question ---\n{user_name}: {self.user_text}")
        final_prompt_parts.append("\n\nNova:")
        
        final_prompt = "\n".join(final_prompt_parts)
        
        # Invoke the RAG chain with the complete context
        response = self.qa_chain.invoke(final_prompt)
        self.finished.emit(response)

class GreetingWorker(QObject):
    finished = Signal(str)
    def run(self):
        """Calls the backend function to generate the welcome greeting."""
        print("GreetingWorker: Main QA chain loaded and emitted.")
        greeting_text = generate_welcome_greeting()
        self.finished.emit(greeting_text)
        print("GreetingWorker: Done")

# --- Hotkey Listener ---
class HotkeyListener(QObject):
    toggle_signal = Signal()
    def __init__(self):
        super().__init__()
        self.hotkey = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+<shift>+X'), self.on_activate)
        self.listener = keyboard.Listener(on_press=self.for_canonical(self.hotkey.press), on_release=self.for_canonical(self.hotkey.release))
    def on_activate(self): self.toggle_signal.emit()
    def for_canonical(self, f): return lambda k: f(self.listener.canonical(k))
    def start_listening(self): self.listener.start()

# --- Main Application Window ---
class SidePanelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nova Companion")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry.width() - 420, 30, 400, screen_geometry.height() - 60)
        self.setStyleSheet(self.get_stylesheet())
        self.is_whisper_ready = False
        self.is_gemma_ready = False

        self.setup_ui()

        self.main_qa_chain = None
        self.is_tts_enabled = False
        self.is_recording = False
        self.chat_history_list = []
        self.onboarding_step = None 
        self.staged_chat_file = None
        self.active_workers = []
        self.start_wake_word_listener()
        self.check_onboarding()
        load_dotenv()

    # In desktop_app.py, inside the SidePanelApp class

    def setup_ui(self):
        """Creates all the UI widgets and tabs with attachment features."""
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # --- Chat Tab ---
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_display = QTextEdit(); self.chat_display.setReadOnly(True)
        
        # Create a horizontal layout for the input area
        self.button_bar_layout = QHBoxLayout()
        self.input_box = QLineEdit(); self.input_box.setPlaceholderText("Type your message...")
        self.input_box.returnPressed.connect(self.handle_user_input)
        
        # NEW: Attach button for chat
        self.chat_attach_button = QPushButton(qta.icon('fa5s.paperclip', color='#D8DEE9'), "")
        self.chat_attach_button.clicked.connect(self.handle_chat_attachment)
        self.chat_attach_button.setToolTip("Attach a file to this message")

        self.tts_button = QPushButton(qta.icon('fa5s.volume-mute', color='#D8DEE9'), "")
        self.tts_button.setCheckable(True) # Make it a toggle button
        self.tts_button.clicked.connect(self.toggle_tts)
        self.tts_button.setToolTip("Toggle spoken responses")

        self.mic_button = QPushButton(qta.icon('fa5s.microphone', color='#D8DEE9'), "")
        self.mic_button.clicked.connect(self.handle_mic_button)
        self.mic_button.setToolTip("Click to speak")
        self.mic_button.setEnabled(False) 

        self.clear_chat_button = QPushButton(qta.icon('fa5s.trash-alt', color='#D8DEE9'), "")
        self.clear_chat_button.clicked.connect(self.handle_clear_chat)
        self.clear_chat_button.setToolTip("Clear all chat history")

        # --- ADD BUTTONS TO THE HORIZONTAL LAYOUT ---
        self.button_bar_layout.addWidget(self.chat_attach_button)
        self.button_bar_layout.addWidget(self.tts_button)
        self.button_bar_layout.addWidget(self.mic_button)
        self.button_bar_layout.addStretch() # Pushes the next button to the right
        self.button_bar_layout.addWidget(self.clear_chat_button)

        # NEW: Label to show what file is attached
        self.chat_attachment_label = QLabel("")
        self.chat_attachment_label.setStyleSheet("color: #81A1C1;") # Nord accent color
        self.chat_attachment_label.hide() # Hide it initially
        

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #D8DEE9; padding-left: 5px;")
        
        self.chat_layout.addWidget(self.chat_display)
        self.chat_layout.addWidget(self.chat_attachment_label) # Add the new label
        self.chat_layout.addWidget(self.input_box)
        self.chat_layout.addLayout(self.button_bar_layout) # Add the horizontal layout
        self.chat_layout.addWidget(self.status_label)
        self.status_label.hide()

        # --- Journal Tab ---
        self.journal_widget = QWidget()
        self.journal_layout = QVBoxLayout(self.journal_widget)
        self.journal_editor = QTextEdit(); self.journal_editor.setPlaceholderText("Write your thoughts...")
        
        # Create a horizontal layout for journal buttons
        self.journal_button_layout = QVBoxLayout()
        self.save_journal_button = QPushButton("Save Journal Entry")
        self.save_journal_button.setIcon(qta.icon('fa5s.save', color='#ECEFF4'))
        self.save_journal_button.clicked.connect(self.handle_save_journal)
        
        # NEW: Attach button for journal
        self.journal_attach_button = QPushButton("Attach File to Entry")
        self.journal_attach_button.setIcon(qta.icon('fa5s.paperclip', color='#ECEFF4'))
        self.journal_attach_button.clicked.connect(self.handle_journal_attachment)

        self.journal_button_layout.addWidget(self.save_journal_button)
        self.journal_button_layout.addWidget(self.journal_attach_button)
        
        self.journal_layout.addWidget(self.journal_editor)
        self.journal_layout.addLayout(self.journal_button_layout)

        # --- Memory Tab (No changes needed, but included for completeness) ---
        self.memory_widget = QWidget()
        self.memory_layout = QVBoxLayout(self.memory_widget)
        self.memory_layout.setAlignment(Qt.AlignTop)
        self.upload_label = QLabel("Add new knowledge to Nova's permanent memory.")
        self.upload_label.setStyleSheet("font-weight: bold; color: #ECEFF4;")
        self.upload_button = QPushButton("Upload PDFs or Images")
        self.upload_button.setIcon(qta.icon('fa5s.upload', color='#ECEFF4'))
        self.upload_button.clicked.connect(self.handle_file_upload)
        self.memory_layout.addWidget(self.upload_label)
        self.memory_layout.addWidget(self.upload_button)

        # Add tabs
        self.tabs.addTab(self.chat_widget, qta.icon('fa5s.comments', color='#ECEFF4'), "Chat")
        self.tabs.addTab(self.journal_widget, qta.icon('fa5s.book', color='#ECEFF4'), "Journal")
        self.tabs.addTab(self.memory_widget, qta.icon('fa5s.brain', color='#ECEFF4'), "Memory")

    def check_onboarding(self):
        """
        The main startup router. Checks for onboarding, then starts the app.
        """
        user_profile = load_user_profile()
        if not user_profile.get("user_name"):
            # Start onboarding if the user has no name
            self.onboarding_step = 0
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(2, False)
            self.add_message_to_chat("Nova", "Hello! I'm Nova, your personal companion. To get started, what should I call you?")
        else:
            # User is already onboarded, so start the normal app launch sequence.
            self.launch_main_app()

    def launch_main_app(self):
        """
        MODIFIED: Now starts loading BOTH Gemma and Whisper in parallel
        at startup.
        """
        self.load_and_display_history()
        
        # Update status to show we're loading everything
        self.status_label.setText("Initializing AI brains...")
        self.status_label.show()

        # --- Start loading the main RAG brain (Gemma) ---
        self.start_chain_loader()
        
        # --- Start loading the Whisper brain in parallel ---
        self.whisper_loader_thread = QThread()
        self.whisper_loader_worker = WhisperLoaderWorker()
        self.whisper_loader_worker.moveToThread(self.whisper_loader_thread)
        
        self.whisper_loader_thread.started.connect(self.whisper_loader_worker.run)
        self.whisper_loader_worker.finished.connect(self.on_whisper_loaded)
        
        # Cleanup
        self.whisper_loader_worker.finished.connect(self.whisper_loader_thread.quit)
        self.whisper_loader_worker.finished.connect(self.whisper_loader_worker.deleteLater)
        self.whisper_loader_thread.finished.connect(self.whisper_loader_thread.deleteLater)
        
        self.whisper_loader_thread.start()

    def on_greeting_finished(self, greeting_text):
        """
        This function runs when the GreetingWorker is done. It adds the new
        greeting to the UI, the live list, and saves it to the database.
        This is now the final step of the startup process.
        """
        print(f"Received greeting: {greeting_text}")
        if greeting_text: # Only add if the greeting is not empty
            self.add_message_to_chat("Nova", greeting_text)
            save_chat_message("Nova", greeting_text)
            self.chat_history_list.append({"role": "Nova", "content": greeting_text})
            self.status_label.setText("Ready to chat!")

    def handle_onboarding_input(self, text):
        if self.onboarding_step == 0:
            save_profile_setting("user_name", text)
            self.add_message_to_chat("Nova", f"Great to meet you, {text}! 😊 What's your date of birth?")
            self.onboarding_step = 1
        elif self.onboarding_step == 1:
            save_profile_setting("date_of_birth", text)
            self.add_message_to_chat("Nova", "Got it. And where is your hometown?")
            self.onboarding_step = 2
        elif self.onboarding_step == 2:
            save_profile_setting("hometown", text)
            self.add_message_to_chat("Nova", "Perfect. Lastly, is there anything else you'd like me to know about you?")
            self.onboarding_step = 3
        elif self.onboarding_step == 3:
            save_profile_setting("extra_info", text)
            self.add_message_to_chat("Nova", "Thank you for sharing! We're all set up.")
            self.onboarding_step = None
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabEnabled(2, True)
            self.start_chain_loader()

    def handle_user_input(self):
        user_text = self.input_box.text().strip()

        # --- Stage 0: Handle Confirmation for Pending Agent Actions (Optional but good to keep) ---
        if hasattr(self, 'pending_action') and self.pending_action:
            if user_text.lower() in ['yes', 'y', 'ok', 'okay']:
                action = self.pending_action
                self.add_message_to_chat("You", user_text)
                self.input_box.clear()
                if action.get("type") == "open_file":
                    self.add_message_to_chat("Nova", f"Okay, opening {action['path']}...")
                    result_msg = open_file_or_app(action['path'])
                    QTimer.singleShot(1000, lambda: self.add_message_to_chat("Nova", result_msg))
                self.pending_action = None
                return
            elif user_text.lower() in ['no', 'n', 'cancel']:
                self.add_message_to_chat("You", user_text)
                self.add_message_to_chat("Nova", "Okay, cancelling that action.")
                self.pending_action = None
                self.input_box.clear()
                return

        # --- Stage 1: Initial validation and onboarding check ---
        if not user_text and not self.staged_chat_file:
            return

        if self.onboarding_step is not None:
            self.add_message_to_chat("You", user_text)
            self.input_box.clear()
            self.handle_onboarding_input(user_text)
            return

        # --- Stage 2: Handle file attachments ---
        file_context = ""
        display_prompt = user_text
        is_image_attached = False
        if self.staged_chat_file:
            file_name = os.path.basename(self.staged_chat_file)
            if self.staged_chat_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                is_image_attached = True #<-- Set the flag
                display_prompt = f"(Regarding image: {file_name})\n{user_text}"
            else:  # PDF, TXT
                display_prompt = f"(Regarding file: {file_name})\n{user_text}"
                file_context = extract_text_from_files([self.staged_chat_file])

        # --- Stage 3: Update UI and save user's message ---
        self.add_message_to_chat("You", display_prompt)
        save_chat_message("You", display_prompt)
        self.chat_history_list.append({"role": "You", "content": display_prompt})
        self.input_box.clear()

        # --- Stage 4: Check if AI is ready ---
        if not self.main_qa_chain:
            self.add_message_to_chat("Nova", "Hold on, still warming up! Try again in a moment.")
            return

        self.add_message_to_chat("Nova", "Thinking...")

        # --- Stage 5: The Clean, Final Routing and Worker Instantiation Logic ---
        worker = None  # Initialize worker to None
        recent_history = self.chat_history_list[-6:]

        # Tier 1: Is a file attached? -> Must be RAG.
        if is_image_attached:
            print("Routing to MultimodalWorker.")
            worker = MultimodalWorker(user_text, self.staged_chat_file)
        # Tier 2: Is a non-image file attached? -> Must be RAG.
        elif file_context:
            print("Routing to BackendWorker (RAG).")
            journal_context = get_recent_journal_summary_for_prompt()
            worker = BackendWorker(user_text, file_context, recent_history, self.main_qa_chain, journal_context)
        else:
            # Tier 3: No files attached. Is it a command? -> Agent.
            command_keywords = ['find', 'open', 'search for', 'launch', 'remind me']
            if any(user_text.lower().startswith(kw) for kw in command_keywords):
                print("Routing to AgentWorker.")
                worker = AgentWorker(user_text, recent_history)
            else:
                # Tier 4: Default to unified Chat/Journal.
                print("Routing to ChatAndJournalWorker.")
                worker = ChatAndJournalWorker(user_text, recent_history)

        if self.staged_chat_file:
            self.staged_chat_file = None
            self.chat_attachment_label.hide()

        # --- Stage 6: Start the chosen worker using the robust pattern ---
        if worker:
            thread = QThread()
            worker.moveToThread(thread)
            worker_ref = (thread, worker)
            self.active_workers.append(worker_ref)
            print(f"Starting new {worker.__class__.__name__}. Active workers: {len(self.active_workers)}")

            thread.started.connect(worker.run)
            worker.finished.connect(self.on_backend_finished)
            worker.finished.connect(lambda: self.on_worker_cleanup(worker_ref))
            
            thread.start()
        else:
            self.add_message_to_chat("Nova", "I'm not sure how to handle that request.")

    def on_worker_cleanup(self, worker_ref):
        """Removes the worker and its thread from the active list."""
        thread, worker = worker_ref
        
        # Standard Qt cleanup
        thread.quit()
        
        # The deleteLater calls are good practice, but let's remove them
        # as the Python garbage collector will handle it once we remove the reference.
        # worker.deleteLater()
        # thread.deleteLater()

        if worker_ref in self.active_workers:
            self.active_workers.remove(worker_ref)
        print(f"Worker finished and cleaned up. Active workers: {len(self.active_workers)}")

    def start_chain_loader(self):
        self.status_label.setText("Initializing brain...")
        self.status_label.show()
        self.chain_loader_thread = QThread()
        self.chain_loader_worker = ChainLoaderWorker()
        self.chain_loader_worker.moveToThread(self.chain_loader_thread)
        self.chain_loader_thread.started.connect(self.chain_loader_worker.run)
        self.chain_loader_worker.finished.connect(self.on_chain_loaded)
        self.chain_loader_worker.finished.connect(self.chain_loader_thread.quit)
        self.chain_loader_worker.finished.connect(self.chain_loader_worker.deleteLater)
        self.chain_loader_thread.finished.connect(self.chain_loader_thread.deleteLater)
        self.chain_loader_thread.start()

    def on_chain_loaded(self, qa_chain):
        self.main_qa_chain = qa_chain
        self.status_label.setText("Ready to chat! Thinking of a greeting...")
        self.is_gemma_ready = True
        print("Main App: QA chain has been received. Now generating welcome greeting.")

        self.check_all_models_ready()

        # --- NEW: Trigger the greeting worker from here ---
        self.greeting_thread = QThread()
        self.greeting_worker = GreetingWorker()
        self.greeting_worker.moveToThread(self.greeting_thread)
        self.greeting_thread.started.connect(self.greeting_worker.run)
        self.greeting_worker.finished.connect(self.on_greeting_finished)
        
        # Standard thread cleanup
        self.greeting_worker.finished.connect(self.greeting_thread.quit)
        self.greeting_worker.finished.connect(self.greeting_worker.deleteLater)
        self.greeting_thread.finished.connect(self.greeting_thread.deleteLater)

        self.greeting_thread.start()
    
    def on_whisper_loaded(self):
        """This function is called when the WHISPER model is ready."""
        print("Main App: Whisper model has been loaded.")
        self.is_whisper_ready = True
        
        # Enable the microphone button now that it's safe to use
        self.mic_button.setEnabled(True)
        self.mic_button.setToolTip("Click to speak")
        
        # Check if both models are ready to update the status
        self.check_all_models_ready()

    # --- ADD THIS HELPER FUNCTION ---
    def check_all_models_ready(self):
        """Checks if all models are loaded and updates the UI."""
        if self.is_gemma_ready and self.is_whisper_ready:
            self.status_label.setText("Ready to chat!")
            # Hide the status label after a few seconds
            QTimer.singleShot(3000, lambda: self.status_label.hide())
        elif self.is_gemma_ready:
            self.status_label.setText("Gemma ready, waiting for Whisper...")
        elif self.is_whisper_ready:
            self.status_label.setText("Whisper ready, waiting for Gemma...")

    def on_backend_finished(self, response):
        """UPDATED: Now triggers TTS if it's enabled."""
        self.chat_display.undo()
        answer = response.get('result', 'Sorry, an error occurred.')
        source = response.get('source')
        if source == "tool_open_confirmation":
            self.add_message_to_chat("Nova", answer)
            # Store the path that's awaiting confirmation
            self.pending_action = {"type": "open_file", "path": response.get("path_to_open")}
        else:
            self.pending_action = None # Clear any pending action
            self.add_message_to_chat("Nova", answer)
            save_chat_message("Nova", answer)
            self.chat_history_list.append({"role": "Nova", "content": answer})

        # --- NEW: Trigger the TTS worker ---
        if self.is_tts_enabled and answer:
            self.tts_thread = QThread()
            self.tts_worker = TTSWorker(answer)
            self.tts_worker.moveToThread(self.tts_thread)
            self.tts_thread.started.connect(self.tts_worker.run)
            self.tts_worker.finished.connect(self.tts_thread.quit)
            self.tts_worker.finished.connect(self.tts_worker.deleteLater)
            self.tts_thread.finished.connect(self.tts_thread.deleteLater)
            self.tts_thread.start()

    def get_stylesheet(self):
        """Returns a polished QSS stylesheet for the app."""
        return """
            QMainWindow {
                background-color: #2E3440; /* Nord dark blue */
                border-radius: 10px;
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background: #3B4252;
                color: #D8DEE9;
                padding: 10px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4C566A;
                color: #ECEFF4;
            }
            QTextEdit {
                background-color: #3B4252;
                color: #ECEFF4;
                border: none;
                font-size: 14px;
                border-radius: 5px;
            }
            QLineEdit {
                background-color: #4C566A;
                color: #ECEFF4;
                border: 1px solid #434C5E;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #5E81AC; /* Nord blue */
                color: #ECEFF4;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            /* Style for the scrollbar */
            QScrollBar:vertical {
                border: none;
                background: #3B4252;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #4C566A;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """

    def add_message_to_chat(self, role, message):
        message = message.replace('<', '<').replace('>', '>')
        role_color = "#88C0D0" if role == "You" else "#A3BE8C"
        formatted_message = f'<p style="margin-bottom: 5px;"><b style="color: {role_color};">{role}:</b><br>{message.replace(chr(10), "<br>")}</p>'
        self.chat_display.append(formatted_message)

    # In desktop_app.py, inside the SidePanelApp class

    def load_and_display_history(self):
        """
        Loads chat history from the DB and displays it. This should only be
        called once at startup.
        """
        # This function is now correct and does not need changes from the last version.
        self.chat_display.clear()
        self.chat_history_list.clear()
        
        history = load_chat_history()
        for role, content in history:
            self.add_message_to_chat(role, content)
            self.chat_history_list.append({"role": role, "content": content})

    def handle_save_journal(self):
        journal_text = self.journal_editor.toPlainText().strip()
        if not journal_text: return
        save_journal_entry(journal_text)
        self.journal_editor.setPlaceholderText("Entry saved!")
        QTimer.singleShot(2000, lambda: self.journal_editor.clear())

    def handle_file_upload(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Files", "", "All Files (*);;PDF (*.pdf);;Images (*.png *.jpg)")
        if file_paths:
            QMessageBox.information(self, "Processing", f"Adding {len(file_paths)} file(s) to memory...")
            num_processed = process_uploaded_files(file_paths)
            QMessageBox.information(self, "Success", f"Successfully added {num_processed} new documents to memory.")

    def toggle_visibility(self):
        if self.isVisible(): self.hide()
        else: self.show()
    
    def toggle_tts(self, checked):
        """Handles the state of the TTS button."""
        self.is_tts_enabled = checked
        if checked:
            self.tts_button.setIcon(qta.icon('fa5s.volume-up', color='#A3BE8C')) # Green when on
        else:
            self.tts_button.setIcon(qta.icon('fa5s.volume-mute', color='#D8DEE9'))
    
    def handle_mic_button(self):
        """
        A simple toggle. First click starts recording, second click stops.
        This version is simpler and more robust.
        """
        if not self.is_recording:
            # --- START RECORDING ---
            self.is_recording = True
            self.mic_button.setIcon(qta.icon('fa5s.stop-circle', color='#BF616A'))
            self.input_box.setPlaceholderText("Recording... Click mic again or stop talking.")
            
            # We will use a simple worker that just listens and saves the file.
            self.stt_thread = QThread()
            # We will create a simple inline worker for this. Let's redefine the STTWorker.
            self.stt_worker = self.RecordingWorker() # Use an inner class for simplicity
            self.stt_worker.moveToThread(self.stt_thread)
            
            self.stt_thread.started.connect(self.stt_worker.run)
            self.stt_worker.finished.connect(self.on_audio_recorded)
            self.stt_worker.error.connect(self.on_speech_error)
            
            self.stt_thread.start()
        else:
            # --- STOP RECORDING (logic is handled by the worker finishing) ---
            # For this simple model, the user just stops talking.
            # Clicking the button again is a visual cue, but the recognizer stops automatically.
            self.is_recording = False
            self.mic_button.setIcon(qta.icon('fa5s.microphone', color='#D8DEE9'))
            self.input_box.setPlaceholderText("Audio recorded! Press Enter to send.")

    def on_transcription_finished(self, transcribed_text):
        """
        Called when the TranscriptionWorker is done.
        Places the text in the input box and sends it.
        """
        self.input_box.setDisabled(False) # Re-enable the input box
        self.input_box.setPlaceholderText("Type your message...")

        if transcribed_text.strip():
            print(f"Transcription received: '{transcribed_text}'")
            self.input_box.setText(transcribed_text)
            # Automatically trigger the send action
            self.handle_user_input()
        else:
            # Handle case where transcription returns empty text
            self.on_speech_error("Could not understand the audio.")

    def on_speech_error(self, error_message):
        """Called when recording fails."""
        self.is_recording = False # Ensure state is reset
        self.mic_button.setIcon(qta.icon('fa5s.microphone', color='#D8DEE9'))
        self.input_box.setPlaceholderText("Type or click the mic to talk...")
        self.status_label.setText(error_message)
        self.status_label.show()
        QTimer.singleShot(3000, lambda: self.status_label.hide())

    def start_wake_word_listener(self):
        # --- IMPORTANT: Paste your Picovoice AccessKey here ---
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        
        self.wake_word_thread = QThread()
        self.wake_word_worker = WakeWordListener(access_key)
        self.wake_word_worker.moveToThread(self.wake_word_thread)

        self.wake_word_worker.wake_word_detected.connect(self.on_wake_word_detected)
        self.wake_word_thread.started.connect(self.wake_word_worker.run)
        
        # Make sure the worker is stopped when the app closes
        QApplication.instance().aboutToQuit.connect(self.wake_word_worker.stop)

        self.wake_word_thread.start()

    def on_wake_word_detected(self):
        """This is the new entry point for voice commands."""
        if self.is_recording:
            return

        print("Wake word detected! Starting active listening...")
        if not self.isVisible(): self.show()

        self.is_recording = True
        self.mic_button.setIcon(qta.icon('fa5s.microphone-alt', color='#50FA7B'))
        self.input_box.setPlaceholderText("Listening...")

        # --- APPLYING THE ROBUST PATTERN ---
        thread = QThread()
        worker = self.RecordingWorker()
        worker.moveToThread(thread)

        worker_ref = (thread, worker)
        self.active_workers.append(worker_ref)
        print(f"Starting RecordingWorker. Active workers: {len(self.active_workers)}")

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_audio_recorded)
        worker.error.connect(self.on_speech_error)
        worker.finished.connect(lambda: self.on_worker_cleanup(worker_ref))

        thread.start()

    def on_audio_recorded(self, audio_data):
        """
        Called when RecordingWorker is done. Now starts TranscriptionWorker robustly.
        """
        self.is_recording = False
        self.mic_button.setIcon(qta.icon('fa5s.microphone', color='#D8DEE9'))
        self.input_box.setPlaceholderText("Transcribing audio...")
        self.input_box.setDisabled(True)

        print("Audio data received. Starting transcription...")

        # --- APPLYING THE ROBUST PATTERN ---
        thread = QThread()
        worker = TranscriptionWorker(audio_data)
        worker.moveToThread(thread)
        
        worker_ref = (thread, worker)
        self.active_workers.append(worker_ref)
        print(f"Starting TranscriptionWorker. Active workers: {len(self.active_workers)}")
        
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_transcription_finished)
        worker.error.connect(self.on_speech_error)
        worker.finished.connect(lambda: self.on_worker_cleanup(worker_ref))

        thread.start()

    # --- We need to define the simple RecordingWorker ---
    class RecordingWorker(QObject):
        # The signal will now emit a generic object (the audio data)
        finished = Signal(object) 
        error = Signal(str)
        
        def run(self):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    print("RecordingWorker: Listening...")
                    audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    print("RecordingWorker: Heard audio, emitting data...")
                    
                    # --- THIS IS THE KEY CHANGE ---
                    # Don't write to a file here. Emit the raw data directly.
                    self.finished.emit(audio_data)

                except sr.WaitTimeoutError:
                    self.error.emit("I didn't hear anything.")
                except Exception as e:
                    self.error.emit(f"Recording failed: {e}")

    def handle_chat_attachment(self):
        """Opens a file dialog and 'stages' a file for the next chat message."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Attach File", "", "All Files (*);;PDF (*.pdf);;Images (*.png *.jpg)")
        
        if file_path:
            # Store the path to be used when the user sends their message
            self.staged_chat_file = file_path
            file_name = os.path.basename(file_path)
            self.chat_attachment_label.setText(f"📎 Attached: {file_name}")
            self.chat_attachment_label.show()

    def handle_journal_attachment(self):
        """Opens a file dialog, extracts text, and appends it to the journal editor."""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Attach Files to Journal", "", "All Files (*);;PDF (*.pdf);;Images (*.png *.jpg)")
        
        if file_paths:
            self.journal_editor.append("\n\n--- Attached Content ---\n")
            # Use our new backend function to get the text
            extracted_text = extract_text_from_files(file_paths)
            self.journal_editor.append(extracted_text)
    
    def handle_clear_chat(self):
        """
        Asks the user for confirmation, then clears the chat display,
        the in-memory history list, and the database.
        """
        # Create a confirmation dialog
        confirm_box = QMessageBox(self)
        confirm_box.setWindowTitle("Confirm Clear")
        confirm_box.setText("Are you sure you want to permanently delete all chat history?")
        confirm_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_box.setDefaultButton(QMessageBox.No)
        confirm_box.setIcon(QMessageBox.Warning)
        
        # Set stylesheet for the dialog to match the app theme
        confirm_box.setStyleSheet(self.get_stylesheet())

        # Execute the dialog and check the result
        return_value = confirm_box.exec()

        if return_value == QMessageBox.Yes:
            print("User confirmed to clear chat history.")
            # 1. Clear the on-screen display
            self.chat_display.clear()

            # 2. Clear the in-memory list
            self.chat_history_list.clear()

            # 3. Clear the database
            if clear_chat_history_from_db():
                self.status_label.setText("Chat history cleared.")
            else:
                self.status_label.setText("Error clearing history from database.")
            
            self.status_label.show()
            QTimer.singleShot(3000, self.status_label.hide)

            # Optional: Add a fresh welcome message to the now-empty chat
            self.add_message_to_chat("Nova", "Let's start a new conversation! What's on your mind?")

class TTSWorker(QObject):
    finished = Signal()
    def __init__(self, text_to_speak):
        super().__init__()
        self.text_to_speak = text_to_speak
    def run(self):
        """Calls the backend function to speak the text."""
        speak_text(self.text_to_speak)
        self.finished.emit()

# In desktop_app.py

class JournalVisionWorker(QObject):
    finished = Signal(str)
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
    def run(self):
        # --- CHANGE: Call the new unified function ---
        description = process_multimodal_file("Describe this image in detail for a journal entry.", self.image_path)
        self.finished.emit(description)

class TranscriptionWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, audio_data):
        super().__init__()
        self.audio_data = audio_data

    def run(self):
        print("TranscriptionWorker: Received audio data, preparing to transcribe...")
        temp_audio_path = os.path.join(APP_ROOT, "temp_transcribe.wav")

        try:
            # --- THE DEFINITIVE FIX IS HERE ---
            print(f"TranscriptionWorker: Saving data to {temp_audio_path}")
            with open(temp_audio_path, "wb") as f:
                f.write(self.audio_data.get_wav_data())
                # This forces the OS to write the file to disk immediately.
                # It is a blocking call that ensures the file is ready.
                f.flush()
                os.fsync(f.fileno())

            # Now, the file is guaranteed to be fully written and accessible.
            universal_path = temp_audio_path.replace('\\', '/')
            print(f"TranscriptionWorker: File write confirmed. Starting transcription with universal path: {universal_path}")
            transcribed_text = transcribe_audio_with_whisper(universal_path)

            if transcribed_text is not None:
                self.finished.emit(transcribed_text)
            else:
                self.error.emit("Transcription failed.")

        except Exception as e:
            self.error.emit(f"Error in transcription worker: {e}")
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"TranscriptionWorker: Cleaned up {temp_audio_path}")
                except OSError as e:
                    print(f"TranscriptionWorker: Error cleaning up file: {e}")
        
        print("TranscriptionWorker: Finished.")


class WhisperLoaderWorker(QObject):
    finished = Signal()

    def run(self):
        """
        Calls the new dedicated backend function to pre-load the model.
        This is now clean and has no side effects.
        """
        print("WhisperLoaderWorker: Starting to pre-load Whisper model...")
        initialize_whisper()
        print("WhisperLoaderWorker: Pre-loading complete.")
        self.finished.emit()

class WakeWordListener(QObject):
    wake_word_detected = Signal()

    def __init__(self, access_key, parent=None):
        super().__init__(parent)
        self.access_key = access_key
        self.is_running = True

    def run(self):
        try:
            self.keywords = ['computer', 'hey nova']

            # 2. Get the root directory of your application
            #    (Assuming APP_ROOT is defined globally in desktop_app.py)
            #    If not, define it: APP_ROOT = os.path.dirname(os.path.abspath(__file__))
            custom_keyword_path = os.path.join(APP_ROOT, 'wakewords', 'Hey-Nova_en_windows_v3_0_0.ppn')

            # 3. Build the list of keyword paths
            keyword_paths = [pvporcupine.KEYWORD_PATHS['computer'], custom_keyword_path]


            porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=keyword_paths
            )

            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            print("Wake Word engine started... Listening for 'Computer'...")

            while self.is_running:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print(f"Wake word 'Computer' detected!")
                    self.wake_word_detected.emit()
                    # Optional: Add a small sleep to prevent immediate re-triggering
                    QThread.msleep(2000)

        except Exception as e:
            print(f"Error in Wake Word Listener: {e}")
        finally:
            if 'porcupine' in locals() and porcupine is not None:
                porcupine.delete()
            if 'audio_stream' in locals() and audio_stream is not None:
                audio_stream.close()
            if 'pa' in locals() and pa is not None:
                pa.terminate()

    def stop(self):
        self.is_running = False

class GeneralChatWorker(QObject):
    finished = Signal(object)

    def __init__(self, user_text, history):
        super().__init__()
        self.user_text = user_text
        self.history = history

    def run(self):
        # Call the new backend function
        response = general_knowledge_chat(self.user_text, self.history)
        self.finished.emit(response)

class AgentWorker(QObject):
    finished = Signal(object)
    def __init__(self, user_text, history):
        super().__init__()
        self.user_text = user_text
        self.history = history

    def run(self):
        # Call the new agent backend function
        response = run_agent_with_tools(self.user_text, self.history)
        self.finished.emit(response)
        
class ChatAndJournalWorker(QObject):
    finished = Signal(object)
    def __init__(self, user_text, history):
        super().__init__()
        self.user_text = user_text
        self.history = history

    def run(self):
        response = unified_chat_and_journal_handler(self.user_text, self.history)
        self.finished.emit(response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SidePanelApp()
    main_window.show()
    hotkey_listener_obj = HotkeyListener()
    listener_thread = threading.Thread(target=hotkey_listener_obj.start_listening, daemon=True)
    listener_thread.start()
    hotkey_listener_obj.toggle_signal.connect(main_window.toggle_visibility)
    sys.exit(app.exec())