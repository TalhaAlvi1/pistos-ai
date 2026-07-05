"""
Web Interface Module for Pistos.ai using Streamlit
Full Desktop Design - Light Black Background
"""

import os
import html
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import streamlit as st

from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class WebInterface:
    """Streamlit web interface with minimalist design."""

    def __init__(self, pipeline: RAGPipeline, config: Dict[str, Any]):
        self.pipeline = pipeline
        self.config = config

        interface_config = config.get('web_interface', {})
        self.host = interface_config.get('host', '0.0.0.0')
        self.port = interface_config.get('port', 7860)
        self.max_history = interface_config.get('max_history', 20)
        
        # Admin password hash from environment variable
        self.admin_password_hash = os.getenv(
            'ADMIN_PASSWORD_HASH',
            '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9'  # Default: "admin123"
        )
        
        # Uploads directory
        self.uploads_dir = Path(config.get('document_processing', {}).get('uploads_dir', 'data/uploads'))
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WebInterface initialized on {self.host}:{self.port}")
    
    def _verify_admin_password(self, password: str) -> bool:
        """Verify admin password against stored hash."""
        if not password:
            return False
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == self.admin_password_hash
    
    def _list_uploaded_documents(self) -> List[Tuple[str, float]]:
        """List all documents in the uploads directory."""
        try:
            files = []
            supported_extensions = self.config.get('document_processing', {}).get(
                'supported_extensions', ['.pdf', '.txt', '.md', '.docx']
            )
            
            for file_path in self.uploads_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    files.append((file_path.name, size_mb))
            
            return files
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def _rebuild_vector_database(self, progress_placeholder, specific_file: str = None) -> str:
        """Rebuild the entire vector database from all documents in uploads directory.
        
        Args:
            progress_placeholder: Streamlit placeholder for progress messages
            specific_file: If provided, only process this specific file (for uploads)
        """
        try:
            logger.info("Starting vector database rebuild...")
            progress_placeholder.info("🔄 Initializing database rebuild...")
            
            # Reset the database
            self.pipeline.reset_database()
            logger.info("Database cleared")
            progress_placeholder.info("✓ Database cleared, scanning files...")
            
            # Get list of files to process
            supported_extensions = self.config.get('document_processing', {}).get(
                'supported_extensions', ['.pdf', '.txt', '.md', '.docx']
            )
            
            files_to_process = []
            
            if specific_file:
                # Only process the specific uploaded file
                file_path = self.uploads_dir / specific_file
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    files_to_process.append(file_path)
                    logger.info(f"Processing only uploaded file: {specific_file}")
            else:
                # Process all files in uploads directory
                for file_path in self.uploads_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        if not (file_path.name.startswith('~') or file_path.name.startswith('.')):
                            files_to_process.append(file_path)
            
            if not files_to_process:
                return "⚠ No documents found to process"
            
            total_files = len(files_to_process)
            progress_placeholder.info(f"📁 Found {total_files} file(s) to process...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_chunks = 0
            processed_files = []
            
            for idx, file_path in enumerate(files_to_process):
                # Update progress
                file_progress = (idx / total_files)
                progress_bar.progress(file_progress)
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                status_text.info(f"📄 [{idx+1}/{total_files}] Processing: {file_path.name} ({file_size_mb:.1f} MB)")
                
                try:
                    logger.info(f"Processing: {file_path.name}")
                    
                    # Show extraction progress for PDFs
                    if file_path.suffix.lower() == '.pdf':
                        status_text.info(f"📖 [{idx+1}/{total_files}] Extracting text from PDF: {file_path.name}")
                    
                    chunks = self.pipeline.process_single_file(str(file_path))
                    
                    # Show embedding progress
                    status_text.info(f"🔢 [{idx+1}/{total_files}] Creating {chunks} embeddings: {file_path.name}")
                    
                    total_chunks += chunks
                    processed_files.append(f"  ✓ {file_path.name}: {chunks} chunks")
                    logger.info(f"Processed {file_path.name}: {chunks} chunks")
                    
                    # Show completion for this file
                    status_text.success(f"✓ [{idx+1}/{total_files}] Completed: {file_path.name} ({chunks} chunks)")
                    time.sleep(0.5)  # Brief pause to show completion
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    processed_files.append(f"  ✗ {file_path.name}: ERROR - {str(e)}")
                    status_text.error(f"✗ [{idx+1}/{total_files}] Error: {file_path.name}")
                    time.sleep(1)
            
            progress_bar.progress(1.0)
            status_text.success("✓ Database rebuild complete!")
            
            result = f"✓ **Database Rebuilt Successfully**\n\n"
            result += f"**Total Chunks:** {total_chunks}\n"
            result += f"**Files Processed:** {len(processed_files)}/{total_files}\n\n"
            result += "**Details:**\n" + "\n".join(processed_files)
            
            logger.info(f"Database rebuild complete: {total_chunks} chunks from {len(processed_files)} files")
            return result
            
        except Exception as e:
            logger.error(f"Error rebuilding database: {e}")
            return f"❌ **Rebuild Failed**\n\n{str(e)}"
    
    def render_chat_interface(self):
        """Render the public chat interface."""
        st.markdown("""
        <style>
        /* Main container - pure black background */
        .stApp {
            background-color: #000000 !important;
        }
        
        /* Hide all Streamlit UI elements */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header {visibility: hidden !important;}
        .stDeployButton {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        
        /* Hide sidebar completely */
        [data-testid="stSidebar"] {display: none !important;}
        section[data-testid="stSidebar"] {display: none !important;}
        
        /* Title styling - bold white */
        h1 {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 32px !important;
            text-align: center !important;
            margin-bottom: 30px !important;
        }
        
        /* Chat container */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 15px 0 !important;
        }
        
        /* User messages - right aligned, bold gray */
        .stChatMessage[data-testid="user-message"] {
            display: flex !important;
            justify-content: flex-end !important;
        }
        
        .stChatMessage[data-testid="user-message"] .stMarkdown {
            color: #bdbdbd !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            text-align: right !important;
            max-width: 70% !important;
        }
        
        .stChatMessage[data-testid="user-message"] p {
            color: #bdbdbd !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }
        
        .stChatMessage[data-testid="user-message"] * {
            color: #bdbdbd !important;
            font-weight: 700 !important;
        }
        
        /* Override Streamlit's default for user messages */
        div[data-testid="user-message"] div[data-testid="stMarkdownContainer"] p {
            color: #bdbdbd !important;
            font-weight: 700 !important;
            font-size: 16px !important;
        }
        
        /* Assistant messages - left aligned, bold white */
        .stChatMessage[data-testid="assistant-message"] {
            display: flex !important;
            justify-content: flex-start !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] .stMarkdown {
            color: #ffffff !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            text-align: left !important;
            max-width: 80% !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] p {
            color: #ffffff !important;
            font-weight: 700 !important;
            margin: 0 !important;
            line-height: 1.7 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] strong {
            font-weight: 900 !important;
        }
        
        /* Force all text in assistant messages to be bold */
        .stChatMessage[data-testid="assistant-message"] * {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] .stMarkdown * {
            font-weight: 700 !important;
        }
        
        /* Override Streamlit's default chat message styles */
        div[data-testid="assistant-message"] div[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        
        /* Additional specificity for assistant text */
        .stChatMessage[data-testid="assistant-message"] [data-testid="stMarkdownContainer"] {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] [data-testid="stMarkdownContainer"] > div {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] [data-testid="stMarkdownContainer"] > div > p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 17px !important;
        }
        
        /* Chat input container - white box at bottom */
        .stChatInputContainer {
            background-color: #ffffff !important;
            border: 1px solid #333333 !important;
            border-radius: 0 !important;
            padding: 12px !important;
        }
        
        .stChatInputContainer textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: none !important;
            font-size: 15px !important;
        }
        
        .stChatInputContainer textarea::placeholder {
            color: #888888 !important;
        }
        
        /* Chat input button */
        .stChatInputContainer button {
            background-color: transparent !important;
            color: #888888 !important;
            border: none !important;
        }
        
        .stChatInputContainer button:hover {
            color: #000000 !important;
        }
        
        /* Spinner - white */
        .stSpinner > div {
            border-top-color: #ffffff !important;
        }
        
        /* Remove padding from main container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 800px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown("<h1>Pistos</h1>", unsafe_allow_html=True)
        
        # Welcome message below title
        st.markdown("""
        <div style='text-align: center; color: #ffffff; font-weight: 600; font-size: 15px; margin-bottom: 30px; padding: 0 20px;'>
        Hi, I am Pistos, the faithful messenger of God's Word. How can I assist you in your search for truth today?
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Display assistant messages with bold HTML
                    st.markdown(f"<div style='color: #ffffff; font-weight: 700; font-size: 17px; line-height: 1.7;'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    # Display user messages with bold HTML
                    st.markdown(f"<div style='color: #bdbdbd; font-weight: 700; font-size: 16px;'>{message['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Type a message..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                # Display user message with bold HTML
                st.markdown(f"<div style='color: #bdbdbd; font-weight: 700; font-size: 16px;'>{prompt}</div>", unsafe_allow_html=True)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Check if it's a greeting
                        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
                        if prompt.strip().lower() in greetings:
                            response = "Hi, I am Pistos, the faithful messenger of God's Word. How can I assist you in your search for truth today?"
                        else:
                            response = self.pipeline.query(prompt, use_streaming=False)
                        
                        # Wrap response in bold HTML
                        st.markdown(f"<div style='color: #ffffff; font-weight: 700; font-size: 17px; line-height: 1.7;'>{response}</div>", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def render_admin_interface(self):
        """Render the admin interface."""
        st.markdown("""
        <style>
        /* Main container - dark background */
        .stApp {
            background-color: #1a1a1a !important;
        }
        
        /* Hide all Streamlit UI elements */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header {visibility: hidden !important;}
        .stDeployButton {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        
        /* Hide sidebar completely */
        [data-testid="stSidebar"] {display: none !important;}
        section[data-testid="stSidebar"] {display: none !important;}
        
        /* All text bold white */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Title */
        h1 {
            font-size: 32px !important;
            text-align: center !important;
            margin-bottom: 30px !important;
        }
        
        /* Section headers */
        h2 {
            font-size: 24px !important;
            margin-top: 30px !important;
            margin-bottom: 15px !important;
            color: #ffffff !important;
        }
        
        /* Force emoji visibility */
        h2::before {
            filter: brightness(1.5) !important;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #ffffff !important;
        }
        
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Input fields */
        .stTextInput input {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
            font-weight: 600 !important;
        }
        
        .stTextInput label {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Buttons */
        .stButton button {
            font-weight: 700 !important;
            border-radius: 4px !important;
            color: #ffffff !important;
            background-color: #3a3a3a !important;
            border: 1px solid #555555 !important;
        }
        
        .stButton button:hover {
            background-color: #4a4a4a !important;
            color: #ffffff !important;
        }
        
        .stButton button[kind="primary"] {
            background-color: #d32f2f !important;
            color: #ffffff !important;
            border: none !important;
        }
        
        .stButton button[kind="primary"]:hover {
            background-color: #b71c1c !important;
            color: #ffffff !important;
        }
        
        /* Force all button text to be white */
        button {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        button p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        button div {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: #2a2a2a !important;
            border: 1px solid #444444 !important;
            border-radius: 4px !important;
        }
        
        .stFileUploader label {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        .stFileUploader section {
            background-color: #2a2a2a !important;
        }
        
        .stFileUploader section button {
            background-color: #3a3a3a !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            font-weight: 700 !important;
        }
        
        .stFileUploader section button:hover {
            background-color: #4a4a4a !important;
            color: #ffffff !important;
        }
        
        .stFileUploader section div {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        .stFileUploader section span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* File uploader text */
        [data-testid="stFileUploader"] {
            background-color: #2a2a2a !important;
        }
        
        [data-testid="stFileUploader"] label {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stFileUploader"] section {
            background-color: #2a2a2a !important;
            border: 1px solid #555555 !important;
        }
        
        [data-testid="stFileUploader"] button {
            background-color: #3a3a3a !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stFileUploader"] small {
            color: #cccccc !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stFileUploader"] span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stFileUploader"] div {
            color: #ffffff !important;
        }
        
        /* Success/Error/Info messages */
        .stSuccess, .stError, .stInfo, .stWarning {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        .stSuccess p, .stError p, .stInfo p, .stWarning p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #4CAF50 !important;
        }
        
        /* Container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 700px !important;
        }
        
        /* Divider */
        hr {
            border-color: #444444 !important;
            margin: 30px 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<h1>🔐 Admin Panel</h1>", unsafe_allow_html=True)
        
        # Authentication
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            st.markdown("### Authentication")
            password = st.text_input("Password", type="password", key="admin_password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Login", use_container_width=True, type="primary"):
                    if self._verify_admin_password(password):
                        st.session_state.authenticated = True
                        st.success("✓ Authentication successful")
                        st.rerun()
                    else:
                        st.error("✗ Invalid password")
        else:
            # Admin controls
            st.markdown("---")
            st.markdown("## 📤 Upload Documents")
            
            uploaded_file = st.file_uploader(
                "Select Document (PDF, DOCX, TXT, MD)",
                type=['pdf', 'docx', 'txt', 'md'],
                key="file_upload"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                upload_btn = st.button("Upload & Rebuild Database", type="primary", use_container_width=True)
            with col2:
                if st.button("Logout", use_container_width=True):
                    st.session_state.authenticated = False
                    st.rerun()
            
            if upload_btn and uploaded_file:
                try:
                    # Save uploaded file
                    destination = self.uploads_dir / uploaded_file.name
                    with open(destination, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✓ File uploaded: {uploaded_file.name}")
                    
                    # Rebuild database with only this file
                    st.info("Starting database rebuild...")
                    progress_placeholder = st.empty()
                    result = self._rebuild_vector_database(progress_placeholder, specific_file=uploaded_file.name)
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
            
            # Current documents
            st.markdown("---")
            st.markdown("## 📁 Current Documents")
            
            if st.button("Refresh List"):
                st.rerun()
            
            files = self._list_uploaded_documents()
            if files:
                st.markdown("**Uploaded Documents:**")
                for filename, size_mb in files:
                    st.markdown(f"• **{filename}** ({size_mb:.2f} MB)")
            else:
                st.info("No documents uploaded yet.")
            
            # Manual rebuild
            st.markdown("---")
            st.markdown("## 🔄 Manual Rebuild")
            st.markdown("Rebuild vector database from all documents in uploads folder")
            
            if st.button("Rebuild Database", use_container_width=True):
                progress_placeholder = st.empty()
                result = self._rebuild_vector_database(progress_placeholder)
                st.markdown(result)
    
    def launch(self, share: bool = False, **kwargs) -> None:
        """Launch the web interface."""
        logger.info(f"Launching Streamlit interface on {self.host}:{self.port}")
        
        # Check if we're in admin mode via URL parameter only
        query_params = st.query_params
        
        # Admin mode is ONLY accessible via ?admin=true in URL
        if "admin" in query_params:
            self.render_admin_interface()
        else:
            self.render_chat_interface()
