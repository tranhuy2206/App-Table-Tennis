import threading
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QTextEdit, QPushButton
from langchain_core.messages import HumanMessage
from chatbot import build_chatbot

class ChatbotWorker(QThread):
    """Worker thread cho việc khởi tạo chatbot"""
    finished = Signal(object)  # agent object
    error = Signal(str)

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def run(self):
        try:
            agent = build_chatbot(self.data_dir)
            self.finished.emit(agent)
        except Exception as e:
            self.error.emit(str(e))

class MessageWorker(QThread):
    """Worker thread cho việc xử lý tin nhắn"""
    finished = Signal(str)  # response text
    error = Signal(str)

    def __init__(self, agent, message, config):
        super().__init__()
        self.agent = agent
        self.message = message
        self.config = config

    def run(self):
        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=self.message)]},
                config=self.config
            )
            final_msg = response['messages'][-1]

            if isinstance(final_msg.content, list):
                clean_text = "".join([item['text'] for item in final_msg.content if 'text' in item])
                self.finished.emit(clean_text)
            else:
                self.finished.emit(str(final_msg.content))
        except Exception as e:
            self.error.emit(str(e))

class ChatbotController(QObject):
    """Controller quản lý chatbot UI"""
    
    def __init__(self, parent_window, *, chat_display, chat_input, btn_send, data_dir="data/"):
        super().__init__()
        self.mw = parent_window
        self.chat_display = chat_display
        self.chat_input = chat_input
        self.btn_send = btn_send
        self.data_dir = data_dir

        # Khởi tạo chatbot
        self.agent = None
        self.config = {"configurable": {"thread_id": "session_1"}}

        # Setup worker threads
        self.init_worker = None
        self.message_worker = None
        self.is_processing = False
        self.is_ready = False

        # Connect button
        if self.btn_send:
            self.btn_send.clicked.connect(self._on_send_click)

        # Cho phép Ctrl+Enter để gửi
        if self.chat_input:
            self.chat_input.keyPressEvent = self._chat_input_key_press

        # Khởi tạo chat display
        if self.chat_display:
            welcome = (
                "Xin chào! Tôi là chatbot tư vấn về giáo trình bóng bàn.\n"
                "Hãy đặt câu hỏi về nội dung, lịch học, kỹ thuật, hoặc bất kỳ thông tin nào từ giáo trình.\n"
            )
            divider = "─" * 60 + "\n"
            welcome += divider + "⏳ Đang khởi tạo chatbot...\n"
            self.chat_display.setText(welcome)

        # Disable input khi chưa sẵn sàng
        if self.chat_input:
            self.chat_input.setEnabled(False)
        if self.btn_send:
            self.btn_send.setEnabled(False)

        # Khởi tạo chatbot trong background thread
        self._start_init_worker()

    def _start_init_worker(self):
        """Khởi tạo worker thread cho chatbot"""
        self.init_worker = ChatbotWorker(self.data_dir)
        self.init_worker.finished.connect(self._on_init_finished)
        self.init_worker.error.connect(self._on_init_error)
        self.init_worker.start()

    def _on_init_finished(self, agent):
        """Xử lý khi khởi tạo chatbot thành công"""
        self.agent = agent
        self.is_ready = True
        self._append_message("Hệ thống", "✅ Chatbot sẵn sàng! Bạn có thể bắt đầu hỏi câu hỏi.", "system")

        # Enable input
        if self.chat_input:
            self.chat_input.setEnabled(True)
        if self.btn_send:
            self.btn_send.setEnabled(True)
        if self.chat_input:
            self.chat_input.setFocus()

    def _on_init_error(self, error_msg):
        """Xử lý khi khởi tạo chatbot thất bại"""
        self.is_ready = False
        self._append_message("Lỗi", f"Không thể khởi tạo chatbot: {error_msg}", "error")
    
    def _chat_input_key_press(self, event):
        """Cho phép Ctrl+Enter để gửi tin nhắn"""
        from PySide6.QtGui import QKeySequence
        from PySide6.QtCore import Qt
        
        if event.key() == Qt.Key.Key_Return and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._on_send_click()
        else:
            QTextEdit.keyPressEvent(self.chat_input, event)
    
    def _on_send_click(self):
        """Xử lý khi nhấn nút Gửi"""
        if not self.is_ready or self.is_processing:
            return
        
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        
        # Disable input khi đang xử lý
        self.is_processing = True
        self.btn_send.setEnabled(False)
        self.chat_input.setEnabled(False)
        
        # Hiển thị tin nhắn của user
        self._append_message("Bạn", message, "user")
        
        # Xóa input
        self.chat_input.clear()
        
        # Tạo worker thread
        self.thread = threading.Thread(target=self._process_message_thread, args=(message,))
        self.thread.start()
    
    def _process_message_thread(self, message):
        """Xử lý tin nhắn trong thread riêng"""
        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=self.config
            )
            final_msg = response['messages'][-1]
            
            if isinstance(final_msg.content, list):
                clean_text = "".join([item['text'] for item in final_msg.content if 'text' in item])
                self._append_message("Chatbot", clean_text, "bot")
            else:
                self._append_message("Chatbot", str(final_msg.content), "bot")
        except Exception as e:
            self._append_message("Lỗi", f"Không thể xử lý: {str(e)}", "error")
        finally:
            # Re-enable input
            self.is_processing = False
            self.btn_send.setEnabled(True)
            self.chat_input.setEnabled(True)
            self.chat_input.setFocus()
    
    def _append_message(self, sender, message, msg_type="normal"):
        """Thêm tin nhắn vào chat display"""
        if not self.chat_display:
            return
        
        # Format tin nhắn dựa trên loại
        if msg_type == "user":
            formatted = f"\n👤 {sender}:\n{message}\n"
        elif msg_type == "bot":
            formatted = f"\n🤖 {sender}:\n{message}\n"
        elif msg_type == "error":
            formatted = f"\n⚠️ {sender}:\n{message}\n"
        elif msg_type == "system":
            formatted = f"\n📌 {sender}:\n{message}\n"
        else:
            formatted = f"\n{sender}:\n{message}\n"
        
        # Append text
        self.chat_display.append(formatted)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
