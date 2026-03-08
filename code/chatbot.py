import os
import time
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
import shutil

# 1. Cấu hình API Key 
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def _get_pdf_list_hash(data_dir):
    """Tạo hash của danh sách PDF để phát hiện file mới"""
    pdf_files = []
    if os.path.exists(data_dir):
        for file in sorted(os.listdir(data_dir)):
            if file.endswith(".pdf"):
                file_path = os.path.join(data_dir, file)
                # Lấy kích thước file để phát hiện thay đổi
                size = os.path.getsize(file_path)
                pdf_files.append(f"{file}:{size}")
    
    files_str = ",".join(pdf_files)
    return hashlib.md5(files_str.encode()).hexdigest()

def _should_rebuild_db(data_dir, persist_dir, db_hash_file):
    """Kiểm tra xem database có cần rebuild không"""
    # Nếu thư mục persist chưa tồn tại, cần build
    if not os.path.exists(persist_dir):
        return True
    
    # Nếu file hash không tồn tại, cần rebuild
    if not os.path.exists(db_hash_file):
        return True
    
    # So sánh hash của danh sách PDF hiện tại với hash cũ
    current_hash = _get_pdf_list_hash(data_dir)
    try:
        with open(db_hash_file, 'r') as f:
            saved_hash = f.read().strip()
        return current_hash != saved_hash
    except:
        return True

def _save_pdf_hash(db_hash_file, data_dir):
    """Lưu hash của danh sách PDF"""
    current_hash = _get_pdf_list_hash(data_dir)
    with open(db_hash_file, 'w') as f:
        f.write(current_hash)

def build_chatbot(data_dir):
    # Đọc file PDF (Giáo trình/Đề cương)
    all_documents = []
    
    # Quét toàn bộ thư mục data
    persist_dir = "./chroma_db_official"
    db_hash_file = "./chroma_db_official/.pdf_hash"
    
    print(f" Đang quét thư mục {data_dir}... ")
    
    # Kiểm tra xem có file PDF mới không
    rebuild_needed = _should_rebuild_db(data_dir, persist_dir, db_hash_file)
    
    if rebuild_needed and os.path.exists(persist_dir):
        print("⚠️  Phát hiện file PDF mới! Đang rebuild database...")
        shutil.rmtree(persist_dir)
    
    pdf_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            pdf_files.append(file)
            print(f"Đang nạp: {file}")
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
    
    if not pdf_files:
        print("⚠️  Không tìm thấy file PDF nào trong thư mục data!")

    # Chia nhỏ văn bản (Chunking) - Giảm chunk size để ít documents hơn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, 
                                                   chunk_overlap=80, 
                                                   separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                                                   )
    texts = text_splitter.split_documents(all_documents)
    print(f"📊 Tổng {len(texts)} đoạn cần xử lý")

    # Tạo Vector Database (Nhúng kiến thức vào ChromaDB)
    print(" Đang số hóa kiến thức... ")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if os.path.exists(persist_dir):
        print("Đã tìm thấy Database và danh sách PDF không thay đổi")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function= embeddings)
    else:
        print("Tạo Database mới...")
        print("⏰ Quá trình này sẽ mất khá lâu do giới hạn API. Vui lòng chờ...")

        batch_size = 5  # Giảm từ 20 xuống 5
        delay_between_batches = 45  # Tăng từ 10 lên 45 giây

        vectorstore = Chroma.from_documents(
            documents=texts[:batch_size], 
            embedding=embeddings, 
            persist_directory=persist_dir
        )

        for i in range(batch_size, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"📥 Cụm {batch_num}... (Đã xong {i}/{len(texts)} đoạn)")
            
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⚠️  API Quota đã hết! Đợi 60 giây trước khi thử lại...")
                    time.sleep(60)
                    vectorstore.add_documents(batch)
                else:
                    raise
            
            # Đợi để tránh vượt quá quota
            print(f"⏳ Đợi {delay_between_batches}s để API hồi phục...")
            time.sleep(delay_between_batches) 
            
        print("✅ Hoàn tất số hóa toàn bộ tài liệu và lưu xuống ổ cứng!")
    
    # Lưu hash của danh sách PDF hiện tại
    os.makedirs(persist_dir, exist_ok=True)
    _save_pdf_hash(db_hash_file, data_dir)

    # Khởi tạo mô hình Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # Tạo chuỗi truy vấn 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Lấy 3 đoạn liên quan nhất
    tool = create_retriever_tool(
    retriever,
    "tra_cuu_giao_trinh_bong_ban",
    "Sử dụng công cụ này để tra cứu toàn bộ thông tin trong Đề cương và Giáo trình, "
    "bao gồm: lịch trình chi tiết từng tuần, thang điểm, điều kiện thi, và kỹ thuật bóng bàn."
)
    
    tools = [tool]

    prompt = (
    "Bạn là trợ lý học tập môn Bóng bàn tại UET. "
    "MỌI câu hỏi của sinh viên về lịch học, nội dung tuần, thang điểm và kỹ thuật "
    "BẮT BUỘC phải được tra cứu từ công cụ 'tra_cuu_giao_trinh_bong_ban' trước khi trả lời. "
    "Trong đề cương học phần có bảng lịch trình chi tiết theo từng buổi/tuần, hãy tìm kỹ trong đó."
)
    agent = create_agent(llm, tools=tools, system_prompt=prompt, checkpointer=MemorySaver())

    return agent

