import os
import sys
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
from langchain_core.tools import create_retriever_tool, tool
from langgraph.checkpoint.memory import MemorySaver
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persist_dir = os.path.join(project_root, "chroma_db_official")
    db_hash_file = os.path.join(persist_dir, ".pdf_hash")
    
    print(f" Đang quét thư mục {data_dir}... ")
    print(f"  - ChromaDB persist dir: {persist_dir}")
    
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

    # Tool tra cứu giáo trình
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Lấy 3 đoạn liên quan nhất
    pdf_tool = create_retriever_tool(
    retriever,
    "tra_cuu_giao_trinh_bong_ban",
    "Sử dụng công cụ này để tra cứu toàn bộ thông tin trong Đề cương và Giáo trình, "
    "bao gồm: lịch trình chi tiết từng tuần, thang điểm, điều kiện thi, và kỹ thuật bóng bàn."
)

    # Tool tìm kiếm video hướng dẫn
    @tool
    def tim_video_huong_dan(query: str, technique: str = None, difficulty: str = None) -> str:
        """
        Tìm kiếm video hướng dẫn động tác bóng bàn.

        Args:
            query: Từ khóa tìm kiếm (ví dụ: "forehand", "backhand", "serve")
            technique: Động tác cụ thể (tùy chọn)
            difficulty: Độ khó (beginner/intermediate/advanced, tùy chọn)

        Returns:
            Thông tin video dưới dạng text + metadata để Android app parse
        """
        try:
            # Import video service
            import sys
            backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if backend_root not in sys.path:
                sys.path.insert(0, backend_root)
            from backend.services.video_service import get_video_service

            video_service = get_video_service()
            videos = video_service.search_videos(
                query=query,
                technique=technique,
                difficulty=difficulty,
                limit=3
            )

            if not videos:
                return f"Không tìm thấy video hướng dẫn cho '{query}'. Vui lòng thử từ khóa khác hoặc liên hệ giáo viên để thêm video."

            # Tạo response text cho người dùng
            result = f"Tìm thấy {len(videos)} video hướng dẫn phù hợp:\n\n"
            for i, video in enumerate(videos, 1):
                result += f"{i}. **{video.title}**\n"
                result += f"   - Động tác: {video.technique}\n"
                result += f"   - Độ khó: {video.difficulty}\n"
                result += f"   - Mô tả: {video.description}\n"
                result += f"   - Tags: {', '.join(video.tags)}\n"

                # Kiểm tra file video tồn tại - sử dụng đường dẫn tuyệt đối
                file_path = video.file_path
                if not os.path.isabs(file_path):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    file_path = os.path.join(project_root, file_path)
                
                if os.path.exists(file_path):
                    result += f"   ✅ Video sẵn sàng để xem\n\n"
                else:
                    result += f"   ⚠️  File video không tìm thấy\n\n"

                # Luôn thêm metadata VIDEO_ID cho Android app parse (bất kể file có tồn tại hay không)
                result += f"[VIDEO_ID:{video.id}]\n"

            return result

        except Exception as e:
            return f"Lỗi khi tìm kiếm video: {str(e)}"

    tools = [pdf_tool, tim_video_huong_dan]

    prompt = (
    "Bạn là trợ lý học tập môn Bóng bàn tại UET. "
    "MỌI câu hỏi của sinh viên về lịch học, nội dung tuần, thang điểm và kỹ thuật "
    "BẮT BUỘC phải được tra cứu từ công cụ 'tra_cuu_giao_trinh_bong_ban' trước khi trả lời. "
    "Trong đề cương học phần có bảng lịch trình chi tiết theo từng buổi/tuần, hãy tìm kỹ trong đó.\n\n"
    "Khi học sinh hỏi về cách thực hiện động tác hoặc muốn xem video hướng dẫn, "
    "hãy sử dụng công cụ 'tim_video_huong_dan' để tìm và giới thiệu video phù hợp. "
    "Luôn kiểm tra và thông báo nếu video có sẵn để xem.\n\n"
    "QUAN TRỌNG: Khi tool 'tim_video_huong_dan' trả về kết quả, "
    "hãy GIỮ NGUYÊN toàn bộ output từ tool, đặc biệt là phần [VIDEO_ID:...] ở cuối. "
    "Đây là metadata quan trọng để Android app có thể tìm kiếm và stream video."
)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
    agent = create_agent(llm, tools=tools, system_prompt=prompt, checkpointer=MemorySaver())

    return agent

