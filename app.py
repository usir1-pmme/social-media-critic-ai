import streamlit as st
import google.generativeai as genai
import os
import json
import time
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
CONFIG = {
    "MODEL_NAME": "gemini-1.5-pro",
    "MAX_FILE_SIZE_MB": 50,
    "UPLOAD_TIMEOUT_SECONDS": 120,
    "SUPPORTED_MIME_TYPES": {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'mp4': 'video/mp4',
        'mov': 'video/quicktime'
    },
    "SUPPORTED_PLATFORMS": {
        "Instagram Reel/TikTok": "Focus on vertical video format, trends, and hook within first 3 seconds.",
        "Instagram Post": "Focus on visual composition, caption strategy, and hashtag usage.",
        "LinkedIn": "Focus on professional tone, thought leadership, and business value.",
        "YouTube Thumbnail": "Focus on click-worthiness, text readability, and visual impact.",
        "Twitter/X": "Focus on conciseness, engagement bait, and thread potential."
    }
}

# --- SESSION STATE ---
def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

@st.cache_resource
def configure_gemini_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(CONFIG["MODEL_NAME"])

def get_api_key() -> Optional[str]:
    if st.session_state.api_key:
        return st.session_state.api_key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    if api_key and len(api_key.strip()) > 30:
        st.session_state.api_key = api_key
    return api_key

@contextmanager
def temporary_file(uploaded_file):
    if not uploaded_file:
        yield None
        return
    suffix = f".{uploaded_file.name.split('.')[-1].lower()}"
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        chunk_size = 8192
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        uploaded_file.seek(0)
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

def validate_file(uploaded_file) -> tuple[bool, str]:
    if not uploaded_file:
        return False, "No file uploaded"
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > CONFIG["MAX_FILE_SIZE_MB"]:
        return False, f"File too large: {file_size_mb:.1f} MB (max {CONFIG['MAX_FILE_SIZE_MB']} MB)"
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in CONFIG["SUPPORTED_MIME_TYPES"]:
        return False, f"Unsupported file type: {ext}"
    return True, ""

def upload_file_to_gemini(file_path: str, mime_type: str) -> Any:
        uploaded_file = genai.upload_file(file_path, mime_name=mime_type)
        start_time = time.time()
    with st.spinner('Processing file upload...'):
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > CONFIG["UPLOAD_TIMEOUT_SECONDS"]:
                raise TimeoutError(f"Upload timed out after {CONFIG['UPLOAD_TIMEOUT_SECONDS']} seconds")
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise ValueError("File upload failed in Gemini API")
    return uploaded_file

def generate_analysis_prompt(platform: str) -> str:
    platform_focus = CONFIG["SUPPORTED_PLATFORMS"].get(platform, "")
    return f"""
Analyze this content for {platform}.
{platform_focus}

Act as a ruthless marketing expert. No sugar-coating.
Evaluate:
1. Visual Quality & Technical Execution
2. Hook/First Impression
3. Value Proposition & Virality Potential

Respond ONLY with valid JSON:
{{"score": <number>, "title": "<string>", "critique": "<string>", "improvement": "<string>", "next_time": "<string>"}}
"""

def analyze_content(file_path: str, mime_type: str, platform: str, api_key: str) -> Dict[str, Any]:
    model = configure_gemini_model(api_key)
    gemini_file = upload_file_to_gemini(file_path, mime_type)
    prompt = generate_analysis_prompt(platform)
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = model.generate_content([gemini_file, prompt])
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    txt = response.text.strip()
    if "```json" in txt:
        txt = txt.split("```json")[1].split("```")[0]
    elif "```" in txt:
        txt = txt.split("```")[1].split("```")[0]
    
    json_start = txt.find('{')
    json_end = txt.rfind('}') + 1
    if json_start == -1 or json_end == 0:
        raise ValueError("No valid JSON found in AI response")
    
    json_str = txt[json_start:json_end]
    result = json.loads(json_str)
    required_fields = ["score", "title", "critique", "improvement", "next_time"]
    for field in required_fields:
        if field not in result:
            result[field] = f"Missing {field}"
    return result

def display_results(data: Dict[str, Any], file_path: str, mime_type: str):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        score = data.get('score', 0)
        st.markdown(f"# Score: {score}/100")
        st.progress(score / 100)
        if mime_type.startswith('video'):
            st.video(file_path)
        else:
            st.image(file_path, use_column_width=True)
    
    with col2:
        st.subheader(data.get('title', 'Analysis'))
        st.error(data.get('critique', 'No critique available'))
        st.success(f"💡 **Optimization:** {data.get('improvement', 'No suggestions')}")
        st.info(f"🧠 **Lesson:** {data.get('next_time', 'No lessons')}")

def main():
    init_session_state()
    st.set_page_config(page_title="Social Media Critic AI Pro", layout="wide")
    api_key = get_api_key()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        if not api_key:
            api_key_input = st.text_input("Google Gemini API Key", type="password", placeholder="AIzaSy...")
            if api_key_input and len(api_key_input.strip()) > 30:
                st.session_state.api_key = api_key_input
                st.rerun()
        else:
            st.success("✅ API key active")
            if st.button("🗑️ Clear API Key"):
                st.session_state.api_key = None
                st.rerun()
        
        platform = st.selectbox("Target Platform:", list(CONFIG["SUPPORTED_PLATFORMS"].keys()))
        if not api_key:
            st.warning("⚠️ Please provide API key")
    
    st.title("🤖 Social Media Critic AI Pro")
    st.caption(f"Powered by {CONFIG['MODEL_NAME']}")
    
    if not st.session_state.analysis_complete:
        st.markdown("""
### Welcome

Upload your social media content and receive brutally honest, AI-powered feedback.

**Supported Formats:** PNG, JPG, JPEG, MP4, MOV (max 50MB)

**How to Use:**
1. Enter API key in sidebar
2. Select target platform
3. Upload content file
4. Click **Analyze**
""")
    
    uploaded_file = st.file_uploader("Upload Content", type=list(CONFIG["SUPPORTED_MIME_TYPES"].keys()))
    
    if uploaded_file:
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 **File:** {uploaded_file.name} | **Size:** {file_size_mb:.1f} MB")
    
    if uploaded_file and api_key and st.button("🔍 Analyze Content", type="primary", use_container_width=True):
        try:
            with temporary_file(uploaded_file) as tmp_path:
                if not tmp_path:
                    st.error("❌ Failed to create temporary file")
                    return
                ext = uploaded_file.name.split('.')[-1].lower()
                mime_type = CONFIG["SUPPORTED_MIME_TYPES"][ext]
                
                with st.spinner('🤖 AI is analyzing your content...'):
                    data = analyze_content(tmp_path, mime_type, platform, api_key)
                display_results(data, tmp_path, mime_type)
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")
        
        except TimeoutError as e:
            st.error(f"⏱️ Timeout: {e}")
        except ValueError as e:
            st.error(f"❌ Analysis Failed: {e}")
        except Exception as e:
            st.error(f"💥 Unexpected Error: {str(e)}")
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        if st.button("🔄 Start New Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.rerun()

if __name__ == "__main__":
    main()
