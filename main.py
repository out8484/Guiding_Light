import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for API token
if 'github_token' not in st.session_state:
    st.session_state.github_token = None

# Streamlit app
st.title("🤖 GitHub Models AI 챗봇")

# Function to validate GitHub Token
def validate_token(token):
    import time
    max_retries = 3
    retry_delay = 1
    
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token
    )
    
    for attempt in range(max_retries):
        try:
            # Make a minimal API call to check validity
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True, "유효한 토큰입니다"
        except Exception as e:
            error_msg = str(e)
            # Check for connection errors
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            return False, error_msg
            
    return False, "연결 오류가 지속됩니다. 인터넷 연결을 확인해주세요."

# Function to get response from GitHub Models
def get_ai_response(question, github_token, language, age_group, gender):
    import time
    max_retries = 3
    retry_delay = 1
    
    # GitHub Models endpoint
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token
    )
    
    system_prompt = f"""You are a helpful assistant. 
    Please answer in {language}. 
    The user is a {age_group} {gender}. 
    Tailor your answer to be appropriate for this demographic, using suitable tone, examples, and complexity."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=1.0,
                max_tokens=4096,
                top_p=1.0
            )
            return {"success": True, "content": response.choices[0].message.content.strip()}
        except Exception as e:
            error_msg = str(e)
            # Check for connection errors
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            
            is_auth_error = "401" in error_msg or "unauthorized" in error_msg.lower() or "credentials" in error_msg.lower()
            return {"success": False, "content": error_msg, "is_auth_error": is_auth_error}
            
    return {"success": False, "content": "Failed to connect after multiple attempts. Please check your internet connection.", "is_auth_error": False}

# Function to generate image using Pollinations.ai (Free alternative)
def get_ai_image(prompt, github_token):
    try:
        # Pollinations.ai doesn't require an API key
        # We just need to construct the URL with the prompt
        import urllib.parse
        
        encoded_prompt = urllib.parse.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        return {"success": True, "url": image_url}
    except Exception as e:
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# Educational Mode & Token Analysis Functions
# -----------------------------------------------------------------------------
import tiktoken

def count_tokens(text, model="gpt-4o"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def get_token_ids(text, model="gpt-4o"):
    """Returns the list of token IDs."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(text)

def calculate_cost(input_tokens, output_tokens):
    """Calculates estimated cost for GPT-4o-mini (as of late 2024)."""
    # Pricing: Input $0.15 / 1M tokens, Output $0.60 / 1M tokens
    input_price = 0.15
    output_price = 0.60
    
    cost = (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)
    return cost

def visualize_tokens(text, model="gpt-4o"):
    """Visualizes tokens with alternating colors."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    decoded_tokens = [encoding.decode([t]) for t in tokens]
    
    # Create HTML for visualization
    html_content = '<div style="line-height: 2.0; font-family: monospace;">'
    colors = ["#FFD700", "#ADD8E6", "#90EE90", "#FFB6C1", "#FFA500"] # Simple pastel colors
    
    for i, token_str in enumerate(decoded_tokens):
        # Escape HTML characters
        safe_token = token_str.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "↵")
        color = colors[i % len(colors)]
        html_content += f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin-right: 2px; color: black;" title="Token ID: {tokens[i]}">{safe_token}</span>'
    html_content += '</div>'
    
    return html_content

# -----------------------------------------------------------------------------

# Sidebar for User Settings
with st.sidebar:
    st.header("👤 사용자 설정")
    st.markdown("AI의 응답 방식을 설정하세요")
    
    language = st.selectbox(
        "언어 선택",
        options=["Korean (한국어)", "English (영어)"],
        index=0
    )
    
    age_group = st.selectbox(
        "연령대",
        options=["10대", "20대", "30대", "40대", "50대", "60대 이상"],
        index=1
    )
    
    gender = st.radio(
        "성별",
        options=["남성", "여성", "기타"],
        index=0
    )
    
    st.divider()
    generate_image = st.checkbox("🖼️ 답변과 함께 이미지 생성하기", value=False, help="체크하면 답변 내용에 어울리는 이미지를 함께 생성합니다.")
    
    st.divider()
    # Educational Mode Toggle
    st.header("🎓 AI 학습 모드")
    educ_mode = st.toggle("AI 동작 분석 모드 켜기", value=False, help="토큰 사용량, 비용, 임베딩/인코딩 과정을 시각적으로 보여줍니다.")
    
    st.divider()
    st.markdown("ℹ️ **참고:** 선택하신 설정에 맞춰 AI가 답변의 톤과 내용을 조절합니다.")

# Function to extract system prompt (refactored for reuse)
def get_system_prompt(language, age_group, gender):
    return f"""You are a helpful assistant. 
    Please answer in {language}. 
    The user is a {age_group} {gender}. 
    Tailor your answer to be appropriate for this demographic, using suitable tone, examples, and complexity."""

# Function to save token and date to .env file
def save_token_to_env(token):
    import datetime
    env_path = ".env"
    current_date = datetime.date.today().isoformat()
    
    # Read existing content
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Update or add GITHUB_TOKEN and GITHUB_TOKEN_DATE
    token_line = f"GITHUB_TOKEN={token}\n"
    date_line = f"GITHUB_TOKEN_DATE={current_date}\n"
    
    new_lines = []
    token_updated = False
    date_updated = False
    
    for line in lines:
        if line.startswith("GITHUB_TOKEN="):
            new_lines.append(token_line)
            token_updated = True
        elif line.startswith("GITHUB_TOKEN_DATE="):
            new_lines.append(date_line)
            date_updated = True
        else:
            new_lines.append(line)
            
    if not token_updated:
        new_lines.append(token_line)
    if not date_updated:
        new_lines.append(date_line)
        
    # Write back to file
    with open(env_path, "w") as f:
        f.writelines(new_lines)

# Function to check token expiration
def check_token_expiration():
    import datetime
    token_date_str = os.getenv("GITHUB_TOKEN_DATE")
    
    if not token_date_str:
        return None, None
        
    try:
        token_date = datetime.date.fromisoformat(token_date_str)
        expiration_date = token_date + datetime.timedelta(days=30)
        today = datetime.date.today()
        
        days_remaining = (expiration_date - today).days
        
        return days_remaining, expiration_date
    except ValueError:
        return None, None

# Main Logic
token_from_env = os.getenv("GITHUB_TOKEN")

# Determine if we have a valid token in session
if 'github_token' not in st.session_state:
    st.session_state.github_token = None

    # Check if we just failed auth (Runtime auto-logout)
    if st.session_state.get('auth_failure_reset', False):
        st.warning("⚠️ 인증 오류가 발생하여 재로그인이 필요합니다.")
        st.session_state.auth_failure_reset = False
    
    # Try to load from env ONLY if session is empty and we didn't just fail
    elif token_from_env and token_from_env != "your_github_token_here":
        # 1. First check if the token is expired by date
        days_left, exp_date = check_token_expiration()
        
        if days_left is not None and days_left <= 0:
            st.warning(f"⚠️ 저장된 토큰의 유효기간이 만료되었습니다 ({exp_date}). 단계를 진행하려면 새로운 토큰이 필요합니다.")
        
        # 2. If date is okay, validate with API
        else:
            with st.spinner("🔄 .env 파일의 토큰을 확인 중입니다..."):
                clean_env_token = token_from_env.strip()
                is_valid, msg = validate_token(clean_env_token)
                if is_valid:
                    st.session_state.github_token = clean_env_token
                else:
                    st.warning(f"⚠️ 저장된 토큰이 유효하지 않거나 연결할 수 없습니다. 수동으로 입력해주세요.\n오류: {msg}")

# 2. If still no valid token, show input
if not st.session_state.github_token:
    st.warning("⚠️ 유효한 GitHub Token이 없습니다. 아래에 토큰을 입력해주세요.")
    
    with st.expander("ℹ️ GitHub Token 발급 방법 보기"):
        st.markdown("""
        **GitHub Models용 토큰 발급 방법:**
        1. [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)로 이동하세요.
        2. "Generate new token (classic)"을 클릭하세요.
        3. 토큰 이름을 입력하고 필요한 권한(Scopes)을 선택하세요.
        4. 생성된 토큰을 복사하여 아래에 붙여넣으세요.
        """)
    
    with st.form(key="token_form"):
        token_input = st.text_input("GitHub Token 입력:", type="password", placeholder="github_pat_...")
        col1, col2 = st.columns([1, 1])
        with col1:
            token_submit = st.form_submit_button("확인 및 저장")
        with col2:
            reset_input = st.form_submit_button("입력 초기화", type="secondary")
            
    if reset_input:
        st.rerun()
        
    if token_submit and token_input:
        with st.spinner("🔐 토큰 확인 중..."):
            clean_token = token_input.strip()
            is_valid, msg = validate_token(clean_token)
            if is_valid:
                st.session_state.github_token = clean_token
                save_token_to_env(clean_token)
                st.success("✅ 토큰이 확인되고 .env 파일에 저장되었습니다!")
                st.rerun()
            else:
                st.error(f"❌ 유효하지 않은 토큰입니다: {msg}")

# 3. If we have a valid token, show the question form
if st.session_state.github_token:
    days_left, exp_date = check_token_expiration()
    if days_left is not None:
        if days_left <= 0:
            st.error(f"🚨 토큰 유효기간(30일)이 지났습니다! ({exp_date} 만료)")
            st.info("새로운 토큰을 발급받아 '토큰 재설정'을 해주세요.")
        elif days_left <= 5:
            st.warning(f"⚠️ 토큰 만료가 {days_left}일 남았습니다. ({exp_date} 만료 예정)")
            
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success("✅ GitHub Token이 활성화되었습니다")
    with col2:
        if st.button("🔄 토큰 재설정"):
            st.session_state.github_token = None
            st.rerun()

    # Chat Interface
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input("💬 질문을 입력하세요:", key="question_input", placeholder="궁금한 내용을 입력하고 Enter를 누르세요")
        submit_button = st.form_submit_button("🚀 전송")

    if submit_button and user_question:
        # Create tabs for Normal View vs Educational View if enabled
        if educ_mode:
            tab1, tab2, tab3 = st.tabs(["💬 채팅", "📊 토큰/비용 분석", "🔍 인코딩/임베딩 원리"])
        else:
            tab1 = st.container()

        # 1. Chat Tab (Visualizing Inference Streaming)
        with tab1:
            st.write("### 📝 답변:")
            message_placeholder = st.empty()
            full_response = ""
            
            # Streaming Logic
            client = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=st.session_state.github_token
            )
            
            lang_code = language.split("(")[0].strip()
            system_prompt = get_system_prompt(lang_code, age_group, gender)
            
            try:
                # Streaming call
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=1.0,
                    max_tokens=4096,
                    top_p=1.0,
                    stream=True  # ENABLE STREAMING
                )
                
                # Inference Simulation: Show tokens appearing one by one
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        # Update UI with "cursor" effect to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                
                # Final clean output
                message_placeholder.markdown(full_response)
                
                # Generate Image if requested
                if generate_image:
                    with st.spinner("🎨 이미지 생성 중..."):
                        image_prompt = f"Create an illustration representing this concept: {user_question}. Context: {full_response[:100]}"
                        image_result = get_ai_image(image_prompt, st.session_state.github_token)
                        if image_result["success"]:
                            st.image(image_result["url"], caption="AI가 생성한 이미지")
                        else:
                            st.warning(f"⚠️ 이미지 생성 실패: {image_result['error']}")

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    st.session_state.github_token = None
                    st.session_state.auth_failure_reset = True
                    st.rerun()

        # 2. Educational Tabs
        if educ_mode:
            # Prepare Data for Analysis
            input_text = system_prompt + "\n" + user_question
            input_tokens = count_tokens(input_text)
            output_tokens = count_tokens(full_response)
            total_tokens = input_tokens + output_tokens
            estimated_cost = calculate_cost(input_tokens, output_tokens)
            
            with tab2:
                st.header("📊 토큰 및 비용 분석")
                st.info("GitHub Models는 현재 무료지만, 실제 서비스(OpenAI API) 기준 예상 비용입니다.")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("입력 토큰", f"{input_tokens}개", help="질문 + 시스템 프롬프트 포함")
                col_b.metric("출력 토큰", f"{output_tokens}개", help="AI가 생성한 답변")
                col_c.metric("총 토큰", f"{total_tokens}개")
                
                st.metric("💰 예상 비용 (USD)", f"${estimated_cost:.6f}")
                
                st.subheader("💡 토큰이란?")
                st.markdown("""
                - **토큰(Token)**은 AI가 텍스트를 처리하는 기본 단위입니다.
                - 영어는 보통단어 하나가 1토큰, 한국어는 글자 하나가 1~3토큰 정도 됩니다.
                - 비용은 이 토큰 수에 비례하여 청구됩니다.
                """)

            with tab3:
                st.header("🔍 인코딩 & 임베딩 시각화")
                st.markdown("AI는 글자를 이해하지 못합니다. 대신 **숫자(Token ID)**로 변환하여 처리합니다.")
                
                st.subheader("1. 입력 질문의 토큰화 (Tokenization)")
                st.markdown(visualize_tokens(user_question), unsafe_allow_html=True)
                
                st.subheader("2. 정수 인코딩 (Encoding)")
                st.markdown("위의 각 조각들은 다음과 같은 **고유한 숫자 ID**로 컴퓨터에 전달됩니다.")
                token_ids = get_token_ids(user_question)
                st.code(str(token_ids), language="json")
                
                st.subheader("3. 임베딩 (Embedding) 개념")
                st.markdown("""
                > **임베딩이란?** 
                > 위에서 본 숫자(Token ID) 하나하나를 **수천 개의 실수(float)로 이루어진 벡터** 좌표로 변환하는 과정입니다. 
                > 이를 통해 AI는 단어의 **'의미'**를 수학적으로 이해하게 됩니다.
                """)
                st.image("https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.png", caption="임베딩 벡터 시각화 예시 (출처: OpenAI)", width=600)

    elif submit_button:
        st.warning("질문을 입력해주세요.")

