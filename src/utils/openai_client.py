import openai
from time import sleep
from time import sleep
from functools import wraps


def configure_openai():
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    return openai.OpenAI()

def get_stream_response(messages):
    stream = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        stream=True
    )
    return stream

@st.cache_resource
def get_client():
    return configure_openai()


def handle_api_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except openai.APIError as e:
            st.error(f"API Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    return wrapper


def rate_limit(calls_per_minute: int):
    def decorator(func):
        last_called = []
        def wrapper(*args, **kwargs):
            now = datetime.now()
            last_called.append(now)
            
            minute_ago = now - timedelta(minutes=1)
            while last_called and last_called[0] < minute_ago:
                last_called.pop(0)
                
            if len(last_called) > calls_per_minute:
                sleep(60 / calls_per_minute)
                
            return func(*args, **kwargs)
        return wrapper
    return decorator