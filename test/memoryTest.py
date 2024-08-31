# 功能:使用LangChain框架实现将消息历史添加到chain中
# 安装必要的软件包
# pip install langchain_community==0.2.15
# pip install langchain==0.2.14
# pip install langchain-openai==0.1.22

# 用于记录日志，便于调试和追踪程序运行状态
import logging
# 用于与LLM的聊天模型交互
from langchain_openai import ChatOpenAI
# 配置可配置字段
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import PromptTemplate
# 定义聊天提示模板，以及占位符替换
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 用于运行带有消息历史的可运行对象
from langchain_core.runnables.history import RunnableWithMessageHistory
# 用于处理和存储对话历史
from langchain_community.chat_message_histories import SQLChatMessageHistory



# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# prompt模版设置相关 根据自己的实际情况进行调整
PROMPT_TEMPLATE_TXT = "prompt_template.txt"

# LLM参数全局配置
# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-EhxvNWXkjzZJADfHA1Ac24Dd0f0b42B2B97f3725D3BcA378"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
# oneapi相关配置(通义千问为例) 根据自己的实际情况进行调整
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_CHAT_API_KEY = "sk-DoU00d1PaOMCFrSh68196328E08e443a8886E95761D7F4Bf"
ONEAPI_CHAT_MODEL = "qwen-max"

# 使用openai的model
openai_model = ChatOpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_CHAT_API_KEY,
    model=OPENAI_CHAT_MODEL,
)
# 使用qwen-plus的model
oneapi_model = ChatOpenAI(
    base_url=ONEAPI_API_BASE,
    api_key=ONEAPI_CHAT_API_KEY,
    model=ONEAPI_CHAT_MODEL
)


# 定义prompt模版
prompt_template = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT)
# 测试返回的prompt_template对象中提取template的内容
# logger.info(f"prompt_template的内容: {prompt_template.template}\n")
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", prompt_template.template),
        MessagesPlaceholder(variable_name="history"),

    ]
)


# 获取对话历史
# 根据用户ID和会话ID获取SQL数据库中的聊天历史
# 该函数返回一个SQLChatMessageHistory对象，用于存储特定用户和会话的历史记录
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", "sqlite:///memory.db")


# 获取prompt在chain中传递的prompt最终的内容
def getPrompt(prompt):
    logger.info(f"最后给到LLM的prompt的内容: {prompt}")
    return prompt


if __name__ == "__main__":
    # 定义Chain
    # 将提示模板和模型连接起来，形成一个可运行的链
    # chain = prompt | getPrompt | openai_model
    chain = prompt | oneapi_model

    #  处理带有消息历史Chain  将可运行的链与消息历史记录功能结合
    # RunnableWithMessageHistory允许在运行链时携带消息历史
    # 实例化的with_message_history是一个配置了消息历史的可运行对象，使用get_session_history来获取历史记录
    # ConfigurableFieldSpec定义了用户ID和会话ID的配置字段，使得这些字段在运行时可以被动态传递
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            ),
        ],
    )

    # 四次不同对话测试
    # 第一次请求，用户ID为123，会话ID为123。询问模型"你好，我是NanGe!"
    response1 = with_message_history.invoke(
        {"language": "中文", "query": "你好，我是NanGe!"},
        config={"configurable": {"user_id": "123", "conversation_id": "123"}},
    )
    logger.info(f"最后给到LLM的response1的内容: {response1}")

    # 第二次请求，用户ID依然为123，会话ID改为456。询问模型"whats my name?"，由于是新的会话ID，模型应该不记得之前的对话历史
    response2 = with_message_history.invoke(
        {"language": "中文", "query": "我叫什么?"},
        config={"configurable": {"user_id": "123", "conversation_id": "456"}},
    )
    logger.info(f"最后给到LLM的response2的内容: {response2}")

    # 第三次请求，用户ID改为456，会话ID保持为123。询问模型"whats my name?"，由于是新的用户ID，模型应该不记得之前的对话历史
    response3 = with_message_history.invoke(
        {"language": "中文", "query": "我叫什么?"},
        config={"configurable": {"user_id": "456", "conversation_id": "123"}},
    )
    logger.info(f"最后给到LLM的response3的内容: {response3}")

    # 第四次请求，与第一次请求保持相同的用户ID和会话ID，询问模型"whats my name?"。由于有历史记录，模型应该记住用户之前自称为"bob"
    response4 = with_message_history.invoke(
        {"language": "中文", "query": "我叫什么?"},
        config={"configurable": {"user_id": "123", "conversation_id": "123"}},
    )
    logger.info(f"最后给到LLM的response4的内容: {response4}")



