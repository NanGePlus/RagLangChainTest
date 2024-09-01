import requests
import json
import logging


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 默认非流式输出 True or False
stream_flag = False


# 四次不同对话测试
# 第一次请求，用户ID为123，会话ID为123
data = {
    "messages": [{"role": "user", "content": "张三九的基本信息是什么"}],
    "stream": stream_flag,
    "userId":"123",
    "conversationId":"123"
}

# # 第二次请求，用户ID依然为123，会话ID改为456。模型不记得之前的对话历史
# data = {
#     "messages": [{"role": "user", "content": "张三九的配偶是谁以及其联系方式"}],
#     "stream": stream_flag,
#     "userId": "123",
#     "conversationId": "456"
# }

# # 第三次请求，用户ID改为为456，会话ID改为123。模型不记得之前的对话历史
# data = {
#     "messages": [{"role": "user", "content": "李四六的基本信息是什么"}],
#     "stream": stream_flag,
#     "userId": "456",
#     "conversationId": "123"
# }

# # 第四次请求，用户ID依然为123，会话ID改为123。模型记得之前的对话历史
# data = {
#     "messages": [{"role": "user", "content": "张三九有哪些生活方式和习惯"}],
#     "stream": stream_flag,
#     "userId": "123",
#     "conversationId": "123"
# }

# 接收流式输出
if stream_flag:
    try:
        with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
            for line in response.iter_lines():
                if line:
                    json_str = line.decode('utf-8').strip("data: ")
                    # 检查是否为空或不合法的字符串
                    if not json_str:
                        logger.info(f"收到空字符串，跳过...")
                        continue
                    # 确保字符串是有效的JSON格式
                    if json_str.startswith('{') and json_str.endswith('}'):
                        try:
                            data = json.loads(json_str)
                            if data['choices'][0]['finish_reason'] == "stop":
                                logger.info(f"接收JSON数据结束")
                            else:
                                logger.info(f"流式输出，响应内容是: {data['choices'][0]['delta']['content']}")
                        except json.JSONDecodeError as e:
                            logger.info(f"JSON解析错误: {e}")
                    else:
                        print(f"无效JSON格式: {json_str}")
    except Exception as e:
        print(f"Error occurred: {e}")

# 接收非流式输出处理
else:
    # 发送post请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # logger.info(f"接收到返回的响应原始内容: {response.json()}\n")
    content = response.json()['choices'][0]['message']['content']
    logger.info(f"非流式输出，响应内容是: {content}\n")