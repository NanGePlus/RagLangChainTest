# 0、相关视频链接
(1)【大模型应用开发基础】集成开发环境搭建Anaconda+PyCharm                       
https://www.bilibili.com/video/BV1q9HxeEEtT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                  
https://youtu.be/myVgyitFzrA                                                
(2)【流程实操】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享，一份代码搞定多类型大模型集成                                   
https://www.bilibili.com/video/BV1ryHxesEHs/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                          
https://youtu.be/xAEi5CWEIl0                                    
(3)【源码解析】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享，一份代码搞定多类型大模型集成                                                 
https://www.bilibili.com/video/BV1JUHxeQEc7/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                       
https://youtu.be/ptrCo8b-wM0                 
(4)【检索进阶Re-ranker】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享                        
https://www.bilibili.com/video/BV19tsteREYx/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                             
https://youtu.be/M_e-XT3x5OA                                    
(5)【进阶-PDF表格预处理】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享                                         
https://www.bilibili.com/video/BV1c7sueSEZe/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                                      
https://youtu.be/Nhal2EswD9o             
(6)【进阶-加载历史对话记忆(1)】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享            
https://www.bilibili.com/video/BV1GdnfeAEEe/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                      
https://youtu.be/wWx986JVXpI   
(7)【进阶-加载历史对话记忆(2)-集成到原项目中】RAG+LangChain+FastAPI+OpenAI+通义千问打造私有领域知识库，构建和检索全流程源码分享                             
https://www.bilibili.com/video/BV1bQHPegE3o/?vd_source=30acb5331e4f5739ebbad50f7cc6b949           
https://youtu.be/PPpfONOj5jI                     
            

# 1、基础概念
## 1.1 RAG定义及技术方案架构
### （1）RAG定义
RAG:Retrieval Augmented Generation(检索增强生成):通过使用检索的方法来增强生成模型的能力       
核心思想:人找知识，会查资料；LLM找知识，会查向量数据库        
主要目标:补充LLM固有局限性，LLM的知识不是实时的，LLM可能不知道私有领域业务知识          
场景类比:可以把RAG的过程想象成为开卷考试。让LLM先翻书查找相关答案，再回答问题              
### （2）技术方案架构
离线步骤:文档加载->文档切分->向量化->灌入向量数据库           
在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复             
### （3）几个关键概念：
向量数据库的意义是快速的检索             
向量数据库本身不生成向量，向量是由Embedding模型产生的             
向量数据库与传统的关系型数据库是互补的，不是替代关系，在实际应用中根据实际需求经常同时使用               

## 1.2 LangChain
### （1）LangChain定义
LangChain是一个用于开发由大型语言模型(LLM)驱动的应用程序的框架，官方网址：https://python.langchain.com/v0.2/docs/introduction/          
### （2）LCEL定义
LCEL(LangChain Expression Language),原来叫chain，是一种申明式语言，可轻松组合不同的调用顺序构成chain            
其特点包括流支持、异步支持、优化的并行执行、重试和回退、访问中间结果、输入和输出模式、无缝LangSmith跟踪集成、无缝LangServe部署集成            
### （3）LangSmith
LangSmith是一个用于构建生产级LLM应用程序的平台。通过它，您可以密切监控和评估您的应用程序，官方网址：https://docs.smith.langchain.com/          

## 1.3 Chroma
向量数据库，专门为向量检索设计的中间件          


# 2、前期准备工作
## 2.1 anaconda、pycharm 安装   
anaconda:提供python虚拟环境，官网下载对应系统版本的安装包安装即可                             
pycharm:提供集成开发环境，官网下载社区版本安装包安装即可                           
可参考如下视频进行安装【大模型应用开发基础】集成开发环境搭建Anaconda+PyCharm                                        
https://www.bilibili.com/video/BV1q9HxeEEtT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                            

## 2.2 OneAPI安装、部署、创建渠道和令牌 
### （1）OneAPI是什么
官方介绍：是OpenAI接口的管理、分发系统             
支持 Azure、Anthropic Claude、Google PaLM 2 & Gemini、智谱 ChatGLM、百度文心一言、讯飞星火认知、阿里通义千问、360 智脑以及腾讯混元             
### (2)安装、部署
使用官方提供的release软件包进行安装部署 ，详情参考如下链接中的手动部署：                  
https://github.com/songquanpeng/one-api                  
下载OneAPI可执行文件one-api并上传到服务器中然后，执行如下命令后台运行             
nohup ./one-api --port 3000 --log-dir ./logs > output.log 2>&1 &               
运行成功后，浏览器打开如下地址进入one-api页面，默认账号密码为：root 123456                 
http://IP:3000/              
### (3)创建渠道和令牌
创建渠道：大模型类型(通义千问)、APIKey(通义千问申请的真实有效的APIKey)             
创建令牌：创建OneAPI的APIKey，后续代码中直接调用此APIKey              

## 2.3 openai使用方案            
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择                   
可以参考这期视频:                   
【GraphRAG最新版本0.3.0对比实战评测】使用gpt-4o-mini和qwen-plus分别构建近2万字文本知识索引+本地/全局检索对比测试                   
https://www.bilibili.com/video/BV1maHxeYEB1/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                     
https://youtu.be/iXfsJrXCEwA                                               

## 2.4 langsmith配置         
直接在langsmith官网设置页中申请APIKey(这里可以选择使用也可以不使用)             
https://smith.langchain.com/o/93f0b841-d320-5df9-a9a0-25be027a4c09/settings                  


# 3、项目初始化
## 3.1 下载源码
GitHub中下载工程文件到本地，下载地址如下：                
https://github.com/NanGePlus/RagLangchainTest                               
https://gitee.com/NanGePlus/RagLangchainTest                                   

## 3.2 构建项目
使用pycharm构建一个项目，为项目配置虚拟python环境               
项目名称：RagLangchainTest                 

## 3.3 将相关代码拷贝到项目工程中           
直接将下载的文件夹中的文件拷贝到新建的项目目录中               

## 3.4 安装项目依赖          
pip install -r requirements.txt            
每个软件包后面都指定了本次视频测试中固定的版本号                  


# 4、项目测试          
## 4.1 准备测试文档             
这里以pdf文件为例，在input文件夹下准备了两份pdf文件                
健康档案.pdf:测试中文pdf文档处理                
llama2.pdf:测试英文pdf文档处理                 

## 4.2 文本预处理后进行灌库
在tools文件夹下提供了pdfSplitTest_Ch.py脚本工具用来处理中文文档、pdfSplitTest_En.py脚本工具用来处理英文文档                
vectorSaveTest.py脚本执行调用tools中的工具进行文档预处理后进行向量计算及灌库                
在使用python vectorSaveTest.py命令启动脚本前，需根据自己的实际情况调整代码中的如下参数：             
**调整1:选择使用哪种模型标志设置:**              
API_TYPE = "oneapi"  # openai:调用gpt模型；oneapi:调用oneapi方案支持的模型(这里调用通义千问)               
**调整2:openai模型相关配置 根据自己的实际情况进行调整:**                
OPENAI_API_BASE = "这里填写API调用的URL地址"               
OPENAI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"              
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"             
**调整3:oneapi相关配置(通义千问为例) 根据自己的实际情况进行调整:**             
ONEAPI_API_BASE = "这里填写oneapi调用的URL地址"            
ONEAPI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"                
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"                    
**调整4:配置测试文本类型:**            
TEXT_LANGUAGE = 'Chinese'  #Chinese 或 English                 
**调整5:配置待处理的PDF文件路径:**               
INPUT_PDF = "input/健康档案.pdf"              
**调整6:指定文件中待处理的页数范围，全部页数则填None:**               
PAGE_NUMBERS=None                  
PAGE_NUMBERS=[2, 3] # 指定页数     
**调整7:设置向量数据库chromaDB相关:**               
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径             
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称           

## 4.3 文本检索测试   
### （1）启动main脚本
在使用python main.py命令启动脚本前，需根据自己的实际情况调整代码中的如下参数：             
**调整1:设置langsmith环境变量:**           
os.environ["LANGCHAIN_TRACING_V2"] = "true"              
os.environ["LANGCHAIN_API_KEY"] = "这里填写申请的API_KEY"               
**调整2:设置待访问的向量数据库chromaDB相关:**            
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径               
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称              
**调整3:prompt模版设置相关:**           
PROMPT_TEMPLATE_TXT = "prompt_template.txt"  # 模版文件路径               
**调整4:选择使用哪种模型标志设置:**             
API_TYPE = "oneapi"  # openai:调用gpt模型；oneapi:调用oneapi方案支持的模型(这里调用通义千问)             
**调整5:openai模型相关配置 根据自己的实际情况进行调整:**                  
OPENAI_API_BASE = "这里填写API调用的URL地址"             
OPENAI_CHAT_API_KEY = "这里填写LLM模型的API_KEY"               
OPENAI_CHAT_MODEL = "gpt-4o-mini"              
OPENAI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"                  
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"              
**调整6:oneapi相关配置(通义千问为例) 根据自己的实际情况进行调整:**              
ONEAPI_API_BASE = "这里填写oneapi调用的URL地址"               
ONEAPI_CHAT_API_KEY = "这里填写LLM模型的API_KEY"                 
ONEAPI_CHAT_MODEL = "qwen-plus"                 
ONEAPI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"                        
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"                     
**调整7:API服务设置相关  根据自己的实际情况进行调整:**                         
PORT = 8012  # 服务访问的端口
**注意事项:**             
在测试使用调用oneapi的接口时，会报错如下所示：                
openai.BadRequestError: Error code: 400 - {'error': {'message': 'input should not be none.: payload.input.contents (request id: 2024082015351785023974771558878)', 'type': 'upstream_error', 'param': '400', 'code': 'bad_response_status_code'}}              
经过分析后，langchain_openai/embeddings包中的base.py源码中默认如下代码的true改为false                
check_embedding_ctx_length: bool = False                   
源码完整路径如下所示:                  
/opt/anaconda3/envs/RagLangchainTest/lib/python3.11/site-packages/langchain_openai/embeddings/base.py                    

### （2）运行apiTest脚本进行检索测试             
在运行python apiTest.py命令启动脚本前，需根据自己的实际情况调整代码中的如下参数，运行成功后，可以查看smith的跟踪情况                  
**调整1:默认非流式输出 True or False**                  
stream_flag = False                  
**调整2:检查URL地址中的IP和PORT是否和main脚本中相同**                  
url = "http://localhost:8012/v1/chat/completions"                      

## 4.4 文本检索测试(进阶1 re-ranker)   
re-ranker是一种用于信息检索系统的技术，旨在对初步检索到的结果进行进一步排序，以提高相关性和准确性                      
通常在一个信息检索任务中，系统会首先根据查询条件从大量文档或数据中初步筛选出一组候选结果          
然后通过一个re-ranker模型对这些候选结果进行细致的分析和重新排序，以确保最相关的结果排在最前面             
### （1）检索mainReranker脚本相关配置修改后运行          
首先通过该地址下载model文件夹，下载完成后将model文件拷贝到项目工程other文件夹下                          
链接: https://pan.baidu.com/s/12oUh-vOVgSqH1fcmhy33wA?pwd=1234 提取码: 1234                  
拉取最新的项目代码或将mainReranker.py脚本单独下载下来放入项目中                 
在使用python mainReranker.py命令启动脚本前，需根据自己的实际情况调整代码中的如下参数：                      
**调整1:安装依赖包:**                             
pip install sentence-transformers==3.0.1                                
**调整2:re-rank模型设置相关 根据自己的实际情况进行调整:**                                        
RERANK_MODEL = 'other/models/bge-reranker-large'                          
**调整3:其他参数调整参考4.3中所述相关配置**                      
### （2）运行apiTest脚本进行检索测试                   
python apiTest.py                       
运行成功后，可以查看smith的跟踪情况                                             
参数调整参考4.3中所述相关配置                   

## 4.5 文本检索测试(进阶2 PDF表格处理相关解决方案)   
首先通过该地址下载所需要用到的模型文件和测试PDF文件                     
下载完成后将模型文件夹拷贝到项目工程other/models文件夹下，将两份PDF文件拷贝到input文件夹下                  
链接: https://pan.baidu.com/s/1QjBa51iWbj2BgPJXSU5--A?pwd=1234 提取码: 1234   
### （1）解决方案思路1                        
流程:PDF转成图片->图片进行表格识别->将识别的表格保存为图片->调用视觉LLM进行摘要描述                                 
测试:运行tools文件夹下的pdfTableSplitTest01.py,使用的测试图片为input文件夹下的“健康档案(含表格01).pdf”         
### （2）解决方案思路2     
流程:PDF文本提取->PDF表格检测和提取->使用大模型将表格转换为自然语言描述->合并文本和表格描述                    
测试:运行tools文件夹下的pdfTableSplitTest02.py,使用的测试图片为input文件夹下的“健康档案(含表格02).pdf”          

## 4.6 文本检索测试(进阶3 添加历史对话记忆)    
### （1）先进行一个小的验证测试                       
使用LangChain框架中提供的方法进行测试        
运行test文件夹下的memoryTest.py脚本                                  
### （2）集成到工程项目     
流程:运行vectorSaveTest.py脚本灌库->运行mainMemry.py脚本启动服务->运行apiMemoryTest.py脚本测试                                     
**注意1:** mainMemry.py、apiMemoryTest.py脚本均在原脚本基础上进行迭代                
**注意2:** mainMemry.py脚本中使用prompt_template_memory.txt模版文件         
