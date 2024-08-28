# 功能说明：提供一种解决思路将PDF文件中表格进行预处理
# 准备工作：安装相关包
# pip install PyMuPDF
# pip install torchvision
# pip install timm

import os
import logging
import fitz
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModelForObjectDetection
import torch
import base64
from openai import OpenAI
import chromadb
from chromadb.config import Settings


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 将PDF每页转成一个PNG图像
def pdf2images(pdf_file):
    # 保存路径为原PDF文件名（不含扩展名）
    output_directory_path, _ = os.path.splitext(pdf_file)
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    # 加载PDF文件
    pdf_document = fitz.open(pdf_file)
    # 每页转一张图
    for page_number in range(pdf_document.page_count):
        # 取一页
        page = pdf_document[page_number]
        # 转图像
        pix = page.get_pixmap()
        # 从位图创建PNG对象
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # 保存PNG文件
        image.save(f"./{output_directory_path}/page_{page_number + 1}.png")
    # 关闭PDF文件
    pdf_document.close()
    # 返回存储图片的文件夹路径
    return output_directory_path

# 缩放图像
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image

# 图像预处理
detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 加载引用TableTransformer模型
# 从本地文件夹中直接加载引用模型
model = AutoModelForObjectDetection.from_pretrained(
    "../other/models/table-transformer-detection",local_files_only=True
)
# 从HuggingFace中直接下载模型
# model = AutoModelForObjectDetection.from_pretrained(
#     "microsoft/table-transformer-detection"
# )

# 识别后的坐标换算与后处理
# 坐标转换
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
# 区域缩放
def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes
# 从模型输出中取定位框坐标
def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects

# 识别表格，并将表格部分单独存为图像文件
def detect_and_crop_save_table(file_path):
    # 加载图像（PDF页）
    image = Image.open(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    # 输出路径
    cropped_table_directory = os.path.join(os.path.dirname(file_path), "table_images")
    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)
    # 预处理
    pixel_values = detection_transform(image).unsqueeze(0)
    # 识别表格
    with torch.no_grad():
        outputs = model(pixel_values)
    # 后处理，得到表格子区域
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)
    logger.info(f"识别的表格数量: {len(detected_tables)}")

    for idx in range(len(detected_tables)):
        # 将识别从的表格区域单独存为图像
        cropped_table = image.crop(detected_tables[idx]["bbox"])
        cropped_table.save(os.path.join(cropped_table_directory,f"{filename}_{idx}.png"))
    return cropped_table_directory


# 实例化一个OpenAI客户端对象
client = OpenAI(
    base_url="https://api.wlai.vip/v1",
    api_key="sk-EhxvNWXkjzZJADfHA1Ac24Dd0f0b42B2B97f3725D3BcA378"
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def image_qa(query, image_path):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        seed=42,
        messages=[{
            "role": "user",
              "content": [
                  {"type": "text", "text": query},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}",
                      },
                  },
              ],
        }],
    )

    # return response.choices[0].message.content
    return f"{image_path} 中包含的内容是{response.choices[0].message.content}"

# get_embeddings方法计算向量
def get_embeddings(texts):
    try:
        data = client.embeddings.create(input=texts,model="text-embedding-3-small").data
        return [x.embedding for x in data]
    except Exception as e:
        logger.info(f"生成向量时出错: {e}")
        return []

# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # 调用向量生成get_embeddings方法  根据调用的API不同进行选择
        response = get_embeddings(batch)
        results.extend(response)
    return results

# 定义向量数据库类
class NewVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()
        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    # 向collection中添加文档与向量
    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    # 向collection中添加图像
    def add_images(self, image_paths):
        documents = [
            image_qa("请简要描述图片中的信息",image)
            for image in image_paths
        ]
        logger.info(f"视觉模型处理后的图片描述信息结果: {documents}")
        self.collection.add(
            embeddings=self.embedding_fn(documents),  #每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))],  # 每个文档的id
            metadatas=[{"image": image} for image in image_paths] # 用metadata标记源图像路径
        )

    # 检索向量数据库
    def search(self, query, top_n):
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results



if __name__ == "__main__":
    # 1、pdf文档预处理
    # 调用pdf2images方法将pdf文档转成图片  返回截取的图片所在路径
    output_directory_path = pdf2images("../input/健康档案(含表格01).pdf")
    logger.info(f"最后得到的pdf图片存储路径是: {output_directory_path}")
    # 识别每张图片中的表格并将表格单独保存为图片
    for file in os.listdir(output_directory_path):
        if file.endswith('.png'):
            cropped_table_directory = detect_and_crop_save_table(os.path.join(output_directory_path, file))
    logger.info(f"最后得到的表格图片存储路径是: {cropped_table_directory}")

    # 2、将处理后的表格图片进行灌库
    # 将文件夹中的图片 存到images数组中
    images = []
    for file in os.listdir(cropped_table_directory):
        if file.endswith('.png'):
            # 打开图像
            images.append(os.path.join(cropped_table_directory, file))
    # 将图片进行灌库
    new_db_connector = NewVectorDBConnector("table_demo001",generate_vectors)
    new_db_connector.add_images(images)

    # 3、测试查询
    # query  = "哪个模型在AGI Eval数据集上表现最好。得分多少"
    query = "王五的空腹血糖是多少"
    results = new_db_connector.search(query, 1)
    metadata = results["metadatas"][0]
    logger.info(f"最后得到的检索结果是: {metadata}")
    response = image_qa(query,metadata[0]["image"])
    logger.info(f"最后得到的回复是: {response}")









