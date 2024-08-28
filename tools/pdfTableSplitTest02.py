# 功能说明：提供一种解决思路将PDF文件中表格进行预处理后并保证文本的上下文连贯
# 准备工作：安装相关包
# pip install pdfplumber
# pip install "camelot-py[cv]"
# pip install PyPDF2==2.12.1
# pip install tabulate
# pip install pandas
# pip install openai
# 在使用camelot做处理需要ghostscript
# brew install ghostscript

import pdfplumber
import camelot
from openai import OpenAI
import pandas as pd
import os
import sys
import logging



# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 实例化一个OpenAI客户端对象
# client = OpenAI(
#     base_url="https://api.wlai.vip/v1",
#     api_key="sk-EhxvNWXkjzZJADfHA1Ac24Dd0f0b42B2B97f3725D3BcA378"
# )
# 实例化一个非OpenAI客户端对象
client = OpenAI(
    base_url="http://139.224.72.218:3000/v1",
    api_key="sk-eNbcweTEQV6L5Iw4F0B033219a1149C9Ab77501e690aD218"
)


# Step 1: PDF文本提取  返回拼接后的文本
def extract_text_from_pdf(pdf_path):
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                else:
                    logger.info(f"警告: 在该{page_number}页面上未找到文本")
    except Exception as e:
        logger.info(f"Error: Failed to extract text from {pdf_path}. Exception: {e}")
    return pages_text


# Step 2: 表格检测和提取  返回提取到的表格数据
def extract_tables_from_pdf(pdf_path):
    filtered_tables = []
    try:
        # 对所有页面进行表格检测
        # line_scale参数调整表格线条检测灵敏度
        # 默认值是15
        # 较小的数值适合表格线条较粗或较明显的PDF文档，适用于避免将文字或其他非表格元素误识别为表格线条
        # 较大的数值适用于表格线条较细、较模糊的PDF文档，适用于需要确保所有表格线条都被识别的场景
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice', line_scale=40)
        if tables.n == 0:
            logger.info(f"警告: 在{pdf_path}上未识别到表格。正在尝试使用 'stream' 方式再次进行检测")
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            if tables.n == 0:
                logger.info(f"警告: 在{pdf_path}上未识别到表格")
        # 对识别的表格进行一些过滤操作
        for table in tables:
            logger.info(f"获取到的表格内容: {table.df}")
            # 过滤掉只包含少量文字或行数过少的表格
            if len(table.df) > 0 and len(table.df.columns) > 0:
                filtered_tables.append(table)
        # 输出处理后的表格数量
        logger.info(f"最后检测到的表格数量是: {len(filtered_tables)}")

    except Exception as e:
        logger.info(f"Error: Failed to extract tables from {pdf_path}. Exception: {e}")

    return filtered_tables


# Step 3: 使用大模型将表格转换为自然语言描述  返回描述的内容
def generate_table_description(table_df):
    try:
        if table_df.empty:
            return "The table is empty."

        # 将DataFrame转换为Markdown表格格式，便于描述
        table_markdown = table_df.to_markdown(index=False)
        # 定一个prompt模版
        prompt = (
            "请使用中文对下表内容进行详细的自然语言描述，记住千万不要使用表格方式进行描述：\n\n"
            f"{table_markdown}\n"
        )
        # 调用大模型进行处理
        response = client.chat.completions.create(
            # model="gpt-4o-mini",
            model="qwen-plus",
            max_tokens=1200,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [{"type": "text","text": prompt},
                ],
            }],
        )
        logger.info(f"LLM给出的表格的描述是: {response.choices[0].message.content.strip()}")
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        logger.info(f"Error: Failed to generate description for table. Exception: {e}")
        return "An error occurred while generating the table description."


# Step 4: 合并文本和表格描述
def process_pdf(pdf_path):
    try:
        # 从pdf中提取文本内容
        pages_text = extract_text_from_pdf(pdf_path)
        # 从pdf中提取表格内容
        tables = extract_tables_from_pdf(pdf_path)

        # 判断是否从文件中提取到文本  若无则返回空字符
        if not pages_text:
            logger.info("错误: 未从文件中提取到文本")
            return ""

        # 判断是否从文件中提取到文本  提取到文本则执行下面的逻辑
        # 创建一个字典，将每页的表格存储起来
        tables_per_page = {}
        # 对识别后的表格处理后的内容进行处理  页码和页码中识别的表格数据对应
        for table in tables:
            page_number = table.page
            if page_number in tables_per_page:
                tables_per_page[page_number].append(table.df)
            else:
                tables_per_page[page_number] = [table.df]

        full_text = ""
        for page_number, text in enumerate(pages_text, start=1):
            full_text += f"--- 这是第{page_number}页的内容如下： ---\n"
            full_text += text.strip() + "\n"
            # 如果对应页面存在表格
            if page_number in tables_per_page:
                for idx, df in enumerate(tables_per_page[page_number], start=1):
                    # 为表格生成详细描述
                    description = generate_table_description(df)
                    full_text += f"表格{idx}的详细内容是:\n{description}\n\n"
            full_text += "\n"

        return full_text
    except Exception as e:
        logger.info(f"Error: Failed to process {pdf_path}. Exception: {e}")
        return ""


# Step 5: 将结果保存到文本文件
def save_to_text_file(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"成功: 文本已经成功保存到{output_path}中")
    except Exception as e:
        logger.info(f"Error: Failed to save content to {output_path}. Exception: {e}")


# 主函数
if __name__ == "__main__":
    pdf_path = "../input/健康档案(含表格02).pdf"  # 请输入您的 PDF 文件路径
    output_path = "./output.txt"  # 结果输出路径

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        logger.info(f"Error: The file {pdf_path} does not exist.")
        sys.exit(1)

    # 处理PDF
    final_text = process_pdf(pdf_path)

    if final_text:
        save_to_text_file(final_text, output_path)
    else:
        logger.info("Error: No content was processed from the PDF.")
