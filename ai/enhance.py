import os
import json
import sys

import dotenv
import argparse

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from structure import Structure
if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

# def main():
#     args = parse_args()
#     model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
#     language = os.environ.get("LANGUAGE", 'Chinese')
#
#     data = []
#     with open(args.data, "r") as f:
#         for line in f:
#             data.append(json.loads(line))
#
#     seen_ids = set()
#     unique_data = []
#     for item in data:
#         if item['id'] not in seen_ids:
#             seen_ids.add(item['id'])
#             unique_data.append(item)
#
#     data = unique_data
#
#     print('Open:', args.data, file=sys.stderr)
#
#     llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="json_mode")
#     print('Connect to:', model_name, file=sys.stderr)
#     prompt_template = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system),
#         HumanMessagePromptTemplate.from_template(template=template)
#     ])
#
#     chain = prompt_template | llm
#
#     for idx, d in enumerate(data):
#         try:
#             response: Structure = chain.invoke({
#                 "language": language,
#                 "content": d['summary']
#             })
#             d['AI'] = response.model_dump()
#         except langchain_core.exceptions.OutputParserException as e:
#             print(f"{d['id']} has an error: {e}", file=sys.stderr)
#             d['AI'] = {
#                  "tldr": "Error",
#                  "motivation": "Error",
#                  "method": "Error",
#                  "result": "Error",
#                  "conclusion": "Error"
#             }
#         with open(args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl'), "a") as f:
#             f.write(json.dumps(d) + "\n")
#
#         print(f"Finished {idx+1}/{len(data)}", file=sys.stderr)


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data

    print('Open:', args.data, file=sys.stderr)

    # 【重要改动 1】: 移除 .with_structured_output()，使用最基础的LLM
    llm = ChatOpenAI(model=model_name)
    print('Connect to:', model_name, file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    # 【重要改动 2】: 只处理第一条数据，然后打印并退出
    if data:
        first_item = data[0]
        print("--- STARTING DEBUG PRINT ---", file=sys.stderr)
        try:
            # 调用LLM
            response_message = chain.invoke({
                "language": language,
                "content": first_item['summary']
            })

            # 打印返回值的类型
            print(f"Type of response: {type(response_message)}", file=sys.stderr)

            # 打印返回值的完整内容
            print(f"Full response object: {response_message}", file=sys.stderr)

            # 单独打印核心的 content 属性
            print(f"Response content attribute: {response_message.content}", file=sys.stderr)

        except Exception as e:
            print(f"An error occurred during invoke: {e}", file=sys.stderr)
            # 打印完整的错误追溯信息
            import traceback
            traceback.print_exc()

        print("--- ENDING DEBUG PRINT ---", file=sys.stderr)

    # 【重要改动 3】: 退出程序
    print("Debug print finished. Exiting.", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
