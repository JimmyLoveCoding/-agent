"""
agent.py —— 材料性质预测助手（终端版）
=========================================================
用户提问 -> DeepSeek 决定调用工具 -> predict.predict_material_property
-> 结果回填 -> DeepSeek 组织回答。

用法：conda activate agent ; python agent.py     （.env 里配 DEEPSEEK_API_KEY）
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import predict as P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"

SYSTEM_PROMPT = """你是一个材料性质预测助手。

用户问材料性质时，必须调用 predict_material_property 工具查询，不要凭记忆回答。
工具会按化学式生成若干候选结构，挑出代表结构，再用 GNN 预测三种性质；
返回 prediction（三项预测值）和 n_structures（候选数）。生成的结构未做弛豫。

用中文简洁回答，先给结论。prediction 是模型近似值、不是实验值。"""


def get_client():
    """读取 .env 并创建 DeepSeek 客户端。"""
    load_dotenv(os.path.join(HERE, ".env"))
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)


def run_agent_loop(client, messages):
    """一轮对话：反复「问模型 -> 若它要调工具就执行并回填」，直到它给出回答。"""
    while True:
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=P.TOOL_SCHEMA,
        ).choices[0].message
        messages.append(msg)

        if not msg.tool_calls:                     # 模型没要工具 -> 这就是最终回答
            return msg.content or ""

        for tc in msg.tool_calls:                  # 逐个执行工具，把结果回填进历史
            args = json.loads(tc.function.arguments or "{}")
            result = P.TOOL_FUNCS[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})


def main():
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("材料性质预测助手（输入 exit 退出）")
    while True:
        q = input("\n你 > ").strip()
        if q in ("exit", "quit", "退出"):
            break
        messages.append({"role": "user", "content": q})
        print(f"\n助手：{run_agent_loop(client, messages)}")


if __name__ == "__main__":
    main()
