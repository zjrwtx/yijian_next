# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from colorama import Fore
import sys
import os
import asyncio
from pathlib import Path
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
    # MCPToolkit,
)
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType, ModelType
from camel.utils import print_text_animated
from image_analysis import analyze_images
# config_path = Path(__file__).parent / "mcp_servers_config.json"
# mcp_toolkit = MCPToolkit(config_path=str(config_path))
models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0},
        ),
    }


    # Configure toolkits
tools = [
    *BrowserToolkit(
        headless=False,  # Set to True for headless mode (e.g., on remote servers)
        web_agent_model=models["browsing"],
        planning_agent_model=models["planning"],
    ).get_tools(),
    *VideoAnalysisToolkit(model=models["video"]).get_tools(),
    *AudioAnalysisToolkit().get_tools(),  # This requires DEEPSEEK Key
    *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
    *ImageAnalysisToolkit(model=models["image"]).get_tools(),
    SearchToolkit().search_duckduckgo,
    SearchToolkit().search_google,  # Comment this out if you don't have google search
    *ExcelToolkit().get_tools(),
    *FileWriteToolkit(output_dir="./").get_tools(),
    # *mcp_toolkit.get_tools()
]
def get_image_paths():
    """获取用户输入的图片路径"""
    print(Fore.CYAN + "请输入图片路径（每行一个，输入空行结束）：")
    paths = []
    while True:
        path = input().strip()
        if not path:
            break
        if os.path.exists(path):
            paths.append(path)
        else:
            print(Fore.RED + f"警告：文件 {path} 不存在，已跳过")
    return paths

def analyze_test_results(patient_info: str) -> str:
    """
    分析检验结果并生成报告
    
    Args:
        patient_info (str): 包含患者信息和检验结果的文本
        
    Returns:
        str: 生成的检验报告
    """
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=ChatGPTConfig(temperature=0.8, n=3).as_dict(),
    )
    
    # 配置角色和参数
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
    critic_kwargs = dict(verbose=True)
    
    # 创建角色扮演会话
    role_play_session = RolePlaying(
        "医学检验教授",
        "检验人员",
        critic_role_name="临床专家",
        task_prompt="分析以下医学检验结果，逐步思考后生成最终的带有诊断和建议等信息的详细高质量检验报告，不要进行任何多余的步骤，且保存最终分析好的检验报告为txt文件到本地" + patient_info,
        with_task_specify=False,
        with_critic_in_the_loop=False,
        assistant_agent_kwargs=assistant_agent_kwargs,
        user_agent_kwargs=user_agent_kwargs,
        critic_kwargs=critic_kwargs,
        output_language="中文",
    )

    # 初始化对话
    chat_turn_limit, n = 15, 0  # 减少对话轮数以提高效率
    input_msg = role_play_session.init_chat()
    responses = []
    
    # 进行对话
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)
        
        # 检查是否终止
        if assistant_response.terminated or user_response.terminated:
            break
            
        # 收集响应
        responses.append(f"AI Assistant: {assistant_response.msg.content}")
        responses.append(f"AI User: {user_response.msg.content}")
        
        # 检查任务是否完成
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
            
        input_msg = assistant_response.msg
    
    # 提取最终报告 - 这里假设助手的最后一个响应包含完整报告
    final_report = assistant_response.msg.content if assistant_response else "未能生成报告"
    
    return final_report

def main() -> None:
    
    # await mcp_toolkit.connect()
    # 获取图片路径并分析
    image_paths = get_image_paths()
    if image_paths:
        print(Fore.YELLOW + "正在分析图片...")
        image_analysis_result = analyze_images(image_paths)
        patient_info = f"\n\n图片分析结果：\n{image_analysis_result}"
    else:
        patient_info = "\n\n没有提供图片进行分析。"
    input_other_patient_note = input("请输入其他患者信息：")
    task_prompt = "分析以下医学检验结果，逐步思考后生成最终的带有诊断和建议等信息的详细高质量检验报告，不要进行任何多余的步骤，且保存最终分析好的检验报告为txt文件到本地" + patient_info + "\n\n其他患者信息：" + input_other_patient_note
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=ChatGPTConfig(temperature=0.8, n=3).as_dict(),
    )
    
    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
    critic_kwargs = dict(verbose=True)
    role_play_session = RolePlaying(
        "医学检验教授",
        "检验人员",
        critic_role_name="临床专家",
        task_prompt=task_prompt,
        with_task_specify=False,
        with_critic_in_the_loop=False,
        assistant_agent_kwargs=assistant_agent_kwargs,
        user_agent_kwargs=user_agent_kwargs,
        critic_kwargs=critic_kwargs,
        output_language="中文",
    )

    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )
    print(
        Fore.MAGENTA
        + f"Critic sys message:\n{role_play_session.critic_sys_msg}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "Specified task prompt:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

    chat_turn_limit, n = 50, 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        print_text_animated(
            Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN + f"AI Assistant:\n\n{assistant_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

        input_msg = assistant_response.msg


if __name__ == "__main__":
    main()






