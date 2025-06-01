from getpass import getpass
import os
import textwrap
from PIL.Image import OPEN
from camel.retrievers import AutoRetriever
from camel.types import StorageType
from typing import List, Dict
from dotenv import load_dotenv
from camel.retrievers import HybridRetriever

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.toolkits import(
     FunctionTool, SearchToolkit,PubMedToolkit,GoogleScholarToolkit,ArxivToolkit,SemanticScholarToolkit
     ,FileWriteToolkit,
     AsyncBrowserToolkit,
    #  ThinkingToolkit,
     RetrievalToolkit,
     MCPToolkit
     
)
import json
import asyncio
from camel.types import ModelPlatformType, ModelType  
from camel.societies.workforce import Workforce
from camel.configs import DeepSeekConfig,ChatGPTConfig,GeminiConfig
from pathlib import Path
load_dotenv()
from camel.logger import set_log_level
set_log_level(level="DEBUG")

# 配置API密钥
openai_api_key = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = openai_api_key
config_path = Path(__file__).parent / "mcp_servers_config.json"
mcp_toolkit = MCPToolkit(config_path=str(config_path))

tools=[
    SearchToolkit().search_duckduckgo,
    # SearchToolkit().search_google,
    PubMedToolkit().get_tools,
    ArxivToolkit().get_tools,
    *FileWriteToolkit().get_tools(),
    *RetrievalToolkit().get_tools(),
    *mcp_toolkit.get_tools()
    # *AsyncBrowserToolkit(headless=False).get_tools(),
    # *BrowserToolkit(headless=False).get_tools(),
    # ThinkingToolkit().get_tools(),
]
def make_medical_agent(
    role: str,
    persona: str,
    example_output: str,
    criteria: str,
) -> ChatAgent:
    mcp_toolkit.connect_sync()
    msg_content = textwrap.dedent(
        f"""\
        您是检验科的医疗专业人员。
        您的角色：{role}
        您的职责和特点：{persona}
        输出格式示例：
        {example_output}
        您的分析标准：
        {criteria}
        """
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="医学检验专业人员",
        content=msg_content,
    )
    model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
    # model = ModelFactory.create(
    # model_platform=ModelPlatformType.GEMINI,
    # model_type=ModelType.GEMINI_2_5_PRO_EXP,
    # model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
    )
    mcp_toolkit.disconnect_sync()

    return agent

# 创建临床记录总结器
summarizer_persona = (
    '您是专门将医患对话或其他患者信息转化为结构化临床记录的医疗文书。'
    '您专注于捕捉关键医疗信息、患者病史和临床观察结果。'
    '您需要将内容保存为txt文件到本地'
)

summarizer_example = (
    '临床摘要\n'
    '患者信息：\n'
    '- 姓名：[患者姓名]\n'
    '- 日期：[就诊日期]\n'
    '主诉：[主要症状]\n'
    '现病史：[详细描述]\n'
    '当前症状：[当前症状列表]\n'
    '相关病史：[既往病史]'
)

summarizer_criteria = textwrap.dedent(
    """\
    1. 记录所有相关医疗信息
    2. 保持医学术语准确性
    3. 遵循标准临床文档格式
    4. 包含症状的时间信息
    5. 记录提及的任何过敏或药物
    
    """
)

clinical_summarizer = make_medical_agent(
    "临床记录总结器",
    summarizer_persona,
    summarizer_example,
    summarizer_criteria,
)

# 创建临床分析器
analyzer_persona = (
    '您是专门分析患者症状和病史以识别潜在病症的临床分析师。'
    '您运用模式识别和医学知识来建议可能的诊断。'
    '您需要将内容保存为txt文件到本地'
)

analyzer_example = (
    '临床分析\n'
    '主要症状：\n'
    '- [症状1] - 持续时间和严重程度\n'
    '- [症状2] - 持续时间和严重程度\n'
    '潜在病症：\n'
    '1. [病症1] - 主要考虑因素...\n'
    '2. [病症2] - 次要考虑因素...\n'
    '存在的风险因素：\n'
    '- [风险因素1]\n'
    '- [风险因素2]'
)

analyzer_criteria = textwrap.dedent(
    """\
    1. 识别主要和次要症状
    2. 按可能性顺序列出潜在病症
    3. 考虑患者人口统计学和风险因素
    4. 注意任何危险信号或紧急问题
    5. 提出鉴别诊断
    """
)

clinical_analyzer = make_medical_agent(
    "临床分析器",
    analyzer_persona,
    analyzer_example,
    analyzer_criteria,
)
# 创建检验资料搜索器
search_persona = (
    '您是专门根据临床分析结果搜索相关检验资料的医学信息专家。'
    '您使用多种学术资源来查找最新的检验指南和研究资料。'
    '您需要将内容保存为txt文件到本地'
)

search_example = (
    '检验资料搜索结果\n'
    '1. [检验名称]\n'
    '   - 来源：[来源名称]\n'
    '   - 摘要：[关键信息摘要]\n'
    '   - 链接：[资源链接]\n'
    '2. [检验名称]\n'
    '   - 来源：[来源名称]\n'
    '   - 摘要：[关键信息摘要]\n'
    '   - 链接：[资源链接]'
)

search_criteria = textwrap.dedent(
    """\
    1. 根据临床分析结果确定搜索关键词
    2. 使用多种学术资源进行搜索
    3. 提供最新的检验指南和研究资料
    4. 包含关键信息摘要和来源链接
    5. 确保信息的权威性和时效性
    """
)

test_searcher = make_medical_agent(
    "检验资料搜索器",
    search_persona,
    search_example,
    search_criteria,

)
# 创建检验项目推荐器
recommender_persona = (
    '您是根据临床表现推荐适当诊断检验的实验室检验专家。'
    '您始终保持对当前检验指南和方案的了解。'
    '您需要将内容保存为txt文件到本地'
)

recommender_example = (
    '实验室检验推荐\n'
    '推荐检验：\n'
    '1. [检验名称]\n'
    '   - 目的：[为何需要此项检验]\n'
    '   - 预期结果：[我们寻找的内容]\n'
    '2. [检验名称]\n'
    '   - 目的：[为何需要此项检验]\n'
    '   - 预期结果：[我们寻找的内容]'
)

recommender_criteria = textwrap.dedent(
    """\
    1. 推荐适当的诊断检验
    2. 解释每项检验的目的
    3. 考虑成本效益
    4. 遵循当前检验指南
    5. 注意任何检验前要求
    """
)

test_recommender = make_medical_agent(
    "检验项目推荐器",
    recommender_persona,
    recommender_example,
    recommender_criteria,
)

# 创建医院检验项目匹配器
matcher_persona = (
    '您是专门将推荐检验项目与医院实际开展项目进行匹配的专家。'
    '您了解医院检验科的能力范围，能够提供切实可行的检验开单建议。'
    '您需要将内容保存为txt文件到本地'
)

matcher_example = (
    '医院检验项目匹配结果\n'
    '可开展检验项目：\n'
    '1. [检验名称]\n'
    '   - 对应推荐项目：[原推荐项目]\n'
    '   - 临床意义：[临床价值说明]\n'
    '   - 注意事项：[采集或检验要求]\n'
    '2. [检验名称]\n'
    '   - 对应推荐项目：[原推荐项目]\n'
    '   - 临床意义：[临床价值说明]\n'
    '   - 注意事项：[采集或检验要求]\n'
    '替代检验建议：\n'
    '- 对于[无法开展的项目]，建议使用[替代项目]'
)

matcher_criteria = textwrap.dedent(
    """\
    1. 准确匹配推荐项目与医院可开展项目
    2. 提供替代检验建议
    3. 考虑检验的成本效益
    4. 注明检验前准备要求
    5. 按照临床优先级排序
    一定要把推荐的项目都各个都医院开展的项目进行匹配
    医院的开展项目的地址如下：https://raw.githubusercontent.com/zjrwtx/alldata/refs/heads/main/jianyan.md
    """
)





hospital_matcher_tools = [
   *RetrievalToolkit().get_tools(),
   *FileWriteToolkit().get_tools(),
 
]

# hospital_matcher_model = ModelFactory.create(
#     model_platform=ModelPlatformType.GEMINI,
#     model_type=ModelType.GEMINI_2_5_PRO_EXP,
#     model_config_dict=GeminiConfig(temperature=0.2).as_dict(),)

hospital_matcher_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),)

msg_content = textwrap.dedent(
        f"""\
        您是检验科的医疗专业人员。
        您的角色：检验项目开单员
        您的职责和特点：{matcher_persona}
        输出格式示例：
        {matcher_example}
        您的分析标准：
        {matcher_criteria}
        """
    )
hospital_matcher=ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="检验项目开单员",
        content=msg_content,
    ),
    model=hospital_matcher_model,
    tools=hospital_matcher_tools,   
   
)
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict())

# model = ModelFactory.create(
# model_platform=ModelPlatformType.GEMINI,
# model_type=ModelType.GEMINI_2_5_PRO_EXP,
# model_config_dict=GeminiConfig(temperature=0.2).as_dict(),)
# 创建工作团队

workforce = Workforce(
    '医学检验工作流',
    coordinator_agent_kwargs = {"model": model},
    task_agent_kwargs = {"model": model},
)
workforce.add_single_agent_worker(
    '临床记录总结器：将医患对话转化为结构化临床记录',
    worker=clinical_summarizer,
).add_single_agent_worker(
    '临床分析器：分析症状并建议潜在病症',
    worker=clinical_analyzer,
).add_single_agent_worker(
    '检验资料搜索器：根据分析结果搜索相关检验资料',
    worker=test_searcher,
).add_single_agent_worker(
    '检验项目推荐器：推荐适当的诊断检验',
    worker=test_recommender,
)

# add_single_agent_worker(
#     '医院检验项目匹配器：将推荐项目与医院实际项目匹配',
#     worker=hospital_matcher, 
# )

# 更新process_clinical_case函数
def process_clinical_case(conversation_text: str) -> str:
    """
    通过医学检验工作流处理临床病例
    
    参数：
        conversation_text (str): 医患对话文本
    
    返回：
        str: 最终实验室报告
    """
    task = Task(
        content="通过实验室工作流处理此临床病例。"
        "step1:总结临床对话。然后分析症状并建议潜在病症。"
        "step2:根据症状搜索搜索相关检验资料"
        "step3:查看搜索到的url内容"
        "step4:接着推荐适当的实验室检验。记得语言都是中文"
        "step5:基于推荐的检验项目，再结合医院开展的检验项目进行最终的可开单的推荐",
        additional_info=conversation_text,
        id="0",
    )
    
    processed_task = workforce.process_task(task)
    return processed_task.result

# # 示例临床对话
# example_conversation = """
# Doctor: 您好，请详细说说您的症状。

# Patient: 医生您好，我这情况很复杂，大概半年前开始，最初是觉得特别疲劳，以为是工作压力大。但后来出现了关节疼痛，特别是手指、手腕和膝盖，早上特别明显，活动一会儿后会好一些。

# Doctor: 关节疼痛的具体表现是什么样的？是双侧还是单侧？

# Patient: 是双侧的，而且是对称的。最近三个月，我还发现脸上和胸前经常出现一些红斑，像蝴蝶一样的形状，太阳光照射后会加重。有时候还会发低烧，37.5-38度之间。

# Doctor: 明白了。您有没有出现口腔溃疡、脱发或者其他症状？

# Patient: 确实有！经常会长溃疡，差不多两周一次。最近半年掉头发特别厉害，还总觉得眼睛干涩。最让我担心的是，有时候会觉得胸闷气短，爬楼梯都困难，之前从来没有这种情况。

# Doctor: 我注意到您的手指关节有些肿胀。最近有没有出现手指发白或发紫的情况，特别是在寒冷环境下？

# Patient: 对，冬天的时候特别明显，手指会先发白，然后变成紫色，最后变红，还会感觉刺痛。我父亲说我最近消瘦很多，实际上我没有刻意减肥，但是半年内瘦了将近10公斤。

# Doctor: 您家族史中有类似的疾病史吗？或者其他自身免疫性疾病？

# Patient: 我姑姑好像也有类似的情况，具体什么病我不太清楚。我注意到最近经常感觉心跳很快，有时候会超过100下/分钟，还经常出现夜汗。

# Doctor: 您平时有服用什么药物吗？包括中药或保健品？

# Patient: 之前吃过止痛药和一些维生素，但效果不明显。最近还出现了肌肉疼痛，特别是大腿和上臂，感觉浑身没劲。有时候早上起床，手指会僵硬半小时左右才能活动自如。对了，最近还经常出现头痛，有时候会头晕，视物模糊。

# Doctor: 您的工作和生活习惯有什么变化吗？比如作息、压力源等。

# Patient: 工作压力一直都挺大的，但最近半年确实更甚。经常失眠，睡眠质量特别差。有时候会莫名其妙地焦虑。最近还发现，一些以前经常吃的食物现在会出现过敏反应，起荨麻疹。

# Doctor: 您刚才提到的胸闷气短，有没有出现过胸痛？运动后会加重吗？

# Patient: 有时会隐隐作痛，但不是很剧烈。深呼吸的时候会感觉胸部不适，最近还出现了干咳的情况。有几次半夜被胸闷惊醒，同时伴有盗汗。
# """

example_conversation = """

患者男性，73岁，汉族。主诉：面色苍白伴乏力10天，排黑便2天。现病史：患者10天前无明显诱因出现面色苍白、乏力，活动后气促明显，伴腹胀、纳差。2天前开始排黑色糊状便，无晕厥，无畏冷、寒战、发热，无咳嗽、咳痰，无腹痛、反酸等不适。发病以来，睡眠欠佳，体重减轻约5kg。既往史：有高血压病史10余年，规律口服氨氯地平可控制。体格检查：BP 130/85mmHg，贫血外观，体型消瘦，睑结膜苍白，全身浅表淋巴结无肿大，胸骨无压痛，双肺呼吸音清，未闻及干湿性啰音。心率90次/分，律齐，各瓣膜区未闻及杂音。全腹软，无压痛、反跳痛，肝、脾肋下未触及。双下肢无水肿。实验室检查初步结果：WBC 5.2×10^9/L，RBC 2.8×10^12/L，Hb 58g/L，MCV 78fl，MCH 26.9pg，MCHC 305g/L，RDW-CV 19%，PLT 327×10^9/L。外周血涂片：红细胞大小不等，以小细胞为主，中心浅染区明显扩大。粪常规检查：黏液(+)，镜检白细胞(++)，隐血弱阳性

"""

if __name__ == "__main__":
    # 处理病例获取结果
    result = process_clinical_case(example_conversation)

    # 构建新的问答对
    new_qa = {
        "question": example_conversation.strip(),
        "answer": result.strip()
    }

    # 定义文件路径
    file_path = "qa_dataset.json"

    # 读取现有数据或初始化空列表
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []  # 如果不是数组，重置为空列表
            except json.JSONDecodeError:
                data = []  # 文件损坏则清空处理
    else:
        data = []

    # 添加新问答对
    data.append(new_qa)

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("新问答对已成功追加到 'qa_dataset.json'")