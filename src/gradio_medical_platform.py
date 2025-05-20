import gradio as gr
import os
import json
from pathlib import Path
from audio_to_text import audio_models
from src.recommend_inspect_item.recommed_inspect_item import process_clinical_case
from src.image_analysis.image_analysis import analyze_images
from src.Analysis_test_results import analyze_test_results
import data_auto_anaylse_with_research_inspiration as data_analysis
from synthetic_data_pipeline import SyntheticDataPipeline
from doctor_to_patient_data import simulate_doctor_patient_dialogue
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from benchmark_test import run_benchmark_from_gradio
import datetime

# SFT training imports
import torch
from datetime import datetime
import requests
# from camel.datagen.cotdatagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.datahubs.huggingface import HuggingFaceDatasetManager
from camel.datahubs.models import Record
# from unsloth import FastLanguageModel, is_bfloat16_supported
# from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


# 创建临时目录用于存储中间文件
os.makedirs("temp", exist_ok=True)
os.makedirs("temp/sft_data", exist_ok=True)
os.makedirs("temp/trained_models", exist_ok=True)
os.makedirs("export", exist_ok=True)

# 定义医疗风格主题颜色和CSS
MEDICAL_THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#E3F2FD",
        c100="#BBDEFB",
        c200="#90CAF9",
        c300="#64B5F6",
        c400="#42A5F5",
        c500="#2196F3",  # 医疗蓝色
        c600="#1E88E5",
        c700="#1976D2", 
        c800="#1565C0",
        c900="#0D47A1",
        c950="#0A2F6F",
    ),
    secondary_hue=gr.themes.Color(
        c50="#E8F5E9",
        c100="#C8E6C9",
        c200="#A5D6A7",
        c300="#81C784",
        c400="#66BB6A",
        c500="#4CAF50",  # 医疗绿色
        c600="#43A047",
        c700="#388E3C",
        c800="#2E7D32",
        c900="#1B5E20",
        c950="#0F3E14",
    ),
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#F8FAFC",
    body_background_fill_dark="#1E293B",
    body_text_color="#334155",
    body_text_color_dark="#E2E8F0",
    button_primary_background_fill="#2196F3",
    button_primary_background_fill_hover="#1976D2",
    button_primary_text_color="white",
    button_primary_border_color="#2196F3",
    button_secondary_background_fill="white",
    button_secondary_text_color="#2196F3",
    button_secondary_border_color="#2196F3",
    button_cancel_background_fill="#EF4444",
    button_cancel_background_fill_hover="#DC2626",
    button_cancel_text_color="white",
    button_cancel_border_color="#EF4444",
    block_background_fill="white",
    block_background_fill_dark="#334155",
    block_border_color="#E2E8F0",
    block_border_color_dark="#475569",
    block_radius="0.5rem",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)",
    block_title_text_color="#0F172A",
    block_title_text_color_dark="#F1F5F9",
    block_title_text_weight="600",
    block_label_text_color="#64748B",
    block_label_text_color_dark="#CBD5E1",
    block_label_text_weight="500",
    input_background_fill="white",
    input_background_fill_dark="#1E293B",
    input_border_color="#CBD5E1",
    input_border_color_dark="#475569",
    input_border_width="1px",
    input_background_fill_focus="white",
    input_background_fill_focus_dark="#1E293B",
    input_border_color_focus="#2196F3",
    input_border_color_focus_dark="#2196F3",
)

# 自定义医疗风格CSS样式
CUSTOM_CSS = """
.gradio-container {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

h1 {
    font-size: 1.8rem !important;
    border-bottom: 2px solid #2196F3 !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1.5rem !important;
    position: relative !important;
}

h1::before {
    content: "🏥" !important;
    margin-right: 0.5rem !important;
}

.tab-nav {
    border-bottom: 1px solid #E2E8F0 !important;
    margin-bottom: 20px !important;
    background-color: rgba(33, 150, 243, 0.05) !important;
    border-radius: 8px 8px 0 0 !important;
}

.tab-nav button {
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 16px !important;
    margin: 0 2px !important;
    color: #64748B !important;
    border: none !important;
}

.tab-nav button:hover {
    background-color: rgba(33, 150, 243, 0.1) !important;
    color: #1E88E5 !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #2196F3 !important;
    color: #2196F3 !important;
    background-color: rgba(33, 150, 243, 0.08) !important;
}

.block {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
    background-color: white !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.block:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.05) !important;
}

.block-label {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.3rem !important;
}

.primary-btn {
    text-transform: capitalize !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    border-radius: 6px !important;
    transition: all 0.3s !important;
    padding: 8px 16px !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3) !important;
}

button.secondary {
    border: 1px solid #2196F3 !important;
    color: #2196F3 !important;
    background-color: white !important;
    transition: all 0.3s !important;
}

button.secondary:hover {
    background-color: rgba(33, 150, 243, 0.05) !important;
}

.gr-input, .gr-textarea {
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2) !important;
    outline: none !important;
}

.gr-panel {
    border-radius: 8px !important;
    overflow: hidden !important;
}

.gr-accordion {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
}

.gr-accordion-header {
    background-color: rgba(33, 150, 243, 0.05) !important;
    padding: 12px 16px !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

.footer {
    text-align: center !important;
    margin-top: 2rem !important;
    padding: 1rem !important;
    color: #64748B !important;
    font-size: 0.8rem !important;
    border-top: 1px solid #E2E8F0 !important;
}

/* 新增医疗风格元素 */
.medical-icon {
    color: #2196F3 !important;
    margin-right: 6px !important;
}

.medical-alert {
    background-color: #FFF4E5 !important;
    border-left: 4px solid #FF9800 !important;
    padding: 12px 16px !important;
    margin-bottom: 1rem !important;
    border-radius: 4px !important;
    color: #7B341E !important;
}

.medical-success {
    background-color: #E8F5E9 !important;
    border-left: 4px solid #4CAF50 !important;
    padding: 12px 16px !important;
    margin-bottom: 1rem !important;
    border-radius: 4px !important;
    color: #1B5E20 !important;
}

/* 为医疗数据展示优化样式 */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

th {
    background-color: #2196F3 !important;
    color: white !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 12px !important;
}

td {
    padding: 10px 12px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}

tr:nth-child(even) {
    background-color: #F8FAFC !important;
}

tr:hover {
    background-color: #E3F2FD !important;
}
"""

# 1. 语音转文本页面
def process_audio(audio_path):
    try:
        # 转换音频为文本
        converted_text = audio_models.speech_to_text(audio_file_path=audio_path)
        # 保存转换后的文本到临时文件
        with open("temp/converted_text.txt", "w", encoding="utf-8") as f:
            f.write(converted_text)
        return converted_text
    except Exception as e:
        return f"处理失败: {str(e)}"

def text_to_audio(input_text, output_path="temp/output_audio.mp3"):
    try:
        # 将文本转换为语音
        audio_models.text_to_speech(input=input_text, storage_path=output_path)
        return output_path, "语音生成成功"
    except Exception as e:
        return None, f"处理失败: {str(e)}"

# 2. 检验项目推荐页面
def recommend_inspection_items(conversation_text):
    try:
        # 调用项目推荐模块
        result = process_clinical_case(conversation_text)
        
        # 保存推荐结果到临时文件
        with open("temp/recommendation_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
        
        # 创建问答数据
        qa_data = {conversation_text: result}
        
        # 保存问答数据到JSON文件
        save_qa_data_to_json(qa_data, "temp/recommendation_qa_data.json")
        
        return result
    except Exception as e:
        return f"处理失败: {str(e)}"

def load_converted_text():
    try:
        with open("temp/converted_text.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

# 3. 检验结果图像分析页面
def analyze_test_images(image_paths, other_patient_info):
    try:
        # 分析上传的图像
        if not image_paths:
            return "请上传至少一张图像"
        
        image_analysis_result = analyze_images(image_paths)
        
        # 准备完整的患者信息
        full_patient_info = f"图片分析结果：\n{image_analysis_result}\n\n其他患者信息：\n{other_patient_info}"
        
        # 保存分析结果和患者信息到临时文件
        with open("temp/image_analysis_result.txt", "w", encoding="utf-8") as f:
            f.write(full_patient_info)
            
        return "图像分析完成，结果已保存"
    except Exception as e:
        return f"处理失败: {str(e)}"

# 4. 检验报告生成页面
def generate_test_report():
    try:
        # 读取图像分析结果和患者信息
        with open("temp/image_analysis_result.txt", "r", encoding="utf-8") as f:
            patient_info = f.read()
        
        # 调用报告生成模块
        report = analyze_test_results(patient_info)
        
        # 保存生成的报告
        with open("temp/test_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # 创建问答数据
        qa_data = {patient_info: report}
        
        # 保存问答数据到JSON文件
        save_qa_data_to_json(qa_data, "temp/test_report_qa_data.json")
        
        return report
    except Exception as e:
        return f"报告生成失败: {str(e)}"

# 5. 检验数据分析与科研灵感页面
def analyze_csv_data(csv_file_path, analysis_requirements):
    try:
        if csv_file_path:
            # 使用用户上传的CSV文件
            result = data_analysis.process_lab_data_analysis(
                user_info=analysis_requirements,
                input_csv_path=csv_file_path
            )
        else:
            
            raise Exception("请上传CSV文件")
        
        # 保存分析结果
        with open("temp/data_analysis_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
            
        return result
    except Exception as e:
        return f"数据分析失败: {str(e)}"

# 6. 合成数据生成页面
def generate_synthetic_data(main_topic, total_examples, num_subtopics, temperature):
    try:
        # 创建合成数据生成器
        pipeline = SyntheticDataPipeline(temperature=temperature)
        
        # 生成数据
        data = pipeline.generate_synthetic_data(
            main_topic=main_topic,
            total_examples=total_examples,
            num_subtopics=num_subtopics
        )
        
        # 保存生成的数据
        output_file = "temp/synthetic_data.json"
        pipeline.save_to_file(data, output_file)
        
        # 读取并返回生成的数据
        with open(output_file, "r", encoding="utf-8") as f:
            result = f.read()
            
        return result
    except Exception as e:
        return f"数据生成失败: {str(e)}"

# 7. 病历生成医患对话页面
def generate_doctor_patient_dialogue(patient_history):
    try:
        # 生成医患对话
        dialogue = simulate_doctor_patient_dialogue(patient_history)
        
        # 保存生成的对话
        with open("temp/simulated_dialogue.txt", "w", encoding="utf-8") as f:
            f.write(dialogue)
            
        return dialogue
    except Exception as e:
        return f"对话生成失败: {str(e)}"

# 创建Benchmark测试相关函数
def run_benchmark(modules_to_test, test_samples):
    results = {
        "模块": [],
        "处理时间(秒)": [],
        "内存使用(MB)": [],
        "成功率(%)": []
    }
    
    for module_name, module_func, test_data in modules_to_test:
        print(f"测试模块: {module_name}")
        total_time = 0
        success_count = 0
        memory_usage = 0
        
        for i, sample in enumerate(test_data[:test_samples]):
            start_time = time.time()
            try:
                # 运行模块函数
                module_func(sample)
                success_count += 1
                # 简单模拟内存使用 (实际应用中应使用更准确的测量方法)
                memory_usage += len(str(sample)) * 0.001
            except Exception as e:
                print(f"样本 {i+1} 失败: {str(e)}")
            finally:
                end_time = time.time()
                total_time += (end_time - start_time)
        
        avg_time = total_time / test_samples if test_samples > 0 else 0
        success_rate = (success_count / test_samples) * 100 if test_samples > 0 else 0
        avg_memory = memory_usage / test_samples if test_samples > 0 else 0
        
        results["模块"].append(module_name)
        results["处理时间(秒)"].append(round(avg_time, 3))
        results["内存使用(MB)"].append(round(avg_memory, 2))
        results["成功率(%)"].append(round(success_rate, 1))
    
    return pd.DataFrame(results)

def plot_benchmark_results(df):
    # 创建性能图表
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 处理时间图表
    df.plot(x="模块", y="处理时间(秒)", kind="bar", ax=ax[0], color="blue", legend=False)
    ax[0].set_title("平均处理时间 (秒)")
    ax[0].set_ylabel("秒")
    
    # 内存使用图表
    df.plot(x="模块", y="内存使用(MB)", kind="bar", ax=ax[1], color="green", legend=False)
    ax[1].set_title("平均内存使用 (MB)")
    ax[1].set_ylabel("MB")
    
    # 成功率图表
    df.plot(x="模块", y="成功率(%)", kind="bar", ax=ax[2], color="orange", legend=False)
    ax[2].set_title("成功率 (%)")
    ax[2].set_ylabel("%")
    
    plt.tight_layout()
    
    # 将图表转换为base64编码的图像
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    return df, f"data:image/png;base64,{img_str}"

def get_test_samples():
    # 返回测试样本
    audio_samples = ["temp/output_audio.mp3"] * 5
    
    text_samples = [
        "患者男性，45岁，主诉胸闷气短3个月，加重1周。查体：BP 160/95mmHg，心率88次/分。",
        "患者女性，35岁，主诉关节疼痛伴晨僵2年，加重1个月。查体：双手近端指间关节肿胀。",
        "患者男性，60岁，主诉尿频尿急尿痛5天。查体：下腹部压痛，肾区叩击痛阴性。",
        "患者女性，28岁，主诉发热、咳嗽、咳痰3天。查体：体温38.5℃，双肺可闻及湿啰音。",
        "患者男性，50岁，主诉上腹痛、恶心、呕吐2天。查体：上腹部压痛，Murphy征阳性。"
    ]
    
    image_samples = [["i01.png", "i03.png"]] * 5
    
    return {
        "audio": audio_samples,
        "text": text_samples,
        "image": image_samples,
        "patient_histories": text_samples
    }

def run_system_benchmark(components_to_test, num_samples):
    test_samples = get_test_samples()
    modules_to_test = []
    
    # 根据选择的组件准备测试模块
    if "语音转文本" in components_to_test:
        modules_to_test.append(("语音转文本", lambda x: process_audio(x), test_samples["audio"]))
    
    if "检验项目推荐" in components_to_test:
        modules_to_test.append(("检验项目推荐", lambda x: recommend_inspection_items(x), test_samples["text"]))
    
    if "图像分析" in components_to_test:
        modules_to_test.append(("图像分析", lambda x: analyze_test_images(x, "测试患者信息"), test_samples["image"]))
    
    if "医患对话生成" in components_to_test:
        modules_to_test.append(("医患对话生成", lambda x: generate_doctor_patient_dialogue(x), test_samples["patient_histories"]))

    # 运行基准测试
    results_df = run_benchmark(modules_to_test, num_samples)
    
    # 生成结果图表
    df, img_data = plot_benchmark_results(results_df)
    
    # 将结果保存到文件
    results_df.to_csv("temp/benchmark_results.csv", index=False)
    
    return results_df, img_data

# COT数据生成和SFT训练相关函数
def create_chat_agent(api_key, model_choice, system_message):
    """创建用于生成COT数据的ChatAgent"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 定义不同的模型类型
        model_types = {
           "gpt-4.1": ModelType.GPT_4_1,
            "gpt-4o-mini": ModelType.GPT_4_1_MINI,
            "gpt-4o": ModelType.GPT_4O,
        }
        
        # 创建模型
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_types.get(model_choice, ModelType.GPT_4_1),
            model_config_dict=ChatGPTConfig().as_dict(),
        )
        
        # 创建ChatAgent
        chat_agent = ChatAgent(
            system_message=system_message,
            model=model,
            message_window_size=10,
        )
        
        return chat_agent, "成功创建ChatAgent"
    except Exception as e:
        return None, f"创建ChatAgent失败: {str(e)}"

def generate_cot_data(chat_agent, input_qa_data, output_file="temp/sft_data/generated_answers.json"):
    """生成并保存CoT思维链数据"""
    try:
        if not chat_agent:
            return "请先创建ChatAgent"
        
        # 解析输入的QA数据
        qa_data = json.loads(input_qa_data) if isinstance(input_qa_data, str) else input_qa_data
        
        # 创建CoTDataGenerator实例
        cot_generator = CoTDataGenerator(chat_agent, golden_answers=qa_data)
        
        # 记录生成的答案
        generated_answers = {}
        generation_results = []
        
        # 为每个问题生成答案
        for i, (question, golden_answer) in enumerate(qa_data.items(), 1):
            generation_results.append(f"问题 {i}: {question}")
            
            # 获取AI的思考过程和答案
            answer = cot_generator.get_answer(question)
            generated_answers[question] = answer
            generation_results.append(f"AI的思考过程和答案:\n{answer}")
            
            # 验证答案
            is_correct = cot_generator.verify_answer(question, answer)
            generation_results.append(f"答案验证结果: {'正确' if is_correct else '不正确'}")
            generation_results.append("-" * 50)
        
        # 导出生成的答案
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'qa_pairs': generated_answers
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 转换为Alpaca训练数据格式
        alpaca_data = []
        for question, answer in generated_answers.items():
            alpaca_data.append({
                "instruction": question,
                "input": "",
                "output": answer
            })
        
        alpaca_file = f"temp/sft_data/alpaca_format_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alpaca_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        return "\n".join(generation_results), alpaca_file
    except Exception as e:
        return f"生成CoT数据失败: {str(e)}", None

def upload_dataset_to_huggingface(alpaca_file, hf_token, username, dataset_name=None):
    """将生成的数据集上传到HuggingFace"""
    try:
        os.environ["HUGGING_FACE_TOKEN"] = hf_token
        
        # 读取转换后的数据
        with open(alpaca_file, 'r', encoding='utf-8') as f:
            transformed_data = json.load(f)
        
        # 初始化HuggingFaceDatasetManager
        manager = HuggingFaceDatasetManager()
        
        # 生成或验证数据集名称
        if dataset_name is None:
            dataset_name = f"{username}/medical-qa-dataset-{datetime.now().strftime('%Y%m%d')}"
        else:
            dataset_name = f"{username}/{dataset_name}"
        
        # 创建数据集
        print(f"创建数据集: {dataset_name}")
        dataset_url = manager.create_dataset(name=dataset_name)
        
        # 创建数据集卡片
        manager.create_dataset_card(
            dataset_name=dataset_name,
            description="医疗问答数据集 - 由CAMEL CoTDataGenerator生成",
            license="mit",
            language=["zh"],
            size_category="<1MB",
            version="0.1.0",
            tags=["camel", "medical", "question-answering", "chinese"],
            task_categories=["question-answering"],
            authors=[username]
        )
        
        # 将数据转换为Record对象并添加到数据集
        records = [Record(**item) for item in transformed_data]
        manager.add_records(dataset_name=dataset_name, records=records)
        
        return f"数据集已成功上传到HuggingFace: {dataset_url}"
    except Exception as e:
        return f"上传到HuggingFace失败: {str(e)}"

def train_model(model_name, alpaca_file, epochs, learning_rate, output_dir="temp/trained_models"):
    """使用Unsloth训练模型"""
    try:
        if not os.path.exists(alpaca_file):
            return "训练数据文件不存在"
        
        # 准备训练数据
        with open(alpaca_file, 'r', encoding='utf-8') as f:
            alpaca_data = json.load(f)
        
        # 将数据保存为jsonl格式用于加载
        jsonl_file = alpaca_file.replace(".json", ".jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in alpaca_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 加载模型和分词器
        max_seq_length = 2048
        dtype = None
        load_in_4bit = True
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # 添加LoRA适配器
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # 定义格式化提示词函数
        alpaca_prompt = """下面是一个描述任务的指令，以及提供进一步上下文的输入。请写一个适当地完成请求的回应。

### 指令:
{}

### 输入:
{}

### 回应:
{}"""
        
        EOS_TOKEN = tokenizer.eos_token
        
        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                texts.append(text)
            return {"text": texts}
        
        # 加载并处理数据集
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=jsonl_file, split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        # 设置训练参数
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"{output_dir}/{model_name.split('/')[-1]}_{timestamp}"
        
        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=model_save_path,
                report_to="none",
            ),
        )
        
        # 开始训练
        trainer_stats = trainer.train()
        
        # 保存模型
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        training_time = trainer_stats.metrics['train_runtime']
        training_time_min = round(training_time / 60, 2)
        
        return f"模型训练完成，用时 {training_time_min} 分钟，已保存到 {model_save_path}"
    except Exception as e:
        return f"模型训练失败: {str(e)}"

def inference_with_model(model_path, question, input_text=""):
    """使用训练好的模型进行推理"""
    try:
        # 加载模型和分词器
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        
        # 启用快速推理
        FastLanguageModel.for_inference(model)
        
        # 准备提示词
        alpaca_prompt = """下面是一个描述任务的指令，以及提供进一步上下文的输入。请写一个适当地完成请求的回应。

### 指令:
{}

### 输入:
{}

### 回应:
"""
        
        # 准备输入
        prompt = alpaca_prompt.format(question, input_text)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 生成回答
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True
        )
        
        # 解码输出
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # 提取回答部分
        response = decoded_output.split("### 回应:")[1].strip()
        
        return response
    except Exception as e:
        return f"推理失败: {str(e)}"

def save_qa_data_to_json(qa_data, file_path):
    """
    保存问答数据到JSON文件
    
    Args:
        qa_data (dict): 问答数据字典
        file_path (str): 输出文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 如果文件已存在，加载现有数据并合并
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # 合并新旧数据
            qa_data.update(existing_data)
        except json.JSONDecodeError:
            # 如果文件格式错误，则覆盖
            pass
    
    # 保存合并后的数据
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    
    return f"数据已保存到 {file_path}"

def load_qa_data(file_path):
    """
    加载QA数据文件
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        str: 格式化的QA数据
    """
    try:
        if not os.path.exists(file_path):
            return "文件不存在"
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 格式化显示
        formatted_data = json.dumps(data, ensure_ascii=False, indent=2)
        return formatted_data
    except Exception as e:
        return f"加载失败: {str(e)}"

def export_qa_data(qa_source, output_path):
    """
    导出QA数据到指定路径
    
    Args:
        qa_source (str): 数据来源名称 ("recommendation" 或 "test_report")
        output_path (str): 目标文件路径
    
    Returns:
        str: 操作结果
    """
    try:
        # 根据数据来源确定文件路径
        if qa_source == "recommendation":
            file_path = "temp/recommendation_qa_data.json"
        elif qa_source == "test_report":
            file_path = "temp/test_report_qa_data.json"
        else:
            return "无效的数据来源"
        
        if not os.path.exists(file_path):
            return "源文件不存在"
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 复制文件
        with open(file_path, "r", encoding="utf-8") as src:
            data = json.load(src)
        
        with open(output_path, "w", encoding="utf-8") as dst:
            json.dump(data, dst, ensure_ascii=False, indent=2)
        
        return f"数据已成功导出到 {output_path}"
    except Exception as e:
        return f"导出失败: {str(e)}"

def convert_qa_to_training_format(file_path, output_path=None):
    """
    将QA数据转换为训练所需的格式
    
    Args:
        file_path (str): 源文件路径
        output_path (str, optional): 输出文件路径，如果为None则自动生成
    
    Returns:
        str: 操作结果和输出路径
    """
    try:
        if not os.path.exists(file_path):
            return "源文件不存在"
        
        # 读取QA数据
        with open(file_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        # 转换为训练格式
        training_data = []
        for question, answer in qa_data.items():
            training_data.append({
                "instruction": question,
                "input": "",
                "output": answer
            })
        
        # 生成输出文件路径
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"export/training_data_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为训练格式
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return f"数据已转换为训练格式并保存到 {output_path}", output_path
    except Exception as e:
        return f"转换失败: {str(e)}", None

def merge_qa_datasets(file_paths, output_path=None):
    """
    合并多个QA数据集
    
    Args:
        file_paths (list): 源文件路径列表
        output_path (str, optional): 输出文件路径，如果为None则自动生成
    
    Returns:
        str: 操作结果
    """
    try:
        if not file_paths:
            return "未提供文件列表"
        
        # 读取并合并所有数据
        merged_data = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged_data.update(data)
        
        if not merged_data:
            return "没有有效数据可合并"
        
        # 生成输出文件路径
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"export/merged_qa_data_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存合并的数据
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        return f"已成功合并 {len(merged_data)} 条QA数据并保存到 {output_path}"
    except Exception as e:
        return f"合并失败: {str(e)}"

# Add a completely new QA Data Management tab function

def create_qa_data_management_tab():
    """
    创建QA数据管理标签页
    """
    with gr.Tab("📊 医疗数据管理"):
        gr.Markdown("### 检验数据知识库管理")
        gr.Markdown('<div class="medical-alert">本模块用于管理系统产生的医疗问答数据，可导出用于训练和改进AI模型</div>')
        
        with gr.Tabs():
            with gr.TabItem("检验项目推荐数据"):
                with gr.Row():
                    with gr.Column():
                        with gr.Box(elem_classes="block"):
                            gr.Markdown("#### 数据操作")
                            inspection_qa_view_btn = gr.Button("查看检验推荐数据", variant="primary", elem_classes="primary-btn")
                            inspection_qa_export_path = gr.Textbox(
                                label="导出路径", 
                                value="export/inspection_qa_data.json",
                                placeholder="输入导出文件路径"
                            )
                            with gr.Row():
                                inspection_qa_export_btn = gr.Button("导出数据", variant="secondary")
                                inspection_qa_convert_btn = gr.Button("转换为训练格式", variant="secondary")
                    
                    with gr.Column():
                        inspection_qa_data = gr.Textbox(label="数据内容", lines=20)
                        inspection_qa_status = gr.Textbox(label="操作状态", lines=1)
            
            with gr.TabItem("检验报告数据"):
                with gr.Row():
                    with gr.Column():
                        with gr.Box(elem_classes="block"):
                            gr.Markdown("#### 数据操作")
                            report_qa_view_btn = gr.Button("查看检验报告数据", variant="primary", elem_classes="primary-btn")
                            report_qa_export_path = gr.Textbox(
                                label="导出路径", 
                                value="export/report_qa_data.json",
                                placeholder="输入导出文件路径"
                            )
                            with gr.Row():
                                report_qa_export_btn = gr.Button("导出数据", variant="secondary")
                                report_qa_convert_btn = gr.Button("转换为训练格式", variant="secondary")
                    
                    with gr.Column():
                        report_qa_data = gr.Textbox(label="数据内容", lines=20)
                        report_qa_status = gr.Textbox(label="操作状态", lines=1)
            
            with gr.TabItem("数据集合并"):
                with gr.Row():
                    with gr.Column():
                        with gr.Box(elem_classes="block"):
                            gr.Markdown("#### 数据集合并配置")
                            merge_files_checkbox = gr.CheckboxGroup(
                                choices=[
                                    "temp/recommendation_qa_data.json", 
                                    "temp/test_report_qa_data.json"
                                ],
                                label="选择要合并的数据集",
                                value=["temp/recommendation_qa_data.json", "temp/test_report_qa_data.json"]
                            )
                            merge_output_path = gr.Textbox(
                                label="输出文件路径（可选）",
                                placeholder="留空将自动生成文件名",
                                value=""
                            )
                            with gr.Row():
                                merge_datasets_btn = gr.Button("合并数据集", variant="primary", elem_classes="primary-btn")
                                merge_convert_btn = gr.Button("合并并转换为训练格式", variant="secondary")
                    
                    with gr.Column():
                        with gr.Box(elem_classes="block"):
                            gr.Markdown("#### 合并结果")
                            merge_status = gr.Textbox(label="状态信息", lines=3)
                            merged_data_view = gr.Textbox(label="数据预览", lines=15)
        
        # 设置事件
        inspection_qa_view_btn.click(
            fn=lambda: load_qa_data("temp/recommendation_qa_data.json"),
            inputs=None,
            outputs=inspection_qa_data
        )
        
        def export_recommendation_data(output_path):
            """导出检验推荐QA数据"""
            return export_qa_data("recommendation", output_path)
        
        inspection_qa_export_btn.click(
            fn=export_recommendation_data,
            inputs=inspection_qa_export_path,
            outputs=inspection_qa_status
        )
        
        def convert_recommendation_data():
            """转换检验推荐QA数据为训练格式"""
            result, _ = convert_qa_to_training_format("temp/recommendation_qa_data.json")
            return result
        
        inspection_qa_convert_btn.click(
            fn=convert_recommendation_data,
            inputs=None,
            outputs=inspection_qa_status
        )
        
        report_qa_view_btn.click(
            fn=lambda: load_qa_data("temp/test_report_qa_data.json"),
            inputs=None,
            outputs=report_qa_data
        )
        
        def export_report_data(output_path):
            """导出检验报告QA数据"""
            return export_qa_data("test_report", output_path)
        
        report_qa_export_btn.click(
            fn=export_report_data,
            inputs=report_qa_export_path,
            outputs=report_qa_status
        )
        
        def convert_report_data():
            """转换检验报告QA数据为训练格式"""
            result, _ = convert_qa_to_training_format("temp/test_report_qa_data.json")
            return result
        
        report_qa_convert_btn.click(
            fn=convert_report_data,
            inputs=None,
            outputs=report_qa_status
        )
        
        merge_datasets_btn.click(
            fn=merge_qa_datasets,
            inputs=[merge_files_checkbox, merge_output_path],
            outputs=merge_status
        )
        
        def run_merge_and_convert(file_paths, output_path):
            """运行合并和转换操作"""
            if not file_paths:
                return "未提供文件列表", ""
            
            # 处理输出路径
            if not output_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                merged_path = f"export/merged_qa_data_{timestamp}.json"
            else:
                merged_path = output_path
            
            # 合并数据
            merge_result = merge_qa_datasets(file_paths, merged_path)
            
            # 转换为训练格式
            training_result, _ = convert_qa_to_training_format(merged_path)
            
            # 加载合并后的数据
            preview = load_qa_data(merged_path)
            
            return f"{merge_result}\n{training_result}", preview
        
        merge_convert_btn.click(
            fn=run_merge_and_convert,
            inputs=[merge_files_checkbox, merge_output_path],
            outputs=[merge_status, merged_data_view]
        )
        
        return {
            "inspection_qa_data": inspection_qa_data,
            "inspection_qa_status": inspection_qa_status,
            "report_qa_data": report_qa_data,
            "report_qa_status": report_qa_status,
            "merge_status": merge_status,
            "merged_data_view": merged_data_view
        }

# Replace the existing "10. QA数据管理" tab with a call to the function
# Find this section in the Gradio interface definition and replace it with:
    qa_data_components = create_qa_data_management_tab()

# 创建全局共享状态
shared_text_state = gr.State("")

# 创建Gradio界面
with gr.Blocks(title="医检AI平台", theme=MEDICAL_THEME, css=CUSTOM_CSS) as app:
    gr.Markdown("# 医院AIagent智能检验平台：基于camel多智能体框架和医检推理小模型")
    gr.Markdown('<div class="medical-success">欢迎使用医院智能检验平台 - 为临床医生和医技人员提供AI辅助诊断和检验分析工具</div>')
    
    # 创建共享状态
    conversation_text = gr.State("")
    
    with gr.Tab("📝 诊室对话记录"):
        gr.Markdown('<div class="medical-success">录制患者对话或上传录音，系统将自动转换为文本，方便诊断记录</div>')
        
        with gr.Row():
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 音频获取方式")
                    with gr.Tabs():
                        with gr.TabItem("实时录音"):
                            audio_recorder = gr.Audio(label="点击麦克风图标开始录音", source="microphone", type="filepath")
                            start_recording_btn = gr.Button("开始录音", variant="primary", elem_classes="primary-btn")
                            gr.Markdown('<div class="medical-alert">💡 提示：录音完成后自动转换，或点击处理音频按钮手动转换</div>')
                            
                        with gr.TabItem("上传音频"):
                            audio_input = gr.Audio(label="上传音频文件", type="filepath")
                    
                    with gr.Row():
                        audio_process_btn = gr.Button("处理音频", variant="primary", elem_classes="primary-btn")
                        clear_audio_btn = gr.Button("清除", variant="secondary")
            
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 转换结果")
                    text_output = gr.Textbox(label="转换结果", lines=15)
                    with gr.Row():
                        save_text_btn = gr.Button("保存转换结果", variant="secondary")
                        copy_to_recommend_btn = gr.Button("复制到检验推荐", variant="secondary")
        
        gr.Markdown("---")
        gr.Markdown("### 文本转语音")
        
        with gr.Row():
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 输入内容")
                    text_input = gr.Textbox(label="输入文本生成语音", lines=5, placeholder="输入需要转换为语音的医嘱或建议...")
                    text_to_speech_btn = gr.Button("生成语音", variant="primary", elem_classes="primary-btn")
            
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 合成语音")
                    audio_output = gr.Audio(label="生成的语音")
                    audio_status = gr.Textbox(label="状态")
        
        # 设置事件
        def process_recorded_or_uploaded_audio(audio_file_path):
            if audio_file_path:
                return process_audio(audio_file_path)
            else:
                return "请先录制或上传音频文件"
        
        def save_text_to_file(text):
            if not text:
                return "没有可保存的文本内容"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"temp/conversation_{timestamp}.txt"
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return f"文本已保存到 {save_path}"
            except Exception as e:
                return f"保存失败: {str(e)}"
        
        def clear_audio():
            return None, None, ""
                
        # 绑定录音和上传的处理函数
        audio_input.change(
            fn=process_recorded_or_uploaded_audio,
            inputs=audio_input,
            outputs=text_output
        )
        
        audio_recorder.change(
            fn=process_recorded_or_uploaded_audio,
            inputs=audio_recorder,
            outputs=text_output
        )
        
        # 添加录音按钮功能
        start_recording_btn.click(
            fn=lambda: None,  # 不做任何处理，只是提示用户点击麦克风图标
            inputs=[],
            outputs=[]
        )
        
        audio_process_btn.click(
            fn=lambda x, y: process_recorded_or_uploaded_audio(x or y),
            inputs=[audio_input, audio_recorder],
            outputs=text_output
        )
        
        clear_audio_btn.click(
            fn=clear_audio,
            inputs=[],
            outputs=[audio_input, audio_recorder, text_output]
        )
        
        save_text_btn.click(
            fn=save_text_to_file,
            inputs=text_output,
            outputs=audio_status
        )
        
        # 添加复制到检验推荐功能
        shared_text_state = gr.State("")
        
        def copy_to_recommendation(text, state):
            """将文本复制到共享状态，并通知用户切换选项卡"""
            if not text:
                return "没有可复制的文本", state
            return "文本已准备好，请切换到检验项目推荐选项卡", text
        
        copy_to_recommend_btn.click(
            fn=copy_to_recommendation,
            inputs=[text_output, conversation_text],
            outputs=[audio_status, conversation_text]
        )
        
        text_to_speech_btn.click(fn=text_to_audio, inputs=text_input, outputs=[audio_output, audio_status])
    
    with gr.Tab("🔬 检验项目推荐"):
        with gr.Row():
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 患者对话内容")
                    load_text_btn = gr.Button("加载转换的文本", variant="secondary")
                    conversation_input = gr.Textbox(
                        label="医患对话内容", 
                        lines=15,
                        elem_id="conversation_input"
                    )
                    recommend_btn = gr.Button("推荐检验项目", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 推荐检验项目")
                    recommendation_output = gr.Textbox(label="推荐结果", lines=20)
                    save_recommendation_btn = gr.Button("保存推荐结果", variant="secondary")
        
        # 设置事件
        load_text_btn.click(
            fn=lambda state: state if state else load_converted_text(),
            inputs=[conversation_text],
            outputs=conversation_input
        )
        
        # 从对话页面获取数据
        def load_shared_text(state):
            if state:
                return state
            return gr.update()
        
        # 定义选项卡切换时的处理函数
        def on_tab_change(tab_index):
            # 当切换到检验项目推荐页面(索引为1)时，检查是否有数据需要加载
            if tab_index == 1:  # 检验项目推荐页面的索引
                return load_shared_text(conversation_text.value)
            return gr.update()
        
        # 注册选项卡变更事件
        app.load(
            fn=load_shared_text,
            inputs=[conversation_text],
            outputs=conversation_input
        )
        
        recommend_btn.click(fn=recommend_inspection_items, inputs=conversation_input, outputs=recommendation_output)
        
        # 保存推荐结果
        def save_recommendation(text):
            if not text:
                return "没有可保存的推荐结果"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"temp/recommendation_{timestamp}.txt"
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return f"推荐结果已保存到 {save_path}"
            except Exception as e:
                return f"保存失败: {str(e)}"
        
        save_recommendation_btn.click(
            fn=save_recommendation,
            inputs=recommendation_output,
            outputs=gr.Textbox(elem_id="recommendation_status", visible=False) # 隐藏状态框
        )
    
    with gr.Tab("🔍 检验结果分析"):
        with gr.Row():
            with gr.Column():
                image_inputs = gr.File(label="上传检验结果图片", file_count="multiple")
                other_info = gr.Textbox(label="其他患者信息", lines=10)
                analyze_images_btn = gr.Button("分析图像", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                image_analysis_output = gr.Textbox(label="分析状态", lines=5)
        
        # 设置事件
        analyze_images_btn.click(fn=analyze_test_images, inputs=[image_inputs, other_info], outputs=image_analysis_output)
    
    with gr.Tab("📋 检验报告生成"):
        with gr.Row():
            with gr.Column():
                generate_report_btn = gr.Button("生成检验报告", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                report_output = gr.Textbox(label="生成的报告", lines=25)
        
        # 设置事件
        generate_report_btn.click(fn=generate_test_report, inputs=None, outputs=report_output)
    
    with gr.Tab("📊 数据分析与研究"):
        with gr.Row():
            with gr.Column():
                csv_input = gr.File(label="上传CSV数据文件（可选）", file_types=[".csv"])
                analysis_requirements = gr.Textbox(
                    label="分析需求", 
                    lines=10,
                    value=data_analysis.example_user_info
                )
                analyze_data_btn = gr.Button("分析数据", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                data_analysis_output = gr.Textbox(label="分析结果", lines=25)
        
        # 设置事件
        analyze_data_btn.click(fn=analyze_csv_data, inputs=[csv_input, analysis_requirements], outputs=data_analysis_output)
    
    with gr.Tab("🧪 临床数据合成"):
        with gr.Row():
            with gr.Column():
                main_topic = gr.Textbox(label="主题", value="肝功能检验数据")
                with gr.Row():
                    total_examples = gr.Slider(label="生成示例数量", minimum=5, maximum=50, value=10, step=5)
                    num_subtopics = gr.Slider(label="子主题数量", minimum=1, maximum=10, value=5, step=1)
                temperature = gr.Slider(label="模型温度", minimum=0.1, maximum=1.0, value=0.2, step=0.1)
                generate_data_btn = gr.Button("生成数据", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                synthetic_data_output = gr.Textbox(label="生成的数据", lines=25)
        
        # 设置事件
        generate_data_btn.click(
            fn=generate_synthetic_data, 
            inputs=[main_topic, total_examples, num_subtopics, temperature], 
            outputs=synthetic_data_output
        )
    
    with gr.Tab("💬 病例对话生成"):
        with gr.Row():
            with gr.Column():
                patient_history_input = gr.Textbox(
                    label="输入患者病历", 
                    lines=15,
                    placeholder="请在此输入患者病历信息，系统将根据病历生成医患对话..."
                )
                generate_dialogue_btn = gr.Button("生成医患对话", variant="primary", elem_classes="primary-btn")
            with gr.Column():
                dialogue_output = gr.Textbox(label="生成的医患对话", lines=25)
        
        # 设置事件
        generate_dialogue_btn.click(
            fn=generate_doctor_patient_dialogue, 
            inputs=patient_history_input, 
            outputs=dialogue_output
        )
    
    with gr.Tab("⚙️ 系统性能评测"):
        gr.Markdown("### 系统性能评测")
        gr.Markdown('<div class="medical-alert">本模块用于评估AI系统各组件性能，确保满足医疗应用质量标准</div>')
        
        with gr.Row():
            with gr.Column():
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 评测配置")
                    benchmark_components = gr.CheckboxGroup(
                        choices=["语音转文本", "检验项目推荐", "图像分析", "检验报告生成", "数据分析", "合成数据生成", "医患对话生成"],
                        label="选择要测试的组件",
                        value=["语音转文本", "检验项目推荐", "图像分析", "医患对话生成"]
                    )
                    num_test_samples = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        value=3, 
                        step=1,
                        label="测试样本数量"
                    )
                    with gr.Row():
                        benchmark_btn = gr.Button("运行性能评测", variant="primary", elem_classes="primary-btn")
                        download_report_btn = gr.Button("下载完整HTML报告", variant="secondary")
            
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("表格结果"):
                        benchmark_table = gr.Dataframe(
                            headers=["模块", "处理时间(秒)", "内存使用(MB)", "成功率(%)", "质量评分"],
                            label="性能评测结果"
                        )
                    with gr.TabItem("图表结果"):
                        benchmark_plot = gr.Image(label="性能评测图表")
                
                with gr.Box(elem_classes="block"):
                    gr.Markdown("#### 评测说明")
                    benchmark_notes = gr.Markdown("""
                    **性能指标解读:**
                    
                    - **处理时间**: 每个模块处理单个样本的平均时间（秒）
                    - **内存使用**: 估计的内存消耗（MB）
                    - **成功率**: 成功处理样本的百分比
                    - **质量评分**: 输出结果的质量评分（0-100）
                    
                    性能评测有助于识别系统瓶颈，优化临床应用效率。完整评测报告包含详细分析和改进建议。
                    """)
                
                benchmark_report_path = gr.Textbox(
                    label="HTML报告路径", 
                    value="./temp/benchmark_results/benchmark_report.html",
                    visible=False
                )
        
        # 设置事件
        benchmark_btn.click(
            fn=run_benchmark_from_gradio, 
            inputs=[benchmark_components, num_test_samples], 
            outputs=[benchmark_table, benchmark_plot]
        )
        
        def open_html_report(report_path):
            import webbrowser
            try:
                webbrowser.open(report_path)
                return "已在浏览器中打开报告"
            except Exception as e:
                return f"打开报告失败: {str(e)}"
        
        download_report_btn.click(
            fn=open_html_report,
            inputs=benchmark_report_path,
            outputs=benchmark_notes
        )
    
    with gr.Tab("🤖 模型微调训练"):
        gr.Markdown("### 医疗AI模型训练中心")
        gr.Markdown('<div class="medical-alert">本模块用于训练和微调医疗专用AI模型，为临床检验和诊断提供更精确的支持</div>')
        
        with gr.Accordion("医疗思维链数据生成", open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### API配置")
                        openai_api_key = gr.Textbox(label="API密钥", type="password")
                        model_choice = gr.Dropdown(
                            choices=["gpt-4.1", "gpt-4o-mini", "gpt-4o"], 
                            label="选择基础模型", 
                            value="gpt-3.5-turbo"
                        )
                        system_message = gr.Textbox(
                            label="专业提示词", 
                            value="你是一位医学专家，擅长进行深入思考并给出详细的医学分析", 
                            lines=2
                        )
                        create_agent_btn = gr.Button("创建智能体", variant="primary", elem_classes="primary-btn")
                        agent_status = gr.Textbox(label="状态", lines=1)
            
            gr.Markdown("---")
            gr.Markdown("### 医疗问答数据")
            
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        qa_data = gr.Textbox(
                            label="临床问答数据（JSON格式）", 
                            lines=10,
                            placeholder="""{"高血压有哪些治疗方法？": "高血压是一种常见慢性疾病...", "糖尿病的早期症状有哪些？": "糖尿病的早期症状包括..."}"""
                        )
                        qa_data_file = gr.File(label="或上传JSON文件", file_types=[".json"])
                        generate_cot_btn = gr.Button("生成思维链数据", variant="primary", elem_classes="primary-btn")
                
                with gr.Column():
                    cot_output = gr.Textbox(label="生成结果", lines=15)
                    alpaca_file_path = gr.Textbox(label="训练格式文件路径", visible=False)
            
            gr.Markdown("---")
            gr.Markdown("### 上传到模型库")
            
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        hf_token = gr.Textbox(label="访问令牌", type="password")
                        hf_username = gr.Textbox(label="用户名")
                        hf_dataset_name = gr.Textbox(label="数据集名称（可选）")
                        upload_hf_btn = gr.Button("上传到数据库", variant="primary", elem_classes="primary-btn")
                
                with gr.Column():
                    upload_status = gr.Textbox(label="上传状态", lines=3)
        
        with gr.Accordion("专业模型训练", open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 训练配置")
                        sft_model_name = gr.Dropdown(
                            choices=[
                                "unsloth/Qwen2.5-0.5B", 
                                "unsloth/Qwen2.5-1.5B", 
                                "unsloth/Qwen2.5-3B",
                                "unsloth/Qwen2.5-7B",
                                "unsloth/Qwen2.5-14B"
                            ],
                            label="选择基础模型",
                            value="unsloth/Qwen2.5-1.5B"
                        )
                        with gr.Row():
                            training_epochs = gr.Slider(label="训练轮次", minimum=1, maximum=10, value=3, step=1)
                            learning_rate = gr.Slider(label="学习率", minimum=1e-5, maximum=5e-4, value=2e-4, step=1e-5)
                        train_btn = gr.Button("开始训练", variant="primary", elem_classes="primary-btn")
                
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 训练进度")
                        training_status = gr.Textbox(label="训练状态", lines=5)
        
        with gr.Accordion("临床问答测试", open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 模型测试")
                        trained_models = gr.Dropdown(
                            label="选择训练好的模型",
                            choices=[],  # 初始为空列表
                            value=None
                        )
                        inference_question = gr.Textbox(
                            label="临床问题", 
                            lines=3,
                            placeholder="请输入一个医疗相关的问题..."
                        )
                        inference_input = gr.Textbox(
                            label="补充信息（可选）", 
                            lines=2,
                            placeholder="可以提供额外的临床数据和病史..."
                        )
                        with gr.Row():
                            inference_btn = gr.Button("获取回答", variant="primary", elem_classes="primary-btn")
                            refresh_models_btn = gr.Button("刷新模型列表", variant="secondary")
                
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### AI回复")
                        inference_output = gr.Textbox(label="临床解答", lines=10)
        
        with gr.Accordion("医疗数据训练", open=True):
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 数据选择")
                        qa_data_source = gr.Dropdown(
                            choices=[
                                "temp/recommendation_qa_data.json",
                                "temp/test_report_qa_data.json",
                                "使用合并数据集"
                            ],
                            label="选择临床数据源",
                            value="使用合并数据集"
                        )
                        with gr.Row():
                            qa_data_refresh_btn = gr.Button("刷新可用数据集", variant="secondary")
                            qa_data_preview_btn = gr.Button("预览数据", variant="secondary")
                    
                with gr.Column():
                    qa_data_preview = gr.Textbox(label="数据预览", lines=10)
                    
            gr.Markdown("---")
            gr.Markdown("### 基于临床数据训练模型")
            
            with gr.Row():
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 模型配置")
                        qa_sft_model_name = gr.Dropdown(
                            choices=[
                                "unsloth/Qwen2.5-0.5B", 
                                "unsloth/Qwen2.5-1.5B", 
                                "unsloth/Qwen2.5-3B",
                                "unsloth/Qwen2.5-7B",
                                "unsloth/Qwen2.5-14B"
                            ],
                            label="选择基础模型",
                            value="unsloth/Qwen2.5-1.5B"
                        )
                        with gr.Row():
                            qa_training_epochs = gr.Slider(label="训练轮次", minimum=1, maximum=10, value=3, step=1)
                            qa_learning_rate = gr.Slider(label="学习率", minimum=1e-5, maximum=5e-4, value=2e-4, step=1e-5)
                        qa_train_btn = gr.Button("开始训练", variant="primary", elem_classes="primary-btn")
                
                with gr.Column():
                    with gr.Box(elem_classes="block"):
                        gr.Markdown("#### 训练进度")
                        qa_training_status = gr.Textbox(label="训练状态", lines=5)
        
        # 存储Agent的状态
        chat_agent_state = gr.State(value=None)
        
        # 设置CoT数据生成与训练相关的事件
        create_agent_btn.click(
            fn=create_chat_agent,
            inputs=[openai_api_key, model_choice, system_message],
            outputs=[chat_agent_state, agent_status]
        )
        
        def load_qa_file(file):
            if file is None:
                return None
            with open(file.name, "r", encoding="utf-8") as f:
                return f.read()
        
        qa_data_file.change(
            fn=load_qa_file,
            inputs=qa_data_file,
            outputs=qa_data
        )
        
        generate_cot_btn.click(
            fn=generate_cot_data,
            inputs=[chat_agent_state, qa_data],
            outputs=[cot_output, alpaca_file_path]
        )
        
        upload_hf_btn.click(
            fn=upload_dataset_to_huggingface,
            inputs=[alpaca_file_path, hf_token, hf_username, hf_dataset_name],
            outputs=upload_status
        )
        
        train_btn.click(
            fn=train_model,
            inputs=[sft_model_name, alpaca_file_path, training_epochs, learning_rate],
            outputs=training_status
        )
        
        # 更新模型列表的函数
        def update_model_list():
            if os.path.exists("temp/trained_models"):
                return [d for d in os.listdir("temp/trained_models") if os.path.isdir(os.path.join("temp/trained_models", d))]
            return []
        
        # 添加刷新模型列表的事件
        refresh_models_btn.click(
            fn=update_model_list,
            inputs=[],
            outputs=trained_models
        )
        
        # 在训练完成后自动刷新模型列表
        train_btn.click(
            fn=update_model_list,
            inputs=[], 
            outputs=trained_models
        )
        
        inference_btn.click(
            fn=lambda model, q, i: inference_with_model(f"temp/trained_models/{model}", q, i) if model else "请先选择一个模型",
            inputs=[trained_models, inference_question, inference_input],
            outputs=inference_output
        )
        
        # Define function to get available QA datasets
        def get_available_qa_datasets():
            """获取可用的QA数据集列表"""
            datasets = []
            
            # 检查基本数据集
            basic_datasets = [
                "temp/recommendation_qa_data.json",
                "temp/test_report_qa_data.json"
            ]
            
            for dataset in basic_datasets:
                if os.path.exists(dataset):
                    datasets.append(dataset)
            
            # 检查导出目录中的合并数据集
            if os.path.exists("export"):
                for file in os.listdir("export"):
                    if file.startswith("merged_qa_data_") and file.endswith(".json"):
                        datasets.append(f"export/{file}")
            
            # 添加一个固定选项
            datasets.append("使用合并数据集")
            
            return datasets
        
        # Now add event handlers for the new buttons at the end of the model training tab section
        qa_data_refresh_btn.click(
            fn=get_available_qa_datasets,
            inputs=None,
            outputs=qa_data_source
        )
        
        qa_data_preview_btn.click(
            fn=lambda source: load_qa_data(source) if source != "使用合并数据集" else "请选择具体的数据集",
            inputs=qa_data_source,
            outputs=qa_data_preview
        )
        
        # Function to prepare data and start training
        def prepare_qa_data_and_train(data_source, model_name, epochs, learning_rate):
            """准备QA数据并开始训练"""
            try:
                # 如果选择了"使用合并数据集"，则创建一个新的合并数据集
                if data_source == "使用合并数据集":
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    merged_path = f"export/merged_qa_data_{timestamp}.json"
                    
                    # 合并可用的基础数据集
                    basic_datasets = []
                    if os.path.exists("temp/recommendation_qa_data.json"):
                        basic_datasets.append("temp/recommendation_qa_data.json")
                    if os.path.exists("temp/test_report_qa_data.json"):
                        basic_datasets.append("temp/test_report_qa_data.json")
                    
                    if not basic_datasets:
                        return "没有找到可用的基础数据集进行合并"
                    
                    # 执行合并
                    merge_result = merge_qa_datasets(basic_datasets, merged_path)
                    data_source = merged_path
                
                # 转换为训练格式
                _, training_file = convert_qa_to_training_format(data_source)
                
                if not training_file:
                    return "数据转换失败"
                
                # 开始训练
                training_result = train_model(model_name, training_file, epochs, learning_rate)
                
                return f"已使用QA数据开始训练\n{training_result}"
            except Exception as e:
                return f"准备训练数据失败: {str(e)}"
        
        # Add the training event handler
        qa_train_btn.click(
            fn=prepare_qa_data_and_train,
            inputs=[qa_data_source, qa_sft_model_name, qa_training_epochs, qa_learning_rate],
            outputs=qa_training_status
        )
    
    # Replace the existing "10. QA数据管理" tab with a call to the function
    qa_data_components = create_qa_data_management_tab()
    
    gr.Markdown(
        """
        <div class="footer">
        © 2023-2024 医院智能检验平台 | 由先进多智能体系统框架驱动 | 支持医疗诊断、检验项目推荐、结果分析、报告生成及科研数据管理
        <br><small>医疗器械备案号：XXXXX-XXXXXXX | 技术支持：医学AI部门</small>
        </div>
        """
    )

# 启动应用
if __name__ == "__main__":
    app.launch(share=True) 