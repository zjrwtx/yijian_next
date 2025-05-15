import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import List, Dict, Any, Tuple, Callable

# 导入系统模块
from audio_to_text import audio_models
from recommed_inspect_item import process_clinical_case
from image_analysis import analyze_images
from Analysis_test_results import analyze_test_results
import data_auto_anaylse_with_research_inspiration as data_analysis
from synthetic_data_pipeline import SyntheticDataPipeline
from doctor_to_patient_data import simulate_doctor_patient_dialogue

class BenchmarkTest:
    """用于医疗数据分析平台的综合基准测试系统"""
    
    def __init__(self, test_data_path="./benchmark_data"):
        """初始化基准测试系统"""
        self.test_data_path = test_data_path
        self.results_path = "./temp/benchmark_results"
        self.ensure_directories()
        
        # 加载或生成测试样本
        self.test_samples = self.load_test_samples()
        
        # 测试模块映射
        self.module_mapping = {
            "语音转文本": self.test_audio_to_text,
            "检验项目推荐": self.test_recommendation,
            "图像分析": self.test_image_analysis,
            "检验报告生成": self.test_report_generation,
            "数据分析": self.test_data_analysis,
            "合成数据生成": self.test_synthetic_data,
            "医患对话生成": self.test_dialogue_generation
        }
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.test_data_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
    
    def load_test_samples(self) -> Dict[str, List[Any]]:
        """加载或生成测试样本"""
        samples = {}
        
        # 语音测试样本
        samples["audio"] = self.get_audio_samples()
        
        # 文本测试样本
        samples["text"] = self.get_text_samples()
        
        # 图像测试样本
        samples["image"] = self.get_image_samples()
        
        # 患者病历样本
        samples["patient_histories"] = samples["text"]  # 复用文本样本
        
        # 检验数据分析样本
        samples["analysis_requirements"] = self.get_analysis_samples()
        
        return samples
    
    def get_audio_samples(self) -> List[str]:
        """获取音频测试样本"""
        # 使用现有的音频文件或生成新的
        audio_files = []
        
        if os.path.exists("temp/output_audio.mp3"):
            audio_files = ["temp/output_audio.mp3"] * 5
        else:
            # 尝试创建测试音频文件
            test_text = "这是一个用于基准测试的音频样本。请记录测试结果。"
            try:
                audio_file = f"{self.test_data_path}/test_audio.mp3"
                audio_models.text_to_speech(input=test_text, storage_path=audio_file)
                audio_files = [audio_file] * 5
            except:
                # 使用示例文件路径
                audio_files = ["./examples/fish_audio_models/example_audio.mp3"] * 5
        
        return audio_files
    
    def get_text_samples(self) -> List[str]:
        """获取文本测试样本"""
        return [
            "患者男性，45岁，主诉胸闷气短3个月，加重1周。查体：BP 160/95mmHg，心率88次/分。血常规示白细胞轻度升高，心电图示V1-V3导联ST段抬高。",
            "患者女性，35岁，主诉关节疼痛伴晨僵2年，加重1个月。查体：双手近端指间关节肿胀。抗CCP抗体阳性，风湿因子阳性。",
            "患者男性，60岁，主诉尿频尿急尿痛5天。查体：下腹部压痛，肾区叩击痛阴性。尿常规示白细胞+++，亚硝酸盐阳性。",
            "患者女性，28岁，主诉发热、咳嗽、咳痰3天。查体：体温38.5℃，双肺可闻及湿啰音。胸部CT示双肺下叶斑片状阴影。",
            "患者男性，50岁，主诉上腹痛、恶心、呕吐2天。查体：上腹部压痛，Murphy征阳性。血淀粉酶升高，B超示胆囊壁增厚。"
        ]
    
    def get_image_samples(self) -> List[List[str]]:
        """获取图像测试样本"""
        # 使用系统中的现有图像
        return [["i01.png", "i03.png"]] * 5
    
    def get_analysis_samples(self) -> List[str]:
        """获取数据分析需求样本"""
        return [data_analysis.example_user_info] * 5
    
    def test_audio_to_text(self, sample: str) -> Dict[str, Any]:
        """测试语音转文本功能"""
        # 检查文件是否存在
        if not os.path.exists(sample):
            return {"status": "failed", "error": "音频文件不存在"}
        
        # 尝试进行语音转换
        try:
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行转换
            start_time = time.time()
            result = audio_models.speech_to_text(audio_file_path=sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            # 评估结果质量 (简单长度检查)
            quality_score = min(100, len(result) / 10)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_recommendation(self, sample: str) -> Dict[str, Any]:
        """测试检验项目推荐功能"""
        try:
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行推荐
            start_time = time.time()
            result = process_clinical_case(sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            quality_score = min(100, len(result) / 20)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_image_analysis(self, sample: List[str]) -> Dict[str, Any]:
        """测试图像分析功能"""
        try:
            # 检查文件是否存在
            for img_path in sample:
                if not os.path.exists(img_path):
                    return {"status": "failed", "error": f"图像文件 {img_path} 不存在"}
            
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行图像分析
            start_time = time.time()
            result = analyze_images(sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            quality_score = min(100, len(result) / 15)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_report_generation(self, sample: str) -> Dict[str, Any]:
        """测试检验报告生成功能"""
        try:
            # 确保有图像分析结果
            with open("temp/image_analysis_result.txt", "w", encoding="utf-8") as f:
                f.write(f"图片分析结果：\n模拟的图像分析结果\n\n其他患者信息：\n{sample}")
            
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行报告生成
            start_time = time.time()
            result = analyze_test_results(sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            quality_score = min(100, len(result) / 30)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_data_analysis(self, sample: str) -> Dict[str, Any]:
        """测试数据分析功能"""
        try:
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行数据分析
            start_time = time.time()
            result = data_analysis.process_lab_data_analysis(user_info=sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            quality_score = min(100, len(result) / 50)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_synthetic_data(self, sample: str) -> Dict[str, Any]:
        """测试合成数据生成功能"""
        try:
            # 创建合成数据生成器
            pipeline = SyntheticDataPipeline(temperature=0.2)
            
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行数据生成
            start_time = time.time()
            data = pipeline.generate_synthetic_data(
                main_topic="肝功能检验数据",
                total_examples=3,
                num_subtopics=2
            )
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            result_str = json.dumps(data, ensure_ascii=False)
            quality_score = min(100, len(data) * 10)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result_str)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def test_dialogue_generation(self, sample: str) -> Dict[str, Any]:
        """测试医患对话生成功能"""
        try:
            # 记录内存使用
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # 执行对话生成
            start_time = time.time()
            result = simulate_doctor_patient_dialogue(sample)
            processing_time = time.time() - start_time
            
            # 再次记录内存
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # 评估结果质量
            quality_score = min(100, len(result) / 20)
            
            return {
                "status": "success", 
                "processing_time": processing_time,
                "memory_used": mem_used,
                "quality_score": quality_score,
                "result_length": len(result)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def run_benchmark(self, modules: List[str], num_samples: int) -> pd.DataFrame:
        """运行基准测试"""
        results = {
            "模块": [],
            "处理时间(秒)": [],
            "内存使用(MB)": [],
            "成功率(%)": [],
            "质量评分": []
        }
        
        detailed_results = {}
        
        for module_name in modules:
            if module_name not in self.module_mapping:
                print(f"警告: 未找到模块 {module_name}")
                continue
                
            print(f"测试模块: {module_name}")
            test_func = self.module_mapping[module_name]
            sample_type = self._get_sample_type_for_module(module_name)
            
            if sample_type not in self.test_samples:
                print(f"警告: 未找到模块 {module_name} 的测试样本")
                continue
                
            samples = self.test_samples[sample_type][:num_samples]
            module_results = []
            
            total_time = 0
            total_memory = 0
            success_count = 0
            total_quality = 0
            
            for i, sample in enumerate(samples):
                print(f"  - 样本 {i+1}/{num_samples}")
                result = test_func(sample)
                module_results.append(result)
                
                if result.get("status") == "success":
                    success_count += 1
                    total_time += result.get("processing_time", 0)
                    total_memory += result.get("memory_used", 0)
                    total_quality += result.get("quality_score", 0)
                else:
                    print(f"    失败: {result.get('error', '未知错误')}")
            
            # 计算平均值
            if success_count > 0:
                avg_time = total_time / success_count
                avg_memory = total_memory / success_count
                avg_quality = total_quality / success_count
            else:
                avg_time = 0
                avg_memory = 0
                avg_quality = 0
                
            success_rate = (success_count / num_samples) * 100 if num_samples > 0 else 0
            
            # 添加结果
            results["模块"].append(module_name)
            results["处理时间(秒)"].append(round(avg_time, 3))
            results["内存使用(MB)"].append(round(avg_memory, 2))
            results["成功率(%)"].append(round(success_rate, 1))
            results["质量评分"].append(round(avg_quality, 1))
            
            detailed_results[module_name] = module_results
        
        # 保存详细结果
        with open(f"{self.results_path}/detailed_results.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.results_path}/summary_results.csv", index=False)
        
        return results_df
    
    def _get_sample_type_for_module(self, module_name: str) -> str:
        """为模块确定样本类型"""
        sample_mapping = {
            "语音转文本": "audio",
            "检验项目推荐": "text",
            "图像分析": "image",
            "检验报告生成": "text",
            "数据分析": "analysis_requirements",
            "合成数据生成": "text",
            "医患对话生成": "patient_histories"
        }
        return sample_mapping.get(module_name, "text")
    
    def generate_plots(self, df: pd.DataFrame) -> str:
        """生成性能图表"""
        # 创建性能图表
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 处理时间图表
        df.plot(x="模块", y="处理时间(秒)", kind="bar", ax=axs[0, 0], color="blue", rot=45)
        axs[0, 0].set_title("平均处理时间 (秒)")
        axs[0, 0].set_ylabel("秒")
        
        # 内存使用图表
        df.plot(x="模块", y="内存使用(MB)", kind="bar", ax=axs[0, 1], color="green", rot=45)
        axs[0, 1].set_title("平均内存使用 (MB)")
        axs[0, 1].set_ylabel("MB")
        
        # 成功率图表
        df.plot(x="模块", y="成功率(%)", kind="bar", ax=axs[1, 0], color="orange", rot=45)
        axs[1, 0].set_title("成功率 (%)")
        axs[1, 0].set_ylabel("%")
        
        # 质量评分图表
        df.plot(x="模块", y="质量评分", kind="bar", ax=axs[1, 1], color="purple", rot=45)
        axs[1, 1].set_title("质量评分 (0-100)")
        axs[1, 1].set_ylabel("分数")
        
        plt.tight_layout()
        
        # 将图表保存为文件
        plot_path = f"{self.results_path}/benchmark_plot.png"
        plt.savefig(plot_path, dpi=300)
        
        # 将图表转换为base64编码的图像
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
    
    def run_full_benchmark(self, modules: List[str], num_samples: int) -> Tuple[pd.DataFrame, str]:
        """运行完整的基准测试并生成报告"""
        # 运行基准测试
        results_df = self.run_benchmark(modules, num_samples)
        
        # 生成图表
        img_data = self.generate_plots(results_df)
        
        # 生成HTML报告
        self.generate_html_report(results_df, img_data)
        
        return results_df, img_data
    
    def generate_html_report(self, df: pd.DataFrame, img_data: str):
        """生成HTML格式的基准测试报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>医疗数据分析平台 - 性能评测报告</title>
            <style>
                body {{
                    font-family: "Google Sans", Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #1a73e8;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .chart {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-size: 14px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>医疗数据分析平台 - 性能评测报告</h1>
            <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>性能评测摘要</h2>
                <p>本报告提供了医疗数据分析平台各模块的性能评测结果，包括处理时间、内存使用、成功率和质量评分。</p>
                <p>测试模块: {", ".join(df["模块"].tolist())}</p>
                <p>每个模块的样本数: {len(df) if not df.empty else 0}</p>
            </div>
            
            <h2>评测结果表格</h2>
            <table>
                <tr>
                    <th>模块</th>
                    <th>处理时间(秒)</th>
                    <th>内存使用(MB)</th>
                    <th>成功率(%)</th>
                    <th>质量评分</th>
                </tr>
                {"".join([f"<tr><td>{row['模块']}</td><td>{row['处理时间(秒)']}</td><td>{row['内存使用(MB)']}</td><td>{row['成功率(%)']}</td><td>{row['质量评分']}</td></tr>" for _, row in df.iterrows()])}
            </table>
            
            <h2>性能图表</h2>
            <div class="chart">
                <img src="{img_data}" alt="性能评测图表" style="max-width:100%;">
            </div>
            
            <h2>结论和建议</h2>
            <p>基于上述性能评测结果，提出以下优化建议：</p>
            <ul>
                {"".join([self._generate_recommendation(row) for _, row in df.iterrows()])}
            </ul>
            
            <div class="footer">
                © 医疗数据分析平台 - 性能评测系统
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        with open(f"{self.results_path}/benchmark_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _generate_recommendation(self, row: pd.Series) -> str:
        """根据性能结果生成优化建议"""
        module = row["模块"]
        recommendations = []
        
        # 处理时间建议
        if row["处理时间(秒)"] > 5:
            recommendations.append(f"考虑优化 {module} 模块的处理速度，当前平均处理时间较长")
        
        # 内存使用建议
        if row["内存使用(MB)"] > 500:
            recommendations.append(f"考虑优化 {module} 模块的内存使用，当前内存消耗较高")
        
        # 成功率建议
        if row["成功率(%)"] < 80:
            recommendations.append(f"提高 {module} 模块的稳定性，当前成功率较低")
        
        # 质量评分建议
        if row["质量评分"] < 70:
            recommendations.append(f"提高 {module} 模块的输出质量，当前质量评分较低")
        
        if not recommendations:
            return f"<li><strong>{module}:</strong> 性能良好，无需立即优化。</li>"
        
        return f"<li><strong>{module}:</strong> {' '.join(recommendations)}。</li>"

def run_benchmark_from_gradio(modules_to_test: List[str], num_samples: int) -> Tuple[pd.DataFrame, str]:
    """从Gradio界面调用的基准测试函数"""
    benchmark = BenchmarkTest()
    return benchmark.run_full_benchmark(modules_to_test, num_samples)

if __name__ == "__main__":
    # 执行所有模块的基准测试
    benchmark = BenchmarkTest()
    all_modules = list(benchmark.module_mapping.keys())
    results, img_data = benchmark.run_full_benchmark(all_modules, 3)
    
    print("\n基准测试完成！")
    print(f"报告已保存到 {benchmark.results_path}/benchmark_report.html")
    print("\n测试结果摘要:")
    print(results) 