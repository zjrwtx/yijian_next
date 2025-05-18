from openai import OpenAI
import json
import concurrent.futures
import time # 用于计时和简单的进度反馈

# --- OpenAI Client Setup ---
# 请确保你使用的是正确的凭证和API端点
# 对于ModelScope，API Key通常放在HTTP Header的Authorization中，格式为 "Bearer YOUR_TOKEN"
# 但你提供的代码直接使用了 api_key 参数，这可能特定于某个封装库或ModelScope的特定配置。
# 以下配置基于你提供的代码。
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='ab86248c-8b98-4c51-981c-c23f923a101a', # 重要：替换为你的真实ModelScope Token
                                     # 你提供的 'ab86248c-8b98-4c51-981c-c23f923a101a' 可能是示例或已失效
)

# set extra_body for thinking control
extra_body = {
    # enable thinking, set to False to disable
    "enable_thinking": True,
    # use thinking_budget to contorl num of tokens used for thinking
    "thinking_budget": 38912,
    
}

def get_ai_response(patient_note, ground_truth):
    """
    Sends a request to the AI model and returns the complete response.
    Print statements are removed to avoid interleaved output in multithreading.
    """
    try:
        response = client.chat.completions.create(
            model='Qwen/Qwen3-235B-A22B',  # ModelScope Model-Id
            messages=[
                {
                    "role": "user",
                    "content": f"你是一个严谨且富有经验的临床医生，请你根据患者信息{patient_note}提出你怀疑的疾病可能有哪些，且为了做出鉴别诊断，应该做什么检验项目？最后的结果要与答案相符{ground_truth}"
                }
            ],
            stream=True,  # 流式输出仍然有用，因为它可以更快地开始接收数据
            extra_body=extra_body,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
        
        
          
           
            
      
        )
         
           
        

        all_thinking = ""
        all_answer = ""

        for chunk in response:
            thinking_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            if thinking_chunk: # Pythonic way to check for non-empty string
                all_thinking += thinking_chunk
            elif answer_chunk:
                all_answer += answer_chunk
        
        final_answer = all_thinking + ("\n\n === Final Answer ===\n" + all_answer if all_answer else "")
        return final_answer

    except Exception as e:
        print(f"API call failed: {e}") # Log API error to console
        return f"Error during AI processing: {str(e)}"


def process_item(item_data):
    """
    Worker function to process a single item from the jsonl file.
    'item_data' is a dictionary loaded from a line of the jsonl file.
    """
    patient_note = item_data['patient_note']
    ground_truth = item_data['ground_truth']
    
    # Get AI response
    ai_response = get_ai_response(patient_note, ground_truth)
    
    # Add rationale to the data
    item_data['rationale'] = ai_response
    
    return item_data


def process_jsonl_threaded(input_file, output_file, max_workers=10):
    """
    Processes a .jsonl file using multiple threads for API calls.
    """
    items_to_process = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                items_to_process.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                continue
    
    if not items_to_process:
        print("No valid items to process.")
        return

    total_items = len(items_to_process)
    print(f"Starting processing for {total_items} items with {max_workers} workers...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Using ThreadPoolExecutor to manage a pool of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use executor.map to apply process_item to each item.
            # map returns results in the order that tasks were submitted.
            # It also handles exceptions from worker functions.
            results_iterator = executor.map(process_item, items_to_process)
            
            for i, result_data in enumerate(results_iterator):
                if result_data: # Ensure result_data is not None (though process_item should always return a dict)
                    f_out.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                if (i + 1) % 10 == 0 or (i + 1) == total_items: # Print progress every 10 items or at the end
                    print(f"Processed and wrote item {i+1}/{total_items}")

    print(f"Processing complete. Output written to {output_file}")


if __name__ == "__main__":
    input_file = "test.jsonl"  # 替换为你的输入文件路径
    output_file = "output_threaded.jsonl"  # 替换为你期望的输出文件路径
    
    # 创建一个示例 test.jsonl 文件 (如果它不存在)
    # 在实际使用中，你应该有自己的 test.jsonl 文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            pass
        print(f"Using existing input file: {input_file}")
    except FileNotFoundError:
        print(f"Creating a dummy '{input_file}' for testing purposes.")
        dummy_data = [
            {"patient_note": "患者主诉发热三天，体温38.5℃，伴有咳嗽、咳痰。", "ground_truth": "考虑为上呼吸道感染，建议血常规检查。"},
            {"patient_note": "患者腹痛一天，位于右下腹，按压痛明显。", "ground_truth": "高度怀疑急性阑尾炎，建议腹部超声检查。"},
            {"patient_note": "患者头晕，视物旋转，伴恶心呕吐。", "ground_truth": "可能为梅尼埃病或良性阵发性位置性眩晕，建议前庭功能检查。"},
            {"patient_note": "患者胸闷，心悸，活动后加重。", "ground_truth": "需排除心脏疾病，建议心电图、心肌酶谱检查。"},
            {"patient_note": "儿童，发热、皮疹，口腔内可见柯氏斑。", "ground_truth": "高度怀疑麻疹，注意隔离，对症支持治疗。"}
        ]
        with open(input_file, 'w', encoding='utf-8') as f:
            for entry in dummy_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Dummy '{input_file}' created with {len(dummy_data)} entries.")

    start_time = time.time()
    
    # 设置并发的 worker 数量。这个值需要根据你的API速率限制、网络带宽和CPU能力进行调整。
    # 太小则速度提升不明显，太大可能导致API拒绝服务或本地资源耗尽。
    num_workers = 10 # 例如，同时运行5个API请求
    
    process_jsonl_threaded(input_file, output_file, max_workers=num_workers)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")