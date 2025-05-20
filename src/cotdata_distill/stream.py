from openai import OpenAI
import json

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='ab86248c-8b98-4c51-981c-c23f923a101a', # ModelScope Token
)

# set extra_body for thinking control
extra_body = {
    # enable thinking, set to False to disable
    "enable_thinking": True,
    # use thinking_budget to contorl num of tokens used for thinking
    # "thinking_budget": 4096
}

def get_ai_response(patient_note, ground_truth):
    response = client.chat.completions.create(
        model='Qwen/Qwen3-235B-A22B',  # ModelScope Model-Id
        messages=[
            {
                "role": "user",
                "content": f"你是一个严谨且富有经验的临床医生，请你根据患者信息{patient_note}提出你怀疑的疾病可能有哪些，且为了做出鉴别诊断，应该做什么检验项目？最后的结果要与答案相符{ground_truth}"
            }
        ],
        stream=True,
        extra_body=extra_body
    )
    

    all_thinking = ""
    all_answer = ""

    done_thinking = False
    for chunk in response:
        thinking_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if thinking_chunk != '':
            print(thinking_chunk, end='', flush=True)
            all_thinking += thinking_chunk
        elif answer_chunk != '':
            if not done_thinking:
                print('\n\n === Final Answer ===\n')
                done_thinking = True
            print(answer_chunk, end='', flush=True)
            all_answer += answer_chunk
    final_answer=all_thinking+all_answer
    return final_answer

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            patient_note = data['patient_note']
            ground_truth = data['ground_truth']
            
            # Get AI response
            ai_response = get_ai_response(patient_note, ground_truth)
            
            # Add rationale to the data
            data['rationale'] = ai_response
            
            # Write updated data to output file
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "test.jsonl"  # Replace with your input file path
    output_file = "output.jsonl"  # Replace with your desired output file path
    process_jsonl(input_file, output_file)