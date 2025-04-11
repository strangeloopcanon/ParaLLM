import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

def bedrock_infer(prompt, model_id, max_tokens=512, temperature=0.7, top_k=None, top_p=0.9):
    provider = model_id.split(".")[0]

    def format_payload():
        if provider == "anthropic":
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop_sequences": [],
            }

        elif provider == "deepseek":
            formatted_prompt = f"<|begin_of_sentence|><|User|>{prompt}<|Assistant|>"
            return {
                "prompt": formatted_prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
            }

        elif provider == "mistral":
            return {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": [],
            }

        elif provider == "meta":
            return {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": [],
            }

        elif provider == "amazon":
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                    "stopSequences": [],
                }
            }

        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def extract_output(response_body):
        if provider == "anthropic":
            return response_body["content"][0]["text"]
        elif provider == "deepseek":
            return response_body["choices"][0]["text"]
        elif provider in {"mistral"}:
            return response_body["outputs"][0]["text"]
        elif provider == "meta":
            return response_body["generation"]
        elif provider == "amazon":
            return response_body["results"][0]["outputText"]
        else:
            return json.dumps(response_body)

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(format_payload()),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return extract_output(response_body)

model_ids = [
    # "anthropic.claude-3-sonnet-20240229-v1:0",
    # "deepseek.r1-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "amazon.titan-text-express-v1"
]

for mid in model_ids:
    print(f"\n--- {mid} ---")
    print(bedrock_infer("Summarize the Habsburg dynasty in 2 sentences.", model_id=mid))
