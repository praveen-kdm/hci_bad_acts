from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFChatCompletionClient:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

    def create(self, messages, max_tokens=200):
        """
        messages: list of dicts like:
        [{"role": "user", "content": "Hello"}]
        """

        # Convert chat messages to a single prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only assistant response
        response = response.split("assistant:")[-1].strip()

        return {
            "choices": [
                {
                    "message": {
                        "content": response
                    }
                }
            ]
        }


    def generate(self, prompt, max_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Only return newly generated text
        generated = full_text[len(prompt):]

        # CRITICAL: stop at first newline (prevents repetition)
        generated = generated.split("\n")[0]

        return generated.strip()