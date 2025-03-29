# reasoning_agent.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ReasoningAgent:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        if use_llm:
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.llm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            except Exception as e:
                print(f"Failed to load LLM: {e}. Falling back to rule-based logic.")
                self.use_llm = False
        else:
            self.use_llm = False

    def generate_description(self, objects, text):
        if not self.use_llm:
            # Rule-based fallback
            description = "I see "
            if objects:
                for obj in objects:
                    description += f"a {obj['object']} at {obj['position']}, "
            if text:
                description += f"and some text that says '{text}'."
            else:
                description += "no text in the scene."
            return description.strip(", ")

        # LLM-based description
        prompt = f"Detected objects: {objects}. Detected text: {text}. Describe the scene for a visually impaired user in a concise way."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description

if __name__ == "__main__":
    reasoning_agent = ReasoningAgent(use_llm=False)  # Set to True if you have a local LLM
    objects = [{"object": "person", "position": "x=100, y=200", "confidence": 0.9}]
    text = "Exit sign"
    description = reasoning_agent.generate_description(objects, text)
    print("Description:", description)