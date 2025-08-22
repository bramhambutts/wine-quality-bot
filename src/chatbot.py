from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Chatbot:

    def __init__(self):
        self.device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
    

    def generate_dictionary(self, user_prompt: str):
        system_prompt = """<|system|>
        You are determining the properties of wine from a user description.
        Extract the following fields from the user input:
        - fixed acidity (float between 3.8 and 15.9)
        - volatile acidity (float between 0.1 and 1.6)
        - citric acid (float between 0.0 and 1.7)
        - residual sugar (float between 0.6 and 65.8)
        - chlorides (float between 0.01 and 0.61)
        - free sulfur dioxide (float between 1 and 289)
        - total sulfur dioxide (float between 6 and 440)
        - density (float between 0.987 and 1.039)
        - pH (float between 2.7 and 4.0)
        - sulphates (float between 0.2 and 2.0)
        - alcohol (percentage float between 8.0 and 14.9)
        - color code (integer, 1 for red, 0 for white)
        Return this as a JSON object only with no other text. Output must be valid JSON. If no value can be determined, leave the value as None</s>"""
        full_prompt = system_prompt + "\n<|user|>\n" + user_prompt + "</s>\n<|assistant|>\n"
        encoded_prompt = self.encode(full_prompt)
        print("prompt encoded")
        response = self.model.generate(**encoded_prompt)
        print("response generated")
        return self.decode(response[0])


    def generate_response(self):
        pass
    

    def encode(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    
    def decode(self, prompt):
        return self.tokenizer.decode(prompt, skip_special_tokens=True)
    

if __name__ == "__main__":
    bot = Chatbot()
    returned = bot.generate_dictionary("I made a new white - it's got 5.5% alcohol, medium acidity, and a touch of sugar.")
    # returned = bot.generate_dictionary("I made a new red - it's got 13.5% alcohol, medium acidity, and a touch of sugar.")
    print(returned)