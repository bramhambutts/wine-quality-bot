from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class Chatbot:
    def __init__(self):
        self.device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
    

    def generate_dictionary(self,user_prompt):
        system_prompt = "<|system|>\nTake the user prompt and convert it into a python dictionary" \
        " with the keys fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide," \
        " total_sulfur_dioxide, density, pH, sulphates, alcohol, and color_code, setting values to None where they are not given.</s>"
        full_prompt = system_prompt + "\n<|user|>\n" + user_prompt + "</s>"
        encoded_prompt = self.encode(full_prompt)
        print("prompt encoded")
        response = self.model.generate(**encoded_prompt)
        print("response generated")
        return self.decode(response)



    def generate_response(self):
        pass
    
    def encode(self,prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    
    def decode(self,prompt):
        return self.tokenizer.decode(prompt,skip_special_tokens=True).to(self.device)
    

if __name__ == "__main__":
    bot = Chatbot()
    returned = bot.generate_dictionary("I made a new red – it’s got 13.5% alcohol, medium acidity, and a touch of sugar.")
    print(returned)