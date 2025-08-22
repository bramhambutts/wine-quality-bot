from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
from src.classifier import WineClassifier


SYSTEM_PROMPT = """<|system|>
You extract structured data from natural language. Respond only with a valid JSON object.
Based on the typical properties of wines, estimate the values for the following structured fields from the user input, leaving values as null if they cannot be determined:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- color code (0 for white, 1 for red)
</s>"""
# You are determining the properties of wine from a user description, extracting only the following fields:
# - fixed acidity (float typically between 3.8 and 15.9)
# - volatile acidity (float typically between 0.1 and 1.6)
# - citric acid (float typically between 0.0 and 1.7)
# - residual sugar (float typically between 0.6 and 65.8)
# - chlorides (float typically between 0.01 and 0.61)
# - free sulfur dioxide (float typically between 1 and 289)
# - total sulfur dioxide (float typically between 6 and 440)
# - density (float typically between 0.987 and 1.039)
# - pH (float typically between 2.7 and 4.0)
# - sulphates (float typically between 0.2 and 2.0)
# - alcohol (percentage float of alcohol level)
# - color code (integer, 1 for red wine, 0 for white wine)
# Return a JSON object. Do not return anything else.
# Output must be valid JSON with null values where a property cannot be determined</s>"""


class Chatbot:

    def __init__(self):
        self.device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
    

    def generate_dictionary(self, user_prompt: str):
        system_prompt = SYSTEM_PROMPT
        
        # system_prompt = "<|system|>Get an estimate of the pH of the wine from the user input.</s>"
        full_prompt = system_prompt + "\n<|user|>\n" + user_prompt + "</s>\n<|assistant|>\n"
        encoded_prompt = self.encode(full_prompt)
        print("prompt encoded")
        response = self.model.generate(**encoded_prompt)
        print("response generated")
        text_response = self.decode(response[0])
        # return text_response
        segments = re.split('```', text_response)
        return segments[-2].strip('json')


    def generate_response(self, user_prompt: str, quality: int):
        system_prompt = f"""<|system|>
        You give a response to the user based on the determined quality of their wine.
        The quality is on a scale of 0-10 and has been found to be {quality}.</s>"""
        full_prompt = system_prompt + '\n<|user|>\n' + user_prompt + '</s>\n<|assistant|>\n'
        encoded_prompt = self.encode(full_prompt)
        response = self.model.generate(**encoded_prompt)
        text_response = self.decode(response[0])
        segments = re.split('<|assistant|>', text_response)
        return segments[-1]
    

    def predict_quality(self, user_prompt: str):
        generated = self.generate_dictionary(user_prompt)
        generated = generated.replace("'", '"')
        generated = re.sub(r'\w \w', lambda x: x.group().replace(' ', '_'), generated)
        print(generated)
        features = json.loads(generated)
        for feature in features.keys():
            features[feature] = [features[feature]]
        classifier = WineClassifier()
        quality = classifier.classify_wine(features)
        return self.generate_response(user_prompt, quality)


    def encode(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    
    def decode(self, prompt):
        return self.tokenizer.decode(prompt, skip_special_tokens=True)
    

if __name__ == "__main__":
    bot = Chatbot()
    returned = bot.predict_quality("I made a new white - it's got 5.5% alcohol, medium acidity, and a touch of sugar.")
    # returned = bot.generate_dictionary("I made a new red - it's got 13.5% alcohol, medium acidity, and a touch of sugar.")
    print(returned)