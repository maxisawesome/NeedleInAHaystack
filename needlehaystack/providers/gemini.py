import tiktoken
import google.generativeai as google_genai
from typing import Optional
from .model import ModelProvider




class Gemini(ModelProvider):
    
    _client = None
    DEFAULT_MODEL_KWARGS: dict = dict(max_output_tokens  = 300,
                                      temperature = 0)
    
    def __init__(self, model_name: str = "models/gemini-1.0-pro-latest",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 api_key: Optional[str] = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        google_genai.configure(api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0125") # the precise tokenizer is irrelevant
        self.model = google_genai.GenerativeModel(model_name)
        ignore = [
            # google_genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
            # google_genai.types.HarmCategory.HARM_CATEGORY_DEROGATORY,
            # google_genai.types.HarmCategory.HARM_CATEGORY_TOXICITY,
            # google_genai.types.HarmCategory.HARM_CATEGORY_VIOLENCE,
            # google_genai.types.HarmCategory.HARM_CATEGORY_SEXUAL,
            # google_genai.types.HarmCategory.HARM_CATEGORY_MEDICAL,
            # google_genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS,
            google_genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            google_genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            google_genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            google_genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        self.safety_settings = {
            category: google_genai.types.HarmBlockThreshold.BLOCK_NONE
            for category in ignore
        }

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return  f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct{context}\nQuestion:{retrieval_question} Don't give information outside the document or repeat your findings."
   
    
    async def evaluate_model(self, 
        prompt: str,
      
    ) -> str:
        
        generation_config=google_genai.types.GenerationConfig(
            candidate_count=1,
            **self.model_kwargs
        )
        response = self.model.generate_content(prompt,
                safety_settings=self.safety_settings,
                generation_config=generation_config)
        return response.text

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])

