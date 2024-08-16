import csv

import os

import backoff

import openai

import pandas as pd

from tqdm import tqdm


class ChatGPTData:
    def __init__(self, input_file: str, output_file: str) -> None:
        self.input_file = input_file  
        self.output_file = output_file
        openai.api_key = os.environ.get('OPENAI_KEY')
        self.completion = openai.ChatCompletion()

        self.prompt =  """Pretend you are a physician: Write a 3 sentence fragment of a clinical note for a patient that mentions the condition {disease} in the second sentence, either explicitly or as a synonym or abbreviation to this condition. 
        Place the tokens <1CUI> before AND after the mention of this condition. For example, <1CUI> {disease} <1CUI>.
        Do NOT include any other conditions or problem in the second sentence, only procedures, tests and social history. 
        Do NOT place <1CUI> tokens around any procedures, tests and social history."""

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.APIError,
            openai.Timeout,
        ),
    )
    def askgpt(self, question, chat_log=None):
        if chat_log is None:
            chat_log = [{
                'role': 'system',
                'content': 'You are a physician.',
            }]
        chat_log.append({'role': 'user', 'content': question})
        response = self.completion.create(model='gpt-3.5-turbo', messages=chat_log, temperature=1.35)
        answer = response.choices[0]['message']['content']
        chat_log.append({'role': 'assistant', 'content': answer})
        return answer, chat_log

    def generate(self) -> None:
        input_df = pd.read_csv(self.input_file, quotechar='"')
        default = False

        with open(self.output_file, "w") as f:
            csvwriter = csv.writer(f) 

            csvwriter.writerow(["cui", "default", "output"])

            for row in tqdm(input_df.itertuples()):
                cui, disease = row.CUI, row.STR
                prepared_prompt = self.prompt.format(disease=disease)
                answer, log = self.askgpt(prepared_prompt)

                answer = answer.replace("\n", " ")

                csvwriter.writerow([cui, default, answer])