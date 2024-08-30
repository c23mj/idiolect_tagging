import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)

model.eval()

paragraph = ["A good example of a paragraph contains a topic sentence, details and a conclusion. 'There are many different kinds of animals that live in China. Tigers and leopards are animals that live in China's forests in the north. In the jungles, monkeys swing in the trees and elephants walk through the brush. There are camels in the deserts in China that people use for transportation. Lots of different kinds of animals make their home in China."]

# Tokenize the paragraph
tokenized_text = tokenizer(
    paragraph,
    max_length=500, 
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

tokenized_text['input_ids'] = tokenized_text['input_ids'].reshape([1, 1, -1])
tokenized_text['attention_mask'] = tokenized_text['attention_mask'].reshape([1, 1, -1])

# Run the model on the tokenized input
outputs = model(**tokenized_text)

print(outputs.size())
print(outputs.detach().cpu().numpy())

