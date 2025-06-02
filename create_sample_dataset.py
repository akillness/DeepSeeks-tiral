import json
import os

def create_deepseek_sample_dataset(output_filename="sample_deepseek_dataset.jsonl"):
    """
    Generates a sample JSONL dataset suitable for fine-tuning DeepSeek models,
    compatible with the formatting_prompts_func in finetune_with_unsloth.py.

    The dataset will primarily use the "user" and "assistant" fields.
    The fine-tuning script (finetune_with_unsloth.py) will then format these
    fields by adding the  and  special tokens automatically.
    """
    sample_data = [
        {
            "user": "Hello, who are you?",
            "assistant": "I am a large language model, trained by Unsloth. I am currently configured to emulate DeepSeek."
        },
        {
            "user": "What is the capital of France?",
            "assistant": "The capital of France is Paris."
        },
        {
            "user": "Can you explain what a transformer model is in simple terms?",
            "assistant": "Imagine a very smart student who can read a whole book (your input text) at once, understand how every word relates to every other word, and then write a summary or answer questions based on that deep understanding. Transformer models work somewhat like that, using a mechanism called 'attention' to weigh the importance of different parts of the input when generating an output."
        },
        {
            "user": "Write a short story about a friendly robot discovering a garden.",
            "assistant": "Unit 734, or 'Bolt' as he preferred, had only ever known the gleaming chrome corridors of the starship. One day, an unscheduled landing on an uncharted planet led him to a vibrant, sprawling garden. His optical sensors, usually analyzing star charts, were overwhelmed by the riot of colors: ruby reds, sapphire blues, emerald greens. He extended a metallic finger to a soft, velvety petal. \"Fascinating,\" he whirred, his monotone voice filled with a new, unknown emotion. He spent hours (or cycles, as he knew them) cataloging each plant, his internal processors creating a new database not of celestial bodies, but of earthly delights. Bolt decided this 'garden' was the most illogical, yet beautiful, place he had ever encountered."
        },
        {
            "user": "I want to learn Python. Where should I start?",
            "assistant": "That's a great choice! Python is very beginner-friendly. I recommend starting with the official Python tutorial on python.org. It's comprehensive and well-structured. Websites like Codecademy, freeCodeCamp, and Coursera also offer excellent interactive courses. What kind of projects are you interested in building? Knowing that might help me suggest more specific resources."
        }
    ]

    # If you wanted to pre-format the text with DeepSeek tokens in the dataset itself (alternative):
    # sample_data_alternative_format = [
    #     {
    #         "text": "Hello, who are you?I am a large language model, trained by Unsloth."
    #     },
    #     {
    #         "text": "What is the capital of France?The capital of France is Paris."
    #     }
    # ]

    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in sample_data:
            json.dump(entry, f)
            f.write('\n')

    print(f"Sample dataset '{output_filename}' created successfully with {len(sample_data)} entries.")
    print(f"This file can be used with 'finetune_with_unsloth.py --dataset_path_train {output_filename}'")

if __name__ == "__main__":
    create_deepseek_sample_dataset() 