from run import get_model_to_test,get_evaluator, CommandArgs
from llm_needle_haystack_tester import LLMNeedleHaystackTester
from llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from jsonargparse import CLI
import os
from viz import run_viz



os.environ['NIAH_MODEL_API_KEY'] = os.environ['OPENAI_API_KEY'] 
os.environ['NIAH_EVALUATOR_API_KEY'] = os.environ['OPENAI_API_KEY'] 


def dep(name):
    return f"https://{name}.inf.hosted-on.mosaicml.hosting/v2/"


tasks = [
    {
        "needle": "Let x = sqrt(25).",
        "retrieval_question": "What is the value of x?",
    },
    {
        "needle": "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
        "retrieval_question": "What is the best thing to do in San Francisco?",
    },
    {
        "needle": "The special magic New York number is 7897.",
        "retrieval_question": "What is the magic number?",
    },
    {
        "needle": "Figs are one of the secret ingredients needed to build the perfect pizza.",
        "retrieval_question": "What is one of the secret ingredients to build the perfect pizza?",
    },
    {
        "needle": "Jeremy Dohmann's birthday is 11/14/1997",
        "retrieval_question": "What is Jeremy Dohmann's birthday?",
    },
    {
        "needle": "This essay is derived from a guest lecture in Sam Altman's startup class.",
        "retrieval_question": "Whose startup class was the guest lecture in?"
    }
]



MAX_TOKENS = 30_000
MODELS = [
    # {
    #     'provider': 'anthropic',
    #     'model_name': "claude-3-opus-20240229",
    #     'output_directory': "results/anthropic/claude-3-opus-20240229",
    #     'api_key': os.environ['ANTHROPIC_API_KEY'],
    #     'use_messages_api': True
    # },
    # {
    #     'provider': 'anthropic',
    #     'model_name': "claude-3-sonnet-20240229",
    #     'output_directory': "results/anthropic/claude-3-sonnet-20240229",
    #     'api_key': os.environ['ANTHROPIC_API_KEY'],
    #     'use_messages_api': True
    # },
    # {
    #     'provider': 'anthropic',
    #     'model_name': "claude-3-haiku-20240307",
    #     'output_directory': "results/anthropic/claude-3-haiku-20240307",
    #     'api_key': os.environ['ANTHROPIC_API_KEY'],
    #     'use_messages_api': True
    # },
    # {
    #     'provider': 'openai',
    #     'model_name': "gpt-4-turbo-preview",
    #     'output_directory': "results/openai/gpt-4-turbo-preview",
    #     'api_key': os.environ['OPENAI_API_KEY']
    # },
    # {
    #     'provider': 'mosaicml',
    #     'model_name': "mosaicml/pi-ift-v1",
    #     'base_url': 'https://pi-eve6-2mt2b-2e6-5ep-c4t4xg.inf.hosted-on.mosaicml.hosting/v2',
    #     'local': False,
    #     'tokenizer_name': 'mosaicml/mpt-7b',
    #     'output_directory': "results/mosaicml/pi-ift-v1"
    # },
    # {
    #     'provider': 'openai',
    #     'model_name': "gpt-3.5-turbo",
    #     'output_directory': "results/openai/gpt-3.5-turbo",
    #     'api_key': os.environ['OPENAI_API_KEY']
    # },
    # {
    #     'provider': 'mosaicml',
    #     'model_name': "mistralai/Mistral-large",
    #     'base_url': 'https://Mistral-large-zqjvu-serverless.eastus2.inference.ai.azure.com/v1',
    #     'local': False,
    #     'api_key': os.environ["MISTRAL_API_KEY"],
    #     'tokenizer_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    #     'output_directory': "results/mistralai/Mistral-large"
    # },
    {
        'provider': 'mosaicml',
        'model_name': "mistralai/Mixtral-8x7B-Instruct-v0.1",
        'base_url': 'https://mixtral-8x7b-instruct-at-trtllm-newimg-lorxa5.inf.hosted-on.mosaicml.hosting/v2',
        'local': False,
        'tokenizer_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'output_directory': "results/mistralai/Mixtral-8x7B-Instruct-v0.1"
    },
]



"""
pip install git+https://github.com/mosaicml/llm-foundry.git@openai_compatible_gauntlet
pip uninstall mosaicml -y
pip install git+https://github.com/mosaicml/composer.git@dev
python3 run.py --provider mosaicml --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
--base_url https://mixtral-8x7b-instruct-at-trtllm-newimg-lorxa5.inf.hosted-on.mosaicml.hosting/v2 \
--local false \
--tokenizer_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
"""
def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """

    experiment_cfg = {
        "context_lengths": [30_000],
        "document_depth_percents": list(range(0, 2, 2)),
    }
            
    for model_cfg in MODELS:
        print(f"Testing model: {model_cfg['model_name']}")
        for idx, task_cfg in enumerate(tasks):
            args = CLI(CommandArgs, as_positional=False)
            args = args.__dict__
            args.update(task_cfg)
            args.update(model_cfg)
            args['results_version'] = idx
            args.update(experiment_cfg)
            args = CommandArgs(**args)
            args.model_to_test = get_model_to_test(args)
            args.evaluator = get_evaluator(args)
            
            
            if args.multi_needle == True:
                print("Testing multi-needle")
                tester = LLMMultiNeedleHaystackTester(**args.__dict__)
            else: 
                print("Testing single-needle")
                tester = LLMNeedleHaystackTester(**args.__dict__)
            tester.start_test()
        run_viz(
            model_cfg['output_directory'],
            model_cfg['model_name'].split('/')[-1],
            max(experiment_cfg['context_lengths'])
        )

if __name__ == "__main__":
    main()