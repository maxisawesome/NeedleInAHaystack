import sys

import numpy
from run import get_model_to_test,get_evaluator, CommandArgs
from llm_needle_haystack_tester import LLMNeedleHaystackTester
from llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from jsonargparse import CLI
import os
from viz import run_viz
from omegaconf import OmegaConf  as om


os.environ['NIAH_MODEL_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')

os.environ['NIAH_EVALUATOR_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')


def dep(name):
    return f"https://{name}.inf.hosted-on.mosaicml.hosting/v2/"







def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """

    
    
    with open(sys.argv[1]) as f:
        yaml_cfg = om.load(f)
    tasks = yaml_cfg['tasks']
    model_names = set(m['model_name'] for m in yaml_cfg['models'])
    experiment_cfg = {
        "context_lengths_min": yaml_cfg['context_lengths_min'],
        "context_lengths_max": yaml_cfg['context_lengths_max'],
        "context_lengths_num_intervals": yaml_cfg['context_lengths_num_intervals'],
        "document_depth_percent_min": yaml_cfg['document_depth_percent_min'],
        "document_depth_percent_max": yaml_cfg['document_depth_percent_max'],
        "context_lengths_num_intervals": yaml_cfg['context_lengths_num_intervals'],
    }

    models = filter(
        lambda d: d['model_name'] in model_names,
        [
            {
                'provider': 'gemini',
                'model_name': "models/gemini-1.0-pro-latest",
                'output_directory': "results/gemini/gemini-1.0-pro-latest",
                'api_key': os.environ.get('GEMINI_API_KEY', ''),
            },
            {
                'provider': 'anthropic',
                'model_name': "claude-3-opus-20240229",
                'output_directory': "results/anthropic/claude-3-opus-20240229",
                'api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
                'use_messages_api': True
            },
            {
                'provider': 'anthropic',
                'model_name': "claude-3-sonnet-20240229",
                'output_directory': "results/anthropic/claude-3-sonnet-20240229",
                'api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
                'use_messages_api': True
            },
            {
                'provider': 'anthropic',
                'model_name': "claude-3-haiku-20240307",
                'output_directory': "results/anthropic/claude-3-haiku-20240307",
                'api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
                'use_messages_api': True
            },
            {
                'provider': 'openai',
                'model_name': "gpt-4-turbo-preview",
                'output_directory': "results/openai/gpt-4-turbo-preview",
                'api_key': os.environ.get('OPENAI_API_KEY', '')
            },
            {
                'provider': 'mosaicml',
                'model_name': "mosaicml/pi-ift-v1",
                'base_url': 'https://pi-eve6-2mt2bncm-valfull-lb2-32k-mz9p3n.inf.hosted-on.mosaicml.hosting/v2',
                'local': False,
                'tokenizer_name': 'mosaicml/mpt-7b',
                'output_directory': "results/mosaicml/pi-ift-v1"
            },
            {
                'provider': 'openai',
                'model_name': "gpt-3.5-turbo",
                'output_directory': "results/openai/gpt-3.5-turbo",
                'api_key': os.environ.get('OPENAI_API_KEY', '')
            },
            {
                'provider': 'mosaicml',
                'model_name': "mistralai/Mistral-large",
                'base_url': 'https://Mistral-large-zqjvu-serverless.eastus2.inference.ai.azure.com/v1',
                'local': False,
                'api_key': os.environ.get("MISTRAL_API_KEY", ''),
                'tokenizer_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'output_directory': "results/mistralai/Mistral-large"
            },
            {
                'provider': 'mosaicml',
                'model_name': "mistralai/Mixtral-8x7B-Instruct-v0.1",
                'base_url': 'https://mixtral-8x7b-instruct-at-trtllm-newimg-lorxa5.inf.hosted-on.mosaicml.hosting/v2',
                'local': False,
                'tokenizer_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'output_directory': "results/mistralai/Mixtral-8x7B-Instruct-v0.1"
            },
        ]
    )



    for model_cfg in models:
        print(f"Testing model: {model_cfg['model_name']}")
        model_overrides = list(
            filter(
            lambda d: d['model_name'] == model_cfg['model_name'],
            yaml_cfg['models']
            )
        )[0]

        model_cfg.update(model_overrides)
        for idx, task_cfg in enumerate(tasks):
            args = CommandArgs()
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
            experiment_cfg['context_lengths_max']
        )

if __name__ == "__main__":
    main()