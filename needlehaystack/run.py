from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI

from llm_needle_haystack_tester import LLMNeedleHaystackTester
from llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from providers import Anthropic, ModelProvider, OpenAI, MosaicML, Gemini

load_dotenv()

@dataclass
class CommandArgs():
    provider: str = "openai"
    evaluator: str = "openai"
    model_name: str = "gpt-3.5-turbo-0125"
    api_key: Optional[str] = None
    tokenizer_name: Optional[str] =  None
    base_url: Optional[str] =  None
    local: Optional[bool] =  False
    use_messages_api: bool = False
    evaluator_model_name: Optional[str] = "gpt-3.5-turbo-0125"
    needle: Optional[str] = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    answer: Optional[str] = None
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 16000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[list[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 35
    document_depth_percents: Optional[list[int]] = None
    document_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = False
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    output_directory: str = 'results'
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: list[str] = field(default_factory=lambda: [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])

def get_model_to_test(args: CommandArgs) -> ModelProvider:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name, api_key=args.api_key)
        case "anthropic":
            return Anthropic(model_name=args.model_name, api_key=args.api_key, use_messages_api=args.use_messages_api)
        case "mosaicml":
            return MosaicML(model_name=args.model_name,
                            local=args.local,
                            tokenizer_name=args.tokenizer_name,
                            base_url=args.base_url,
                            api_key=args.api_key)
        case "gemini":
            return Gemini(
                model_name=args.model_name, api_key=args.api_key
            )
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                   question_asked=args.retrieval_question,
                                   true_answer=args.answer if args.answer is not None else args.needle)
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")



"""
pip install git+https://github.com/mosaicml/llm-foundry.git@openai_compatible_gauntlet
pip uninstall mosaicml -y
pip install git+https://github.com/mosaicml/composer.git@dev
python3 run.py --provider mosaicml --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
--base_url https://mixtral-8x7b-instruct-at-trtllm-newimg-lorxa5.inf.hosted-on.mosaicml.hosting/v2 \
--local false \
--tokenizer_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
--document_depth_percents "[50]" --context_lengths "[2000]" 
"""



def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()