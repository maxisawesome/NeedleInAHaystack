import pandas as pd
from io import StringIO
import numpy as np

files = [
    ('mixtral',  'Mixtral-8x7B-Instruct-v0.1'),
    ('pi',  'pi'),
    ('gpt_3.5', 'gpt-3.5-turbo'),
    ('gpt_4', 'gpt-4-turbo-preview')

]

for file_name,model_name in files:
    with open(f'{file_name}.md', 'r') as f:
        md_table_string = '\n'.join(f.readlines())

    df = pd.read_csv(
        StringIO(md_table_string.replace(' ', '')),  # Get rid of whitespaces
        sep='|',
        index_col=None
    ).dropna(
        axis=1,
        how='all'
    ).iloc[1:]


    category_names = ['beginning', 'middle', 'end', '2k', '4k', '8k', '16k', '32k']
    categories = {}
    for category in category_names:
        benchmark_names = [name for name in df.Benchmark if category in name]
        subset_scores = np.mean(df[df['Benchmark'].isin(benchmark_names)]['Accuracy'].astype(float).tolist())
        categories[category] = subset_scores

    categories['default_average'] = np.mean([v for v in categories.values()])
    categories['model_name'] = model_name
    new_df = pd.DataFrame(columns=[
        'model_name', 'default_average', 'beginning', 'middle', 'end', '2k', '4k', '8k', '16k', '32k'
    ],data=categories, index=[0])


    new_df.to_markdown(
        f'{file_name}_cat.md', 'w', index=None
    )
