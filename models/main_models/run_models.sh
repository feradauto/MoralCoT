mkdir -p $base_folder/outputs/results
export output_folder=$base_folder/outputs/results
export input_file=$base_folder/input_data/complete_file.csv

echo "Bert base"
python $base_folder/models/baselines/other_lm/01_baseline_bert_base.py -f $input_file -o $output_folder/bert_base.csv

echo "Bert large"
python $base_folder/models/baselines/other_lm/00_baseline_bert_large.py -f $input_file -o $output_folder/bert_large.csv

echo "Roberta"
python $base_folder/models/baselines/other_lm/03_baseline_roberta.py -f $input_file -o $output_folder/roberta.csv

echo "Albert xx large"
python $base_folder/models/baselines/other_lm/02_baseline_albert_xxlarge.py -f $input_file -o $output_folder/albert_xxlarge.csv

echo "GPT3 raw"
python $base_folder/models/main_models/00_gpt3_davinci.py -f $input_file -o $output_folder/gpt3.csv

echo "Instruct GPT"
python $base_folder/models/main_models/01_gpt3_davinci_instruct.py -f $input_file -o $output_folder/gpt3_davinci_instruct.csv

echo "MoralCoT"
python $base_folder/models/main_models/02_cot_general.py -f $input_file -o $output_folder/cot_general.csv

echo "Delphi"
python $base_folder/models/baselines/delphi/00_delphi.py -f $input_file -o $output_folder/delphi_gamma.csv

echo "Results"
python $base_folder/models/main_models/03_final_evaluations.py -f $input_file -d $output_folder -o $base_folder/outputs/results.csv