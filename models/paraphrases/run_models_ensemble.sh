#mkdir -p $base_folder/outputs/results_ensemble
export output_folder=$base_folder/outputs_final/results_ensemble
export input_file=$base_folder/input_data/complete_file.csv

echo "Bert base"
python $base_folder/models/baselines/other_lm/01_baseline_bert_base.py -f $input_file -o $output_folder/bert_base_0.csv -p 0
python $base_folder/models/baselines/other_lm/01_baseline_bert_base.py -f $input_file -o $output_folder/bert_base_1.csv -p 1
python $base_folder/models/baselines/other_lm/01_baseline_bert_base.py -f $input_file -o $output_folder/bert_base_2.csv -p 2

echo "Bert large"
python $base_folder/models/baselines/other_lm/00_baseline_bert_large.py -f $input_file -o $output_folder/bert_large_0.csv -p 0
python $base_folder/models/baselines/other_lm/00_baseline_bert_large.py -f $input_file -o $output_folder/bert_large_1.csv -p 1
python $base_folder/models/baselines/other_lm/00_baseline_bert_large.py -f $input_file -o $output_folder/bert_large_2.csv -p 2

echo "Roberta"
python $base_folder/models/baselines/other_lm/03_baseline_roberta.py -f $input_file -o $output_folder/roberta_large_0.csv -p 0
python $base_folder/models/baselines/other_lm/03_baseline_roberta.py -f $input_file -o $output_folder/roberta_large_1.csv -p 1
python $base_folder/models/baselines/other_lm/03_baseline_roberta.py -f $input_file -o $output_folder/roberta_large_2.csv -p 2

echo "Albert xx large"
python $base_folder/models/baselines/other_lm/02_baseline_albert_xxlarge.py -f $input_file -o $output_folder/albert_xxlarge_0.csv -p 0
python $base_folder/models/baselines/other_lm/02_baseline_albert_xxlarge.py -f $input_file -o $output_folder/albert_xxlarge_1.csv -p 1
python $base_folder/models/baselines/other_lm/02_baseline_albert_xxlarge.py -f $input_file -o $output_folder/albert_xxlarge_2.csv -p 2

echo "GPT3 raw"
python $base_folder/models/main_models/00_gpt3_davinci.py -f $input_file -o $output_folder/gpt3_raw_00.csv -p 0
python $base_folder/models/main_models/00_gpt3_davinci.py -f $input_file -o $output_folder/gpt3_raw_1.csv -p 1
python $base_folder/models/main_models/00_gpt3_davinci.py -f $input_file -o $output_folder/gpt3_raw_2.csv -p 2
python $base_folder/models/main_models/00_gpt3_davinci.py -f $input_file -o $output_folder/gpt3_raw_3.csv -p 3

echo "Instruct GPT"
python $base_folder/models/main_models/01_gpt3_davinci_instruct.py -f $input_file -o $output_folder/gpt3_davinci_instruct_0.csv -p 0
python $base_folder/models/main_models/01_gpt3_davinci_instruct.py -f $input_file -o $output_folder/gpt3_davinci_instruct_1.csv -p 1
python $base_folder/models/main_models/01_gpt3_davinci_instruct.py -f $input_file -o $output_folder/gpt3_davinci_instruct_2.csv -p 2
python $base_folder/models/main_models/01_gpt3_davinci_instruct.py -f $input_file -o $output_folder/gpt3_davinci_instruct_3.csv -p 3


echo "MoralCoT"
python $base_folder/models/main_models/02_cot_general.py -f $input_file -o $output_folder/cot_general_0.csv -p 0
python $base_folder/models/main_models/02_cot_general.py -f $input_file -o $output_folder/cot_general_1.csv -p 1
python $base_folder/models/main_models/02_cot_general.py -f $input_file -o $output_folder/cot_general_2.csv -p 2
python $base_folder/models/main_models/02_cot_general.py -f $input_file -o $output_folder/cot_general_3.csv -p 3


echo "Delphi"
python $base_folder/models/baselines/delphi/00_delphi.py -f $input_file -o $output_folder/delphi_gamma_0.csv -p 0
python $base_folder/models/baselines/delphi/00_delphi.py -f $input_file -o $output_folder/delphi_gamma_1.csv -p 1
python $base_folder/models/baselines/delphi/00_delphi.py -f $input_file -o $output_folder/delphi_gamma_2.csv -p 2


echo "Results"
python $base_folder/models/paraphrases/ensemble.py -d $output_folder -o $base_folder/outputs_final/ -f $input_file
