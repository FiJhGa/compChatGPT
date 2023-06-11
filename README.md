#For testing
generated_MOD : Generate predictions given the utterance
python -m models.reinforce_model.generate_MOD --dataset_path="data/expanded_profiles_personachat.json" --model_checkpoint_dir="models/reinforce_model/runs/May29_09-41-09_pgth07_gpt2_reinforce_ChatGPT_beams_ALL_Det_Hist" --dataset_cache="persona_ChatGPT_expansions_test" --num_beams=-1 --test_run_num=-1 --load_checkpoint_from="checkpoint_mymodel_911160.pt" --max_history=2 --max_length=20 --temperature=0.7 --save_loc="data/bleu_gen_hist/generationsChatGPT_Temp_07_bleu_history.json" --top_k=50 --top_p=0.95 --greedy_profile=True --use_history=True
bleu.py : Generate bleu and ppl metrics
