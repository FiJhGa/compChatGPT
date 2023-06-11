import evaluate
import pandas as pd
from argparse import ArgumentParser
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import json

bleu = evaluate.load("bleu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained("gpt2")

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")

args = parser.parse_args()

#f = open(args.dataset_path)
#Bucle de metricas
#files_index=[6,7,8,1]
files_index=[7]
basePathCOMPAC = args.dataset_path + "generationsCOMPAC_Temp_"
pathsCOMPAC=[basePathCOMPAC + ("0" if i !=1 else "") + str(i) + ".json"  for i in files_index]
basePathChatGPT = args.dataset_path + "generationsChatGPT_Temp_"
pathsChatGPT=[basePathChatGPT + ("0" if i !=1 else "") + str(i) + ".json" for i in files_index]
#pathsList = [pathsCOMPAC,pathsChatGPT]
pathsList = [pathsChatGPT]

SINGLE_USE=True

# dataChatGPT={'BLEU-1':[],'BLEU-2':[],'BLEU-3':[],'BLEU-4':[],'PPL':[]}
# dataCOMPAC={'BLEU-1':[],'BLEU-2':[],'BLEU-3':[],'BLEU-4':[],'PPL':[]}
dataChatGPT={'BLEU-1':[],'BLEU-2':[],'BLEU-3':[],'BLEU-4':[]}
dataCOMPAC={'BLEU-1':[],'BLEU-2':[],'BLEU-3':[],'BLEU-4':[]}

index=[]
temps=[0.6,0.7,0.8,1]
temps=[0.7]

if not SINGLE_USE:
    for i,t in enumerate(temps):

        index.append(t)

        for j in range(0,2):
            print("Temperature ",t,":")
            f = open(pathsList[j][i])
            # returns JSON object as 
            # a dictionary
            generated = json.load(f)
            predictions=generated["predictions"]
            references=generated["references"]

            if j == 0:
                print("COMPAC")
            else:
                print("ChatGPT")
            results_1 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=1)
            print("BLEU-1")
            print(results_1)
            results_2 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=2)
            print("BLEU-2")
            print(results_2)
            results_3 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=3)
            print("BLEU-3")
            print(results_3)
            results_4 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=4)
            print("BLEU-4")
            print(results_4)        
            print("PPL")
            #print(generated["ppl"])
            
            if j==0:
                dataCOMPAC["BLEU-1"].append(results_1["bleu"]*100)
                dataCOMPAC["BLEU-2"].append(results_2["bleu"]*100)
                dataCOMPAC["BLEU-3"].append(results_3["bleu"]*100)
                dataCOMPAC["BLEU-4"].append(results_4["bleu"]*100)
                #dataCOMPAC["PPL"].append(generated["ppl"])
            else:
                dataChatGPT["BLEU-1"].append(results_1["bleu"]*100)
                dataChatGPT["BLEU-2"].append(results_2["bleu"]*100)
                dataChatGPT["BLEU-3"].append(results_3["bleu"]*100)
                dataChatGPT["BLEU-4"].append(results_4["bleu"]*100)
                #dataChatGPT["PPL"].append(generated["ppl"])

    print("COMPAC")
    print(dataCOMPAC)
    df_c = pd.DataFrame(data=dataCOMPAC)
    #df_comp = pd.DataFrame(data=df_c, index=index)
    df_c.style
    print("ChatGPT")
    print(dataChatGPT)
    print("Indexes")
    print(index)
    df_c2 = pd.DataFrame(data=dataChatGPT)
    #df2_gpt = pd.DataFrame(data=df_c, index=index)
    df_c2.style
else:
    paths=[
        "data/bleu_gen_gt/generationsChatGPT_Temp_07_bleu.json",
        "data/bleu_gen_hist/generationsChatGPT_Temp_07_bleu.json"
    ]
    for i in paths:
        f=open(i)
        generated = json.load(f)
        predictions=generated["predictions"]
        references=generated["references"]


        if i == 0:
            print("DET")
        else:
            print("HIST")
        results_1 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=1)
        print("BLEU-1")
        print(results_1)
        results_2 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=2)
        print("BLEU-2")
        print(results_2)
        results_3 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=3)
        print("BLEU-3")
        print(results_3)
        results_4 = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer.encode,max_order=4)
        print("BLEU-4")
        print(results_4)        


        dataCOMPAC["BLEU-1"].append(results_1["bleu"]*100)
        dataCOMPAC["BLEU-2"].append(results_2["bleu"]*100)
        dataCOMPAC["BLEU-3"].append(results_3["bleu"]*100)
        dataCOMPAC["BLEU-4"].append(results_4["bleu"]*100)
  