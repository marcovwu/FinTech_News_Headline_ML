# FinTech news headline predict
***

Easy run code with Colab [url here](<https://colab.research.google.com/drive/1dG9PnUcb8gT-coglLzMLUKvM08RS3v7N?usp=sharing>)

***
[0]. Prepare work

Download [synthesis link](<https://drive.google.com/drive/folders/1-OF8ytKnGyHEUorfRxTMFnDa4a0ekbl3?usp=sharing>)

Download pretrained weighted

$ python utils/utils.py

        result  | synthesis ( if you need to synthesis old example )
                | RF_moddel ( pretrained model )
                | XGBoost ( pretrained model )

***
[1]. Create virtual env and install library

$ cd /{your_path}/News_headline_ML

$ pip install -r requirements.txt

***

[2]. Run main.py file directly

$ python main.py

Output will produce in >  ./{your_path}/News_headline_ML/result/output

    Myself output :
    1. eval.txt ( NLTK eval output )
    2. eval_rfc.txt ( RandomForestClassifier eval output )
    3. eval_triple_rfc.txt ( RandomForestClassifier eval output ) 
        > by tripple barrier label
    4. eval_triple_xgb.txt ( xXGBoost eval output )
        > by trple barrier label

Delete result/RF_model will and run this command will get new trained model
Delete result/XGBoost will and run this command will get new trained model

$ python main.py

***

[3]. Run inference.py file directly

$ python inference.py

Output will produce in >  ./{your_path}/News_headline_ML/result/inference_output

    Myself output :
    five companies folder (apple amazon fb nvidia tesla)
        1. title_bar.png ( compound means bar each days in DATE_TIME )
        2. title_3bar.png ( pos neg neu Probability in each time )
        3. title_line.png ( compound means point each days in DATE_TIME )
    five companies csv (apple amazon fb nvidia tesla)
        > ticker,Date,Title,Link,neg,neu,pos,compound,means

[More Info] Run this command will get help in each arguments

$ python inference.py -h
