![DALL·E 2023-04-17 01 58 34 - charles dickens in the form of a robot digital art](https://user-images.githubusercontent.com/97404986/232341961-a16184d2-ba26-40d5-be51-516d5c9f8a9a.png)

## Transformer Writer
As a big fan of classic literature, I decided to make one generator for it.
In this repository, code is presented to train **a simple light-weight decoder transformer** that can generate realistic prompts of english text.

I trained it on the works of the inimitable, Charles Dickens. One of the lines (that speaks for its power) the model generated was:
       __"it mattered me like the applause being reluctant in getting out his language accordingly as it seemed to hold him in the world"__

In this repository, I also, examined the effect of different tokenization techniques on the model's performance.
I trained the model, character leveled, word level and byte-level by using the GPT-2's tokenizer.
It is demonstrated that byte-level works better than word level which works better than character level.
The user can get all the trained model from the trained_models directory.

The model has a total of around 50 million parameters. It is advisable to use a GPU if you want to train it from scratch.

