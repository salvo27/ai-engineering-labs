# Deep Learning Intro Labs

Hands-on companion to the [deep-learning-intro-course](../deep-learning-intro-course) theory notes. Every lab is a Jupyter notebook, available in English (`en/`) and Italian (`it/`).

| Lab | Topic | Notebooks |
| --- | --- | --- |
| 01-music-generation | Character-level LSTM that composes Irish folk music in ABC notation | [en](01-music-generation/en) / [it](01-music-generation/it) |
| 02-computer-vision | MNIST digit classification (fully connected vs CNN) and debiasing a facial detection system with a DB-VAE | [en](02-computer-vision/en) / [it](02-computer-vision/it) |

## How to run the labs (Kaggle)

These notebooks are designed to run on **Kaggle**, which provides a free GPU and a clean way to store API keys:

1. Create a new Notebook on [Kaggle](https://www.kaggle.com/code) and import the lab (File > Import Notebook);
2. In the notebook settings, set **Accelerator** to a GPU (for example a T4 or P100) and enable **Internet**;
3. Create a free account on [Comet](https://www.comet.com) (used to track experiments) and copy your **API key**;
4. On Kaggle, open **Add-ons > Secrets**, add a secret with label `COMET_API_KEY`, and attach it to the notebook.

The key is read through `kaggle_secrets`, so it never appears in the code. To run elsewhere (Colab, local machine), just replace that block with your own way of loading the key.

The `mitdeeplearning` pip package is a small utility library from the open MIT deep learning course: the labs use it only for dataset downloads and a few plotting and audio helpers.

## Come eseguire i lab (Kaggle)

Questi notebook sono pensati per girare su **Kaggle**, che offre una GPU gratuita e un modo pulito per conservare le API key:

1. Crea un nuovo Notebook su [Kaggle](https://www.kaggle.com/code) e importa il lab (File > Import Notebook);
2. Nelle impostazioni del notebook, imposta **Accelerator** su una GPU (ad esempio una T4 o una P100) e abilita **Internet**;
3. Crea un account gratuito su [Comet](https://www.comet.com) (usato per tracciare gli esperimenti) e copia la tua **API key**;
4. Su Kaggle, apri **Add-ons > Secrets**, aggiungi una secret con label `COMET_API_KEY` e collegala al notebook.

La key viene letta tramite `kaggle_secrets`, quindi non compare mai nel codice. Per eseguire i lab altrove (Colab, macchina locale), basta sostituire quel blocco con il proprio modo di caricare la key.

Il pacchetto pip `mitdeeplearning` è una piccola libreria di utility del corso open di deep learning del MIT: i lab la usano solo per scaricare i dataset e per alcuni helper di plotting e audio.
