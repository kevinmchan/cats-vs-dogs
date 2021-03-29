# Cats vs Dogs

An exploration of transfer learning, using a pre-trained mobilenet to learn a cat vs dog image classifier. Given mobilenet's compact size, we can also serve our model as a static webpage using tensorflowjs.

Try the model at https://catdog.azureedge.net/

## Setup

Create conda environment used for model build.
```bash
conda env create -f environment.yml
```

Create npm environment for sharing model as a static webpage using tensorflowjs.
```bash
npm install
```

## Download training dataset

```bash
mkdir data && kaggle competitions download -c dogs-vs-cats -p data  # download dataset
unzip data/dogs-vs-cats.zip -d data && unzip data/train.zip -d data  # unzip
python -m catsvsdogs.data  # sample and organize into cat and dog folders
```

## Train model

Train a keras model:

```bash
python -m catsvsdogs.model
```

Convert to a tfjs model:

```bash
tensorflowjs_converter --input_format=keras ./model/cat_dog_mobilenet_pooled_finetuned_best.h5 ./model/
```

## Serving our static webpage locally

```bash
npm run start
```
