# PyThia
PyThia performs video prediction for simple physical systems:
given the first few frames in a video,
it predicts the images that will appear in the future.

## Start here!

The easiest way to see what PyThia can do is to peruse the demonstration notebook.
It is available as an [interactive notebook](
    https://colab.research.google.com/github/aremondtiedrez/pythia/blob/main/demo.ipynb
) served via Google Colab.

As you will see in that notebook, the PyThia models are trained on sequences
of images depicting a round ball bouncing around a closed box.

![Training sample](https://github.com/aremondtiedrez/pythia/blob/main/pythia/demo/sample_gifs/training_sample_1.gif)

A trained PyThia model is then capable of predicting the future trajectory:

![Prediction](https://github.com/aremondtiedrez/pythia/blob/main/pythia/demo/sample_gifs/prediction_1.gif)

## Why the name `PyThia`?
The objective of this project is to predict what will happen in the future,
more specifically predicting what will happen in the future of a video
whose initial frames are given as input.
This project is therefore named after a famous oracle purporting to divine the future:
in Ancient Greece the Pythia served as oracle at the temple of Apollo, in Delphi.
