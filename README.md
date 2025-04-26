# fcnn
This is a Fourier Convolutional Neural Network for note identification.

This was a passion project for determining whether octave equivalence has analogy to systematic confusion due to similar harmonic frequencies. The FCNN trained well enough that the idea was ruled out. Confusion was mainly between close pitches, which is also intuitive. FCNNs are valuable because they do similar work to CNNs without as much computational overhead, and with more interpretability if activation functions are unnecessary.

fcnn.py is the neural architecture. Note -> discrete Fourier transform -> flattening to squared magnitude -> fully-connected layer -> softmax -> argmax.

chunking_script.py chunks piano data and adds Wiener noise with a 10:1 signal-to-noise ratio. Edit filepaths for use. Data is provided here: https://archive.org/details/media_202504

fcnn_training.py trains on piano data after it has been chunked using stochastic gradient descent with 80-20% train-test split. Edit filepaths for use. Training reached 100% accuracy before 100 epochs.

arbitrary_training.py generates and trains on synthesized notes with randomized amplitudes for each fundamental and harmonic frequency and SNR 10:1. Training reached 99.8% accuracy by 100 epochs.
