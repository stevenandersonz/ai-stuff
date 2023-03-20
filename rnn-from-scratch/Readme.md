# VANILLA RNN FROM SCRATCH

A vanilla implemetation of a rnn using torch.tensors as building blocks. 
I hacked my way testing my backprop implementation by comparing the gradients from a real torch.RNN model (similar to what karpathy did in [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

Also I took a lot of inspiration from [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086#file-min-char-rnn-py-L80)

### Dataset
It comes from the [makemore](https://github.com/karpathy/makemore). 

### Notes
I need to refactor the code, but you know how that goes. 