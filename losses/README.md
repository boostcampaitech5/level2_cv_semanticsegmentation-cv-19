## Various Loss fns


- Various loss fns are used in medical image segmentation. 
- Basic loss implemenations are from aistages notice board. 
- Additional losses(e.g. Tversky, LovaszHinge) are from 
<href>https://www.kaggle.com/code/sungjunghwan/loss-function-of-image-segmentation

<br>

### Usage Tips
In my experience testing and debugging these losses, 
I have some observations that may be useful to beginners experimenting with different loss functions. 
These are not rules that are set in stone; they are simply my findings and your results may vary.

- Tversky and Focal-Tversky loss benefit from very low learning rates, 
of the order 5e-5 to 1e-4. They would not see much improvement in my kernels until around 7-10 epochs, 
upon which performance would improve significantly.

- In general, if a loss function does not appear to be working well (or at all), 
experiment with modifying the learning rate before moving on to other options.

- You can easily create your own loss functions by combining any of the above 
with Binary Cross-Entropy or any combination of other losses. 
Bear in mind that loss is calculated for every batch, so more complex losses will increase runtime.

- Care must be taken when writing loss functions for PyTorch. 
If you call a function to modify the inputs that doesn't entirely use PyTorch's numerical methods, 
the tensor will 'detach' from the the graph that maps it back through the neural network 
for the purposes of backpropagation, making the loss function unusable.
<br>

### Data, Target shapes during training

At forward path, (bs=8)

- input  : torch.Size([8, 3, 512, 512])
- output : torch.Size([8, 29, 512, 512])
- target : torch.Size([8, 29, 512, 512])
