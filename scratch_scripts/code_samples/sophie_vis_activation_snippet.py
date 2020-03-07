# From Sophie (Thanks)

from keras import models
layer_outputs = [layer.output for layer in x.layers]
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=x.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

# (where x is my model that i’ve already trained)
# get the activation layers
activations = activation_model.predict(image)

# look at the shape for each layer to get an idea of what is going on (example with second layer)
second = activations[1]
print(second.shape)

my result for the one image = (1, 256, 256, 128)
# View a filter of choice at a particular activation layer (the 4th dimension of the tensor relates to a filter…. i believe… so you can chose which one you wanna look at)

plt.matshow(activations[2][0, :, :, 0], cmap=‘viridis’)