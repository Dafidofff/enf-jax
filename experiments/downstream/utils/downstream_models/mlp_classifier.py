import flax.linen as nn


class MLPClassifier(nn.Module):
    num_hidden: int
    num_classes: int

    def setup(self):
        self.hidden1 = nn.Dense(self.num_hidden)
        self.hidden2 = nn.Dense(self.num_hidden)
        self.output = nn.Dense(self.num_classes)

    def __call__(self, latents):

        # Unpack
        p, a, window = latents

        # Flatten the num_latents dim
        a = a.reshape(a.shape[0], -1)

        x = self.hidden1(a)
        x = nn.gelu(x)
        x = self.hidden2(x)
        x = nn.gelu(x)
        x = self.output(x)
        return x
