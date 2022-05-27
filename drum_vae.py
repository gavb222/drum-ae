import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import os
import time

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*60*60, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)
        self.pool = nn.MaxPool2d(2, 2)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5, stride=2)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = self.pool(F.relu(self.encConv2(x)))
        x = x.view(-1, 32*60*60)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 60, 60)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

#load data
dataset_path = "D:/drum_samples_augmented"
n_files = 0
dataload_starttime = time.time()
for subdir, dirs, files in os.walk(dataset_path):

    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".wav"):
            if n_files == 0:
                dataset,fs = torchaudio.load(filepath)
                #print(dataset.size())
            elif n_files == 1:
                new_file,fs = torchaudio.load(filepath)
                if new_file.size()[1] > 16384:
                    new_file = new_file[:,:16384]
                #print(new_file.size())
                dataset = torch.stack((dataset,new_file))
            else:
                new_file,fs = torchaudio.load(filepath)
                if new_file.size()[1] > 16384:
                    new_file = new_file[:,:16384]
                elif new_file.size()[1] < 16384:
                    pad = torch.nn.ConstantPad1d((0,16384-new_file.size()[1]),0)
                    new_file = pad(new_file)
                dataset = torch.cat((dataset,new_file.unsqueeze(0)))

            n_files = n_files + 1

dataload_donetime = time.time()
print("finished loading: {} files loaded in {} seconds".format(n_files, dataload_donetime - dataload_starttime))

net = VAE().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=.0005)

#run the thing
n_epochs = 200
training_losses = []
train = False

n_fft = 255
hop_sz = int((n_fft+1)/2)
window_fn = torch.hann_window(n_fft).cuda()

def mean(list):
    return sum(list)/len(list)

def make_wav(mag,phase):
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.stack([real,imag], dim = -1)
    wav = torch.istft(spec.cpu(), n_fft = n_fft, hop_length = hop_sz, window = window_fn.cpu(), normalized = True)
    return wav

if train:

    print("training start!")
    for epoch_n in range(n_epochs):
        epoch_losses = []
        start_time = time.time()

        #load data in
        for j in range(dataset.size()[0]):
            #file,waveform
            input_wav = dataset[j,:].cuda()

            complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = hop_sz, window = window_fn, normalized = True)
            complex_mix_pow = complex_mix.pow(2).sum(-1)
            complex_mix_mag = torch.sqrt(complex_mix_pow)

            #TiFGAN
            complex_mix_mag = complex_mix_mag/torch.max(complex_mix_mag)
            complex_mix_mag = torch.log(complex_mix_mag)

            #now clip log mag at -10 minimum
            complex_mix_mag[complex_mix_mag < -10] = -10

            #scale to range of tanh:
            complex_mix_mag = (complex_mix_mag/10) + 1

            #not sure about the sizes of these, this might be wrong
            complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,1],complex_mix[:,:,0]))

            out, mu, logVar = net(complex_mix_mag.unsqueeze(0))

            complex_mix_loss = complex_mix_mag.squeeze()[:127,:127]

            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out.squeeze(), complex_mix_loss, size_average=False) + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()


        print("Epoch {} finished! Average Loss: {}, Total Time: {}".format(epoch_n+1,mean(epoch_losses),time.time()-start_time))

        training_losses.append(mean(epoch_losses))

    print("training finished, saving results")
    torch.save(net.state_dict(), 'drum_vae_tifgan.pth')

else:

    state = torch.load("drum_vae_tifgan.pth")
    net.load_state_dict(state)
    net.eval()

    results_path = "vae_outs_tifgan"
    #
    # generate some results
    with torch.no_grad():
        for j in range(dataset.size()[0]):
            input_wav = dataset[j,:].cuda()

            complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = hop_sz, window = window_fn, normalized = True)
            complex_mix_pow = complex_mix.pow(2).sum(-1)
            complex_mix_mag = torch.sqrt(complex_mix_pow)

            div = torch.max(complex_mix_mag)

            complex_mix_mag = complex_mix_mag/div
            complex_mix_mag = torch.log(complex_mix_mag)

            #now clip log mag at -10 minimum
            complex_mix_mag[complex_mix_mag < -10] = -10

            #scale to range of tanh:
            complex_mix_mag = (complex_mix_mag/10) + 1

            #not sure about the sizes of these, this might be wrong
            complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,:,1],complex_mix[:,:,:,0]))

            out, mu, logVar = net(complex_mix_mag.unsqueeze(0))

            p2d = (0, 1, 0, 1) # pad last dim by (0, 3) and 2nd to last by (0, 3)
            out = F.pad(out, p2d, "constant", 0)

            out = (out-1)*10
            out = torch.exp(out)
            out = out*div

            wav_out = make_wav(out.cpu().squeeze(),complex_mix_phase.cpu().squeeze())
            torchaudio.save(results_path + os.sep + "output_" + str(j) + ".wav", wav_out.cpu().squeeze(),fs)
            torchaudio.save(results_path + os.sep + "input_" + str(j) + ".wav", input_wav.cpu().squeeze(),fs)
    #save state_dicts
    print("results generated, saving model")
