import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import os
import time

def split_wav(wavData):
    complex_mix = torch.stft(wavData, n_fft = n_fft, hop_length = hop_sz, window = window_fn)

    complex_mix_pow = complex_mix.pow(2).sum(-1)
    complex_mix_mag = torch.sqrt(complex_mix_pow)

    complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,1],complex_mix[:,:,0]))
    complex_mix_combo = torch.stack([complex_mix_mag,complex_mix_phase], dim = -1)

    n_splits = (complex_mix_combo.size()[1]//hop_sz)

    complex_mix_combo = complex_mix_combo[:,0:n_splits*512,:]

    chunks = torch.chunk(complex_mix_combo,n_splits,1)
    stack = torch.stack(chunks,dim = 0)
    fs_global = fs
    return stack

class Linear_Block(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=True):
        super(Linear_Block, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        if self.activation:
            out = self.fc(F.tanh(x))
        else:
            out = self.fc(x)

        return out

#could use LSTM here before output
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.fc1 = Linear_Block(input_channels, 128)
        self.fc2 = Linear_Block(128,20)

    def forward(self,x):
        #NB do TiFGAN before putting in and after pulling out
        x = self.fc1(x)
        x = self.fc2(x)
        out = nn.Sigmoid()(x)
        return out

class Decoder(nn.Module):
    def __init__(self, input_channels):
        super(Decoder, self).__init__()

        self.fc1 = Linear_Block(20, 128)
        self.fc2 = Linear_Block(128,input_channels)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        #what size is this?
        out = torch.tanh(x)
        return out

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

#init stuff
E = Encoder(512)
D = Decoder(512)

E.cuda()
D.cuda()

E.train()
D.train()

loss_fn = nn.MSELoss()

E_optim = torch.optim.Adam(E.parameters(), lr = .0001, betas = (.5,.999))
D_optim = torch.optim.Adam(D.parameters(), lr = .0001, betas = (.5,.999))

#run the thing
keep_training = True
counter = 0
training_losses = []

n_fft = 1023
hop_sz = int((n_fft+1)/4)
window_fn = torch.hann_window(n_fft).cuda()

def mean(list):
    return sum(list)/len(list)

def make_wav(mag,phase):
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.stack([real,imag], dim = -1)
    wav = torch.istft(spec.cpu(), n_fft = n_fft, hop_length = hop_sz, window = window_fn.cpu(), normalized = True)
    return wav

print("training start!")
while keep_training:
    epoch_losses = []
    start_time = time.time()

    #load data in
    for j in range(dataset.size()[0]):
        #file,waveform
        input_wav = dataset[j,:].cuda()

        complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = hop_sz, window = window_fn, normalized = True)
        complex_mix_pow = complex_mix.pow(2).sum(-1)
        complex_mix_mag = torch.sqrt(complex_mix_pow)
        #not sure about the sizes of these, this might be wrong
        complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,1],complex_mix[:,:,0]))

        #TiFGAN
        complex_mix_mag = complex_mix_mag/torch.max(complex_mix_mag)
        complex_mix_mag = torch.log(complex_mix_mag)

        #now clip log mag at -10 minimum
        complex_mix_mag[complex_mix_mag < -10] = -10

        #scale to range of tanh:
        complex_mix_mag = (complex_mix_mag/10) + 1
        #/TiFGAN

        #check shapes of all of these things!
        #for every frame in complex_mix_mag do:
        #print(complex_mix_mag.size())
        for i in range(complex_mix_mag.squeeze().size()[1]):
            net_input = complex_mix_mag.squeeze()[:,i]

            E.zero_grad()
            D.zero_grad()

            encoding = E(net_input)
            output = D(encoding)

            loss = loss_fn(output,net_input)
            loss.backward()

            E_optim.step()
            D_optim.step()

            epoch_losses.append(loss.item())

    print("Epoch {} finished! Average Loss: {}, Total Time: {}".format(counter,mean(epoch_losses),time.time()-start_time))

    if counter > 2:
        if mean(training_losses[-2:]) < mean(epoch_losses):
            keep_training = False
            print("training finished!")

    training_losses.append(mean(epoch_losses))
    counter = counter + 1

print("saving results")
torch.save(E.state_dict(), 'Enc_param_sm.pth')
torch.save(D.state_dict(), 'Dec_param_sm.pth')

#state = torch.load("model_param_2.pth")
#model.load_state_dict(state)
E_state = torch.load("Enc_param_sm.pth")
D_state = torch.load("Dec_param_sm.pth")

E.load_state_dict(E_state)
D.load_state_dict(D_state)

E.eval()
D.eval()

results_path = "roche_output_smallenc"

#generate some results
with torch.no_grad():
    for j in range(dataset.size()[0]):
        input_wav = dataset[j,:].cuda()

        complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = hop_sz, window = window_fn)
        #print(complex_mix.size())
        complex_mix_pow = complex_mix.pow(2).sum(-1)
        complex_mix_mag = torch.sqrt(complex_mix_pow)
        #not sure about the sizes of these, this might be wrong
        complex_mix_phase = torch.tan(torch.div(complex_mix[:,:,:,1],complex_mix[:,:,:,0]))


        #TiFGAN
        complex_mix_mag = complex_mix_mag/torch.max(complex_mix_mag)
        complex_mix_mag = torch.log(complex_mix_mag)

        #now clip log mag at -10 minimum
        complex_mix_mag[complex_mix_mag < -10] = -10

        #scale to range of tanh:
        complex_mix_mag = (complex_mix_mag/10) + 1
        #/TiFGAN

        #check shapes of all of these things!
        #for every frame in complex_mix_mag do:
        #print(complex_mix_mag.size())

        cols = []
        for i in range(complex_mix_mag.squeeze().size()[1]):
            net_input = complex_mix_mag.squeeze()[:,i]

            encoding = E(net_input)
            output = D(encoding)

            cols.append(output)

        output_logscale = torch.stack(cols)
        output_logscale = output_logscale.permute(1,0)

        magspec_out = (output_logscale - 1)*10
        magspec_out = torch.exp(magspec_out)
        magspec_out = magspec_out * torch.max(complex_mix_mag.cpu())

        #print(magspec_out.squeeze().size())
        #print(complex_mix_phase.squeeze().size())

        wav_out = make_wav(magspec_out.squeeze(),complex_mix_phase.squeeze())
        torchaudio.save(results_path + os.sep + "output_" + str(j) + ".wav",wav_out.squeeze(),fs)
#save state_dicts
print("results generated, saving model")
