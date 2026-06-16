import numpy as np
import os
from scipy import signal

def generatedataset(classvec, subvec, trainprop, width = 32, overlap = 16, traindata=True, src_path="/home/rffl/scratch/THINGS-EEG/preprocessed",
                    out_path_train="./train_spec.npy", out_path_test="./test_spec.npy", single=False):
    
    paths = []
    for i in range(len(subvec)):
        sub_str = f"sub-{subvec[i]:02d}" if int(subvec[i]) < 10 else f"sub-{subvec[i]}"
        file_name = "preprocessed_eeg_training.npy" if traindata else "preprocessed_eeg_test.npy"
        paths.append(os.path.join(src_path, sub_str, file_name))
    
    data = []
    for p in paths:
        data.append(np.load(p, allow_pickle=True).item()["preprocessed_eeg_data"])

    train = []
    test = []

    print("Generating spectrograms...")
    for cls in classvec:
        for sub in range(len(subvec)):
            for i in range(10 if traindata else 1):
                run = data[sub][(10 if traindata else 1) * (cls - 1) + i] # Shape (4, 17, 100)
                
                split = trainprop * len(run)
                
                for j in range(len(run)):
                    trial_eeg = run[j, :, :]
                    
                    trial_specs = []
                    for ch in range(17):
                        f, t, spec = signal.spectrogram(trial_eeg[ch, :], fs=100, nperseg=width, noverlap=overlap)
                        trial_specs.append(spec)
                    spec_3d = np.array(trial_specs) 
                    if j < split:
                        train.append(spec_3d)
                    else:
                        test.append(spec_3d)

    train = np.array(train, dtype=np.float32)
    np.save(out_path_train, train)
    print(f"Saved {out_path_train}. Shape: {train.shape}")

    if not single:
        test = np.array(test, dtype=np.float32)
        np.save(out_path_test, test)
        print(f"Saved {out_path_test}. Shape: {test.shape}")
        return out_path_train, out_path_test
    
    return out_path_train

if __name__ == "__main__":
    classvec = [i+1 for i in range(10)]
    subvec = [i+1 for i in range(10)]
    generatedataset(classvec, subvec, trainprop=0.75, traindata=False)