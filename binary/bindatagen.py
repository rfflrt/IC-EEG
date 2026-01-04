import numpy as np

def generatetrain(c1, c2, src_path="/home/rffl/scratch/THINGS-EEG/preprocessed/sub-02/preprocessed_eeg_test.npy",
                  out_path="./bindatatrain.npy"):
    
    data = np.load(src_path, allow_pickle=True).item()["preprocessed_eeg_data"]
    out = []

    # 75-15 from training dataset (3-1)
    #for i in range(10):
    #   runs = data[10*(c1-1) + i]
    #    for rep in range(60):
    #        out.append(runs[rep, :, :])

    # 75-15 from testing dataset (60-20)
    runs = data[c1 + 1]
    for rep in range(60):
        out.append(runs[rep, :, :])

    # same for c2
    #for i in range(10):
    #    runs = data[10*(c2-1) + i]
    #    for rep in range(3):
    #        out.append(runs[rep, :, :])

    # same for c2
    runs = data[c2 + 1]
    for rep in range(60):
        out.append(runs[rep, :, :])

    out = np.asarray(out)
    np.save(out_path, out)
    print("Saved", out_path, "shape=", out.shape)
    return out_path

def generatetest(c1, c2, src_path="/home/rffl/scratch/THINGS-EEG/preprocessed/sub-02/preprocessed_eeg_test.npy",
                 out_path="./bindatatest.npy"):

    data = np.load(src_path, allow_pickle=True).item()["preprocessed_eeg_data"]
    out = []

    #for i in range(10):
    #    runs = data[10*(c1-1) + i]
    #    out.append(runs[3, :, :])

    runs = data[c1 + 1]
    for rep in range(60, 80):
        out.append(runs[rep,:,:])

    #for i in range(10):
    #    runs = data[10*(c2-1) + i]
    #    out.append(runs[3, :, :])

    runs = data[c2 + 1]
    for rep in range(60, 80):
        out.append(runs[rep,:,:])


    out = np.asarray(out)
    np.save(out_path, out)
    print("Saved", out_path, "shape=", out.shape)
    return out_path

if __name__=="__main__":
    generatetrain(1,2)