import numpy as np
import os

def generatedataset(classvec, subject, trainprop, traindata = True, src_path="/home/rffl/scratch/THINGS-EEG/preprocessed",
                    out_path_train="./kclasstrain.npy", out_path_test="./kclasstest.npy"):
    
    fullpath = os.path.join(src_path, ("sub-0" + str(subject)) if subject < 10 else ("sub-" + str(subject)),
                            "preprocessed_eeg_training.npy" if traindata else "preprocessed_eeg_test.npy") if src_path[-3:-1] != "npy" else src_path
    
    data = np.load(fullpath, allow_pickle=True).item()["preprocessed_eeg_data"]

    train = []
    test = []

    for cls in classvec:
        for i in range(10 if traindata else 1):
            run = data[(10 if traindata else 1) * (cls - 1) + i]
            split = trainprop * len(run)
            for j in range(len(run)):
                if(j < split):
                    train.append(run[j,:,:])
                else:
                    test.append(run[j,:,:])

    train = np.array(train)
    test = np.array(test)
    
    np.save(out_path_train, train)
    np.save(out_path_test, test)
    print("Saved", out_path_train, "shape =", train.shape)
    print("Saved", out_path_test, "shape =", test.shape)

    return out_path_train, out_path_test     