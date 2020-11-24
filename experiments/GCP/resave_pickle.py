import pickle
import numpy as np

# load synthesizer from saved object
with open('/home/rzhang_dal/project/TVAE_synthesizer.pkl', 'rb') as input:
    synthesizer = pickle.load(input)

# check out sample
sampled = synthesizer.sample(3)
np.set_printoptions(suppress = True, precision = 2)
print(sampled)

# resave the synthesizer
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # overwrite any existing file
        pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)

save_object(synthesizer, '/home/rzhang_dal/project/TVAE_synthesizer_1.pkl')
