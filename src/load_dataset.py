import pickle

def load_dataset(dataset_path:str):
    metadata = None
    occupancy = []

    with open(dataset_path, "rb") as file:
        try:
            metadata = pickle.load(file)
            while(True):
                data = pickle.load(file)
                print(type(data))
                occupancy.append(data)
        except EOFError:
            pass
    
    return metadata, occupancy

